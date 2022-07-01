import time
from algorithms import UCB, LinUCB
from alg_util import train_alg_UCB
from estimators import Baseline1, Baseline2, estimate_linucb_means_lp, estimate_ucb_means_lp, Baseline2_LP, estimate_linucb_means_simple
import numpy as np
from sklearn.metrics import mean_squared_error
from oracle import Oracle
from estimator_util import initialize_taus_np_optarm
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from operator import add
import cvxpy as cp
import gurobipy as gp
import math
import matplotlib.pyplot as plt
def normalize_subopt(means):
	means = np.asarray(means)
	best_value = max(means)
	return best_value - means

def test_Baseline1(theta, action_set, sigma, T=1000):
	dim = theta.shape[-1]
	oracle = Oracle(theta, sigma=sigma)
	alg = UCB(action_set, T=T, dim=dim)
	train_alg_UCB(alg, T, theta, oracle)
	
	t1 = time.time()
	estimate_sample_means = normalize_subopt(Baseline1(alg))
	t2 = time.time()
	
	return mean_squared_error(normalize_subopt(alg.sample_means), estimate_sample_means), t2 - t1
	
def test_Baseline2(theta, action_set, sigma, T=1000):
	dim = theta.shape[-1]
	oracle = Oracle(theta, sigma=sigma)
	alg = UCB(action_set, T=T, dim=dim)
	train_alg_UCB(alg, T, theta, oracle)
	
	t1 = time.time()
	estimate_sample_means = normalize_subopt(Baseline2(alg))
	t2 = time.time()
	return mean_squared_error(normalize_subopt(alg.sample_means), estimate_sample_means), t2 - t1


def test_UCB(theta, action_set, sigma, T=1000, timelimit=None):
	dim = theta.shape[-1]
	oracle = Oracle(theta, sigma=sigma)
	alg = UCB(action_set, T=T, dim=dim)
	train_alg_UCB(alg, T, theta, oracle)
	original_lp_vals = estimate_ucb_means_lp(alg, timelimit=timelimit)
	t1 = time.time()
	lp_vals = normalize_subopt(original_lp_vals)
	t2 = time.time()
	# print("top")
	# print(lp_vals)
	# print(alg.sample_means)
	# print(normalize_subopt(alg.sample_means))
	return mean_squared_error(normalize_subopt(alg.sample_means), lp_vals), t2 - t1, is_baseline_3_feasible_baseline_4(alg, original_lp_vals), is_baseline_3_feasible_baseline_4(alg, alg.sample_means)
	
def test_LinUCB(theta, action_set, sigma, T=1000, timelimit=None):
	oracle = Oracle(theta, sigma=sigma)
	dim = theta.shape[-1]
	alg = LinUCB(action_set, dim=dim, T=T)
	
	train_alg_UCB(alg, T, theta, oracle)

	t1 = time.time()
	true_means = normalize_subopt(action_set @ alg.hat_theta)

	# eigenvalues, eigenvectors = np.linalg.eig(alg.Vs[-1])

	# A_inv = np.asarray(np.linalg.inv(eigenvectors))
	worst_arms = []
	all_ucb_values = []
	list_of_discrepancies = []
	ts = []
	as_gurobis = []
	ucb_rankings_list = []
	coef_kinks = []
	ucb_kinks = []
	coef_kink_vals = []
	ucb_kink_vals = []
	all_ucb_vals = []
	for t in range(alg.T - 1):
		curr_eigenvalues, curr_eigenvectors = np.linalg.eigh(alg.Vs[t])
		def calc_final_val(as_vals):
			weightings =np.squeeze(np.asarray(curr_eigenvectors @  np.squeeze(np.asarray(as_vals))))
			reward = alg.debug_theta[t].T @ weightings 
			ci = alg.beta * np.sqrt(weightings @ np.linalg.inv(alg.Vs[t]) @ weightings)
			return reward + ci, reward, ci
		A_inv = np.linalg.inv(curr_eigenvectors)

		
		model = gp.Model()
		best_vec = model.addMVar(alg.actions[t].shape[1])
		ci_vec = model.addMVar(1)
		model.params.LogToConsole = 0

		model.params.NonConvex = 2
		model.params.FeasibilityTol = 1e-9
		model.setObjective(alg.debug_theta[t].T @ best_vec + ci_vec * alg.beta, gp.GRB.MAXIMIZE)
		model.addConstr(ci_vec @ ci_vec == best_vec @ np.linalg.inv(alg.Vs[t]) @ best_vec)
		model.addConstr(np.sum(best_vec @ best_vec) <= 1)
		model.optimize()
		
		gurobi_val = np.asarray(best_vec.x)
		# if not np.isclose(best_vec.X @ np.linalg.inv(alg.Vs[t]) @ best_vec.X,ci_vec.X @ ci_vec.X):
		# 	breakpoint()
		as_gurobi = np.squeeze(np.asarray(A_inv @ gurobi_val))


		eigen_ci = np.squeeze(np.asarray(alg.beta * np.diag(np.sqrt(curr_eigenvectors.T @ np.linalg.inv(alg.Vs[t]) @ curr_eigenvectors))))
		eigen_mu = np.squeeze(np.asarray(curr_eigenvectors.T @ alg.debug_theta[t]))



		ucb_values = eigen_ci + abs(eigen_mu)


		coef_rankings  = rankdata(abs(as_gurobi), method="ordinal")
		ucb_rankings = rankdata(ucb_values, method="ordinal")
		
		if t > 0:
			if (coef_rankings != prev_coef_rankings).any():
				coef_kinks.append(t)
				coef_kink_vals.append(coef_rankings)
			if (ucb_rankings != prev_ucb_rankings).any():
				ucb_kinks.append(t)
				ucb_kink_vals.append(ucb_rankings)
			
			all_ucb_vals.append(ucb_rankings)

		if len(coef_kink_vals) > 0 and not (coef_rankings == coef_kink_vals[-1]).all():
			breakpoint()
		# if t > 4 and (coef_kink_vals[-1] == coef_kink_vals[-3]).all() and (coef_kink_vals[-2] == coef_kink_vals[-4]).all():
		# 	coef_kink_vals = coef_kink_vals[:-2]
		# 	coef_kinks = coef_kinks[:-2]
		# if t > 4 and (ucb_kink_vals[-1] == ucb_kink_vals[-3]).all() and (ucb_kink_vals[-2] == ucb_kink_vals[-4]).all():
		# 	ucb_kink_vals = ucb_kink_vals[:-2]
		# 	ucb_kinks = ucb_kinks[:-2]
		
		prev_coef_rankings = coef_rankings
		
		prev_ucb_rankings = ucb_rankings
		
		as_gurobis.append(as_gurobi)
		ucb_rankings_list.append(ucb_values)
	
	ucb_pointer = 0
	min_coef_pointer = 0
	mappings = []
	while ucb_pointer < len(ucb_kink_vals):
		coef_pointer = min_coef_pointer 
		set_val = False
		while coef_pointer < len(coef_kink_vals):
			if (coef_kink_vals[coef_pointer] == ucb_kink_vals[ucb_pointer]).all() and ucb_kinks[ucb_pointer - 1] < coef_kinks[coef_pointer]:
				mappings.append(coef_kinks[coef_pointer])
				min_coef_pointer = coef_pointer
				set_val = True
				break
			coef_pointer += 1
		if not set_val:
			mappings.append(None)
		ucb_pointer += 1
		
	reverse_mappings = []
	ucb_pointer = len(ucb_kink_vals) - 1
	coef_pointer = len(coef_kink_vals) - 1
	max_ucb_pointer = len(ucb_kink_vals) - 1
	while coef_pointer > -1:
		ucb_pointer = max_ucb_pointer 
		set_val = False
		while ucb_pointer > -1:

			if (coef_kink_vals[coef_pointer] == ucb_kink_vals[ucb_pointer]).all() and (coef_pointer == len(coef_kink_vals) - 1 or coef_kinks[coef_pointer + 1] > ucb_kinks[ucb_pointer]):
				reverse_mappings.append(ucb_kinks[ucb_pointer])
				max_ucb_pointer = ucb_pointer
				set_val = True
				break
			ucb_pointer -= 1
		if not set_val:
			reverse_mappings.append(None)
		coef_pointer -= 1
	reverse_mappings.reverse()



	bad_mappings = []
	for idx, mapping in enumerate(mappings):
		if mapping is None:
			try:
				if idx > 0  and idx < len(mappings) - 1 and (mappings[idx - 1] != mappings[idx + 1] or mappings[idx + 1] is None):
					bad_mappings.append(idx)
			except:
				breakpoint()

	
	bad_mappings_r = []
	for idx, mapping in enumerate(reverse_mappings):
		if mapping is None:
			if idx > 0  and idx < len(reverse_mappings) - 1 and (reverse_mappings[idx - 1] != reverse_mappings[idx + 1] or reverse_mappings[idx + 1] is None):
				bad_mappings_r.append(idx)
	breakpoint()
	breakpoint()
	def get_cmap(n, name='hsv'):
		'''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
		RGB color; the keyword argument name must be a standard mpl colormap name.'''
		return plt.cm.get_cmap(name, n)
	dimension = 4
	as_gurobis = np.asarray(as_gurobis) * alg.beta
	ucb_rankings_list = np.asarray(ucb_rankings_list)
	cmap = get_cmap(dimension * 2)
	
	for dim in range(dimension):
		
		plt.plot(list(range(alg.T - 1)),abs(as_gurobis[:, dim]), label="Alpha_{}".format(dim), color=cmap(2 * dim + 1))
		plt.plot(list(range(alg.T - 1)),ucb_rankings_list[:, dim], '--', label="UCB_{}".format(dim), color=cmap(2 * dim + 2))
		
	plt.xlabel("Times")
	plt.ylabel("Discrepancies")
	plt.legend()

	plt.savefig("effort.png")
	breakpoint()
					# if abs(true_ucb_val - swapped_ucb_val) > .05:
					# 	breakpoint()
		
		# if len(np.unique(np.round(curr_eigenvalues, decimals = 7))) == 4:
		# 	if abs(np.sum(eigen_mu * as_gurobi) +  alg.beta * np.sqrt(np.sum(np.square(as_gurobi)/curr_eigenvalues)) - model.ObjVal) > 1e-5:
		# 		breakpoint()
		# # assert(np.square(as_gurobi)[0] * curr_eigenvectors[:, 0].T @ np.linalg.inv(alg.Vs[t]) @ curr_eigenvectors[:, 0] + np.square(as_gurobi)[1] * curr_eigenvectors[:, 1].T @ np.linalg.inv(alg.Vs[t]) @ curr_eigenvectors[:, 1] + np.square(as_gurobi)[2] * curr_eigenvectors[:, 2].T @ np.linalg.inv(alg.Vs[t]) @ curr_eigenvectors[:, 2] + np.square(as_gurobi)[3] * curr_eigenvectors[:, 3].T @ np.linalg.inv(alg.Vs[t]) @ curr_eigenvectors[:, 3])
		
		# assert((abs(np.squeeze(curr_eigenvectors[:, 0] * as_gurobi[0] + curr_eigenvectors[:, 1] * as_gurobi[1] + curr_eigenvectors[:, 2] * as_gurobi[2] + curr_eigenvectors[:, 3] * as_gurobi[3]) - gurobi_val) < 1e-5).all())
		# if len(np.unique(np.round(curr_eigenvalues, decimals = 7))) == 4 and not (coef_rankings == ucb_rankings).all():
		# 	print(t)
		# 	if t > 500:
		# 		breakpoint()
		
		# worst_arms.append(np.argmin(ucb_values))
	
	all_ucb_values = np.asarray(all_ucb_values)
	breakpoint()
	for i in range(all_ucb_values.shape[1]):
		plt.plot(list(range(alg.T - 1)), all_ucb_values[:, i], label=i)
	plt.legend()
	plt.savefig("movement_of_arms.png")
	breakpoint()
	taus = []
	for i in range(alg.dim):
		if i not in worst_arms:
			taus.append(alg.T)
		else:
			taus.append(max(index for index, item in enumerate(worst_arms) if item != i))
	breakpoint()


	# theta_estimate, all_answers = estimate_linucb_means_lp(alg, timelimit=timelimit)
	
	# # best_arm_idx = np.argmax(alg.arm @ theta.T)

	# similarity_to_best_arm = []
	# for t in range(1, alg.T):
	#     print(t)
	#     similarity_to_best_arm.append(np.squeeze(np.dot(all_answers[t].T, all_answers[t - 1].T)/(np.linalg.norm(all_answers[t]) * np.linalg.norm(all_answers[t - 1].T))))
	# # plt.plot(list(range(1, alg.T)), np.squeeze(np.asarray(similarity_to_best_arm)))
	# # plt.savefig("new_usvthem.png")
	
	all_actions = np.asarray(alg.actions).squeeze()
	projection_vals = []
	eigen_values, eigenvectors = np.linalg.eigh(alg.Vs[-1])



	for i in range(alg.dim):  
		total_sum = 0
		total_direction = 0
		for t in range(alg.T):
			alpha = alg.debug_reward[t] / np.dot(alg.actions[t], eigenvectors[:, i])
			
			# if alpha <= 0:
			#     breakpoint()
			total_sum += alpha
		actual_val = np.dot(alg.hat_theta.T, eigenvectors[:, i])
		breakpoint()
		

	projection_vals = np.asarray(projection_vals)
	for i in range(theta.shape[0]):
		plt.plot(list(range(alg.T - 1)), projection_vals[:, i], label=i)
	plt.legend()
	plt.savefig("projected_num_100.png")


	breakpoint()

	if theta_estimate is None:
		return None, None
	estimate_means = normalize_subopt(action_set @ theta_estimate)
	t2 = time.time()
	try:
		return mean_squared_error(true_means, estimate_means), t2 - t1
	except:
		print(theta_estimate)
		print(true_means)
		print(estimate_means)

def test_LinUCB_simple(theta, action_set, sigma, T=1000, timelimit=None):
	oracle = Oracle(theta, sigma=sigma)
	dim = theta.shape[-1]
	alg = LinUCB(action_set, dim=dim, T=T)
	
	train_alg_UCB(alg, T, theta, oracle)
	t1 = time.time()
	true_means = normalize_subopt(action_set @ alg.hat_theta)
	theta_estimate = estimate_linucb_means_simple(alg, timelimit=timelimit)

	theta_estimate = theta_estimate/np.linalg.norm(theta_estimate) * np.linalg.norm(alg.hat_theta)

	if theta_estimate is None:
		return None, None
	theta_estimate = np.asarray(theta_estimate).squeeze(axis=0)
	estimate_means = normalize_subopt(action_set @ theta_estimate)
	
	t2 = time.time()
	return mean_squared_error(true_means, estimate_means.T), t2 - t1
	

def test_Baseline2_LP(theta, action_set, sigma, T=1000, timelimit=None):
	dim = theta.shape[-1]
	oracle = Oracle(theta, sigma=sigma)
	alg = UCB(action_set, T=T, dim=dim)
	train_alg_UCB(alg, T, theta, oracle)
	
	t1 = time.time()
	original_vals = Baseline2_LP(alg, timelimit=timelimit)
	if original_vals is None:
		return None, None
	lp_vals = normalize_subopt(original_vals)
	t2 = time.time()
	return mean_squared_error(normalize_subopt(alg.sample_means), lp_vals), t2 - t1
	
def is_baseline_3_feasible_baseline_4(alg, lp_vals):
	
	optimal_arm, taus, num_pulls, tau_bars = initialize_taus_np_optarm(alg)

	

	values_of_constraints = []
	for idx, tau in enumerate(taus):
		try:
			if idx is not optimal_arm:
				values_of_constraints.append(lp_vals[idx] - lp_vals[optimal_arm] >= UCB.gcb(alg.T, alg.alpha, num_pulls[optimal_arm][tau-1]) - UCB.gcb(alg.T, alg.alpha, num_pulls[idx][tau-1]))
				values_of_constraints.append(lp_vals[idx] - lp_vals[optimal_arm] <= UCB.gcb(alg.T, alg.alpha, num_pulls[optimal_arm][tau_bars[idx] - 1]) - UCB.gcb(alg.T, alg.alpha, num_pulls[idx][tau_bars[idx] - 1]))
		except:
			print(num_pulls[optimal_arm][tau-1])
			print(num_pulls[optimal_arm][tau])

			print(num_pulls[idx][tau-1])

	return all(values_of_constraints)