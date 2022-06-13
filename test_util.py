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

	eigenvalues, eigenvectors = np.linalg.eig(alg.Vs[-1])

	A_inv = np.linalg.inv(eigenvectors)
	worst_arms = []
	all_ucb_values = []

	for t in range(alg.T - 1):
		ucb_values = []

		#JUST CHECKING
		# eigenvalues, eigenvectors = np.linalg.eigh(alg.Vs[t])
		for idx, e_l in enumerate(eigenvalues):
			mu = alg.debug_theta[t].T @ eigenvectors[:, idx]
			if mu < 0:
				eigenvectors[:, idx] = -1 * eigenvectors[:, idx]
				eigenvalues[idx] = -1 * eigenvalues[idx]

			
		A_inv = np.linalg.inv(eigenvectors)


		linear_solution = A_inv @ alg.actions[t].T
		for idx, e_l in enumerate(eigenvalues):
			ucb_l = alg.debug_theta[t].T @ eigenvectors[:, idx] + alg.beta * np.diag(np.sqrt(eigenvectors[:, idx].T @ np.linalg.inv(alg.Vs[t]) @ eigenvectors[:, idx]))
			ucb_values.append(ucb_l.item())
		picked_value = alg.debug_theta[t].T @ alg.actions[t].T + alg.beta * np.sqrt(np.diag(alg.actions[t] @ np.linalg.inv(alg.Vs[t]) @ alg.actions[t].T))

		estimated_mean = 0
		for idx in range(linear_solution.shape[0]):
			estimated_mean += linear_solution[idx] * alg.debug_theta[t].T @ eigenvectors[:, idx]

		

		all_ucb_values.append(ucb_values)
		coef_rankings  = rankdata(linear_solution, method="ordinal")
		ucb_rankings = rankdata(ucb_values, method="ordinal")
		
		import copy
		if not (coef_rankings == ucb_rankings).all():
			copy_vec = copy.deepcopy(linear_solution)
			for blah_idx in range(alg.dim):
				copy_vec[blah_idx] = linear_solution[np.where(coef_rankings == ucb_rankings[blah_idx])]
			import gurobipy as gp

			model = gp.Model()
			best_vec = model.addMVar(copy_vec.shape[0])
			model.params.NonConvex = 2
			model.setObjective(alg.debug_theta[t].T @ best_vec +  best_vec @ np.linalg.inv(alg.Vs[t]) @ best_vec * alg.beta, gp.GRB.MAXIMIZE)
			model.addConstr(np.sum(best_vec @ best_vec) <= 1)
			model.optimize()
			gurobi_val = np.asarray(model.x)
			
			theoretical_vec = eigenvectors @ copy_vec
			simply_positive_vec = eigenvectors @ abs(linear_solution)

			positive_copy_vec = copy.deepcopy(linear_solution)
			for blah_idx in range(alg.dim):
				positive_copy_vec[blah_idx] = abs(linear_solution)[np.where(coef_rankings == ucb_rankings[blah_idx])]

			positive_swapped_vec = eigenvectors @ positive_copy_vec

			positive_swapped_vec_value =  alg.debug_theta[t].T @ positive_copy_vec + alg.beta * np.sqrt(np.diag(positive_copy_vec.T @ np.linalg.inv(alg.Vs[t]) @ positive_copy_vec))
			positive_value =  alg.debug_theta[t].T @ simply_positive_vec + alg.beta * np.sqrt(np.diag(simply_positive_vec.T @ np.linalg.inv(alg.Vs[t]) @ simply_positive_vec))
			better_value = alg.debug_theta[t].T @ theoretical_vec + alg.beta * np.sqrt(np.diag(theoretical_vec.T @ np.linalg.inv(alg.Vs[t]) @ theoretical_vec))

			other_estimated_mean = 0
			for idx in range(copy_vec.shape[0]):
				other_estimated_mean += copy_vec[idx] * alg.debug_theta[t].T @ eigenvectors[:, idx]

			if (better_value > picked_value and abs(better_value - picked_value) > 1e-5).all() or (positive_value > picked_value and abs(positive_value - picked_value) > 1e-5).all():
				better_value *= 1
			else:
				print("Breaking Time: {}".format(t))
				other_ci = alg.beta * np.sqrt(np.diag(theoretical_vec.T @ np.linalg.inv(alg.Vs[t]) @ theoretical_vec))
				picked_ci = alg.beta * np.sqrt(np.diag(alg.actions[t] @ np.linalg.inv(alg.Vs[t]) @ alg.actions[t].T))
				print(picked_ci - other_ci)

				other_ucb = theoretical_vec.T @ alg.debug_theta[t] + other_ci
				picked_ucb = alg.actions[t] @ alg.debug_theta[t] + picked_ci
				curr_A_inv = np.linalg.inv(eigenvectors)

				as_gurobi = np.asarray(curr_A_inv @ gurobi_val)

				eigen_ci = alg.beta * np.diag(np.sqrt(eigenvectors.T @ np.linalg.inv(alg.Vs[t]) @ eigenvectors))
				eigen_mu = eigenvectors.T @ alg.debug_theta[t]
				
				as_theoretical = np.squeeze(np.asarray(curr_A_inv @ theoretical_vec))
				as_picked = np.squeeze(np.asarray(curr_A_inv @ alg.actions[t].T))
				theoretical_inside = np.divide(np.square(as_theoretical), eigenvalues)
				picked_inside = np.divide(np.square(as_picked), eigenvalues)

				assert((abs(eigenvectors[:, 0] * as_theoretical[0] + eigenvectors[:, 1] * as_theoretical[1] + eigenvectors[:, 2] * as_theoretical[2] + eigenvectors[:, 3] * as_theoretical[3] - theoretical_vec) < 1e-5).all())

				breakpoint()
		
		worst_arms.append(np.argmin(ucb_values))
	import matplotlib.pyplot as plt
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