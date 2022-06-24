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
		if not np.isclose(best_vec.X @ np.linalg.inv(alg.Vs[t]) @ best_vec.X,ci_vec.X @ ci_vec.X):
			breakpoint()
		as_gurobi = np.squeeze(np.asarray(A_inv @ gurobi_val))


		eigen_ci = np.squeeze(np.asarray(alg.beta * np.diag(np.sqrt(curr_eigenvectors.T @ np.linalg.inv(alg.Vs[t]) @ curr_eigenvectors))))
		eigen_mu = np.squeeze(np.asarray(curr_eigenvectors.T @ alg.debug_theta[t]))



		ucb_values = eigen_ci + abs(eigen_mu)


		coef_rankings  = rankdata(abs(as_gurobi), method="ordinal")
		ucb_rankings = rankdata(ucb_values, method="ordinal")

		
		if len(np.unique(np.round(curr_eigenvalues, decimals = 7))) == 4:
			if abs(np.sum(eigen_mu * as_gurobi) +  alg.beta * np.sqrt(np.sum(np.square(as_gurobi)/curr_eigenvalues)) - model.ObjVal) > 1e-5:
				breakpoint()
		# assert(np.square(as_gurobi)[0] * curr_eigenvectors[:, 0].T @ np.linalg.inv(alg.Vs[t]) @ curr_eigenvectors[:, 0] + np.square(as_gurobi)[1] * curr_eigenvectors[:, 1].T @ np.linalg.inv(alg.Vs[t]) @ curr_eigenvectors[:, 1] + np.square(as_gurobi)[2] * curr_eigenvectors[:, 2].T @ np.linalg.inv(alg.Vs[t]) @ curr_eigenvectors[:, 2] + np.square(as_gurobi)[3] * curr_eigenvectors[:, 3].T @ np.linalg.inv(alg.Vs[t]) @ curr_eigenvectors[:, 3])
		
		assert((abs(np.squeeze(curr_eigenvectors[:, 0] * as_gurobi[0] + curr_eigenvectors[:, 1] * as_gurobi[1] + curr_eigenvectors[:, 2] * as_gurobi[2] + curr_eigenvectors[:, 3] * as_gurobi[3]) - gurobi_val) < 1e-5).all())
		if len(np.unique(np.round(curr_eigenvalues, decimals = 7))) == 4 and not (coef_rankings == ucb_rankings).all():
			print(t)
			if t > 500:
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