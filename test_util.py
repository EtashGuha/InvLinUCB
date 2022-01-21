import time
from algorithms import UCB, LinUCB
from alg_util import train_alg_UCB
from estimators import Baseline1, Baseline2, estimate_linucb_means_lp, estimate_ucb_means_lp, Baseline2_LP
import numpy as np
from sklearn.metrics import mean_squared_error
from oracle import Oracle
from estimator_util import initialize_taus_np_optarm

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
    theta_estimate = estimate_linucb_means_lp(alg, timelimit=timelimit)
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