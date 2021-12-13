import time
from algorithms import UCB, LinUCB
from alg_util import train_alg_UCB
from estimators import Baseline1, Baseline2, estimate_linucb_means_lp, estimate_ucb_means_lp
import numpy as np
from sklearn.metrics import mean_squared_error
from oracle import Oracle


def normalize_subopt(means):
    means = np.asarray(means)
    best_value = max(means)
    return best_value - means

def test_Baseline1(theta, action_set, T=1000):
    dim = theta.shape[-1]
    oracle = Oracle(theta)
    alg = UCB(action_set, T=T, dim=dim)
    train_alg_UCB(alg, T, theta, oracle)
    
    t1 = time.time()
    estimate_sample_means = normalize_subopt(Baseline1(alg))
    t2 = time.time()
    
    return mean_squared_error(normalize_subopt(alg.sample_means), estimate_sample_means), t2 - t1
    
def test_Baseline2(theta, action_set, T=1000):
    dim = theta.shape[-1]
    oracle = Oracle(theta)
    alg = UCB(action_set, T=T, dim=dim)
    train_alg_UCB(alg, T, theta, oracle)
    
    t1 = time.time()
    estimate_sample_means = normalize_subopt(Baseline2(alg))
    t2 = time.time()
    return mean_squared_error(normalize_subopt(alg.sample_means), estimate_sample_means), t2 - t1


def test_UCB(theta, action_set, T=1000):
    dim = theta.shape[-1]
    oracle = Oracle(theta)
    alg = UCB(action_set, T=T, dim=dim)
    train_alg_UCB(alg, T, theta, oracle)
    
    t1 = time.time()
    lp_vals = normalize_subopt(estimate_ucb_means_lp(alg))
    t2 = time.time()
    return mean_squared_error(normalize_subopt(alg.sample_means), lp_vals), t2 - t1
    
def test_LinUCB(theta, action_set, T=1000):
    
    oracle = Oracle(theta)
    dim = theta.shape[-1]
    alg = LinUCB(action_set, dim=dim, T=T)
    
    train_alg_UCB(alg, T, theta, oracle)
    t1 = time.time()
    true_means = normalize_subopt(action_set @ alg.hat_theta)
    theta_estimate = estimate_linucb_means_lp(alg)
    estimate_means = normalize_subopt(action_set @ theta_estimate)
    t2 = time.time()
    return mean_squared_error(true_means, estimate_means), t2 - t1