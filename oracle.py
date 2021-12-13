import numpy as np
from scipy.stats import truncnorm

class Oracle:
    def __init__(self, theta, upper=1, lower=0):
        self.theta = theta
        self.upper = upper
        self.lower = lower
        
    def compute_reward(self, action):
        mu = np.inner(action, self.theta)
        sigma = .2
        X = truncnorm((self.lower - mu) / sigma, (self.upper - mu) / sigma, loc=mu, scale=sigma)
        return X.rvs(1)[0]

def calc_pseudoregret(theta, actions):
    diff = theta - np.max(theta)
    new_actions = np.stack(actions, axis=0)
    return -np.sum(np.dot(new_actions, theta))