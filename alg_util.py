import random
import numpy as np
import matplotlib.pyplot as plt

def train_alg_UCB(alg, T, theta, oracle):
    for time in range(T):
        action = alg.compute()
        reward = oracle.compute_reward(action)
        alg.observe_reward(reward)

def generate_k_dim_vector(dim=2, L = 1):
    full_index = random.randint(0,dim-1)
    vec = np.random.random(int(dim)) * 2 * L - L
    vec[full_index] = np.sign(vec[full_index]) * 1
    return vec

def generate_random_vec(dim=2, mag=1, ord=2):
    vec = np.random.random(int(dim)) * 2 - 1
    vec = vec/np.linalg.norm(vec, ord=ord)*mag
    return vec

def generate_random_l1_vec(dim=2, mag=1) -> np.ndarray:
    vec = np.random.random(int(dim))
    vec = vec/np.linalg.norm(vec, ord=1)*mag
    return vec

def generate_random_linf_vec(dim=2, mag=1) -> np.ndarray:
    vec = np.random.random(int(dim))
    vec = vec/np.linalg.norm(vec, ord=np.inf)*mag
    return vec

def plot_vectors(theta, actions):
    plt.axis([-1, 1, -1, 1])
    plt.quiver(0, 0, theta[0], theta[1], color='green', scale=1)
    for idx, action in enumerate(actions):
        plt.quiver(0, 0, action[0], action[1], color = [(idx/len(actions), 0, 0)], scale=1)
