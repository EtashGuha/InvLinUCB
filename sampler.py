import numpy as np
from alg_util import generate_random_vec
from dataset_util import load_battery_dataset

class SyntheticSampler():
    def __init__(self, num_arms, dim=2, sigma=.2, ord=2, seed=None) -> None:
        self.dim = dim
        self.sigma = sigma
        self.num_arms = num_arms
        self.ord = ord
        if seed:
            np.random.seed(seed)

    def sample(self):
        theta = np.matrix(generate_random_vec(dim = self.dim))
        action_set = np.matrix([generate_random_vec(dim = self.dim, ord = self.ord) for _ in range(self.num_arms)])
        return theta, action_set, self.sigma

class BatterySampler():
    def __init__(self):
        pass
    def sample(self):
        action_set, theta, sigma = load_battery_dataset()
        return np.squeeze(np.asarray(theta)), action_set, sigma

class UniformSyntheticSampler():
    def __init__(self, num_arms, dim=2, sigma=.2, ord=2, seed=None) -> None:
        self.dim = dim
        self.sigma = sigma
        self.num_arms = num_arms
        self.ord = ord

    def sample(self):
        angles_per_dim = int(self.num_arms ** (1/(self.dim - 1)))
        angle = 360 / angles_per_dim
        # TODO: convert angles to vectors. placeholder for now
        theta = np.matrix(generate_random_vec(dim = self.dim))
        action_set = np.matrix([generate_random_vec(dim = self.dim, ord = self.ord) for _ in range(self.num_arms)])
        return theta, action_set, self.sigma
