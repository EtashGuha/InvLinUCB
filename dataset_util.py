from typing import Tuple
import numpy as np


def load_battery_dataset() -> Tuple[np.matrix, np.matrix, float]:
    dataset = np.load('battery_hi.npy')
    n_arms = dataset.shape[0]
    max_mean = np.max(dataset[:, 0])
    action_set = np.matrix(np.identity(n_arms))
    # All the sigmas are the same. In the previous work, just divide these by the max mean
    sigma = (dataset[0, 1] / max_mean)
    theta = np.matrix(dataset[:, 0]) / max_mean
    return action_set, theta, sigma
