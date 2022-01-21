import numpy as np

def calc_Vs(alg):
    Vs = []
    Vs.append(alg.lamda * np.identity(alg.dim))
    for t in range(alg.T):
        Vs.append(Vs[-1] + (alg.actions[t].T * alg.actions[t]))
    return(Vs)

def get_orthogonal_matrix(vec):
    first = np.identity(len(vec) - 1)
    if vec[0].item() == 0:
        second = -1 * vec[1:]
    else:
        second = -1 * vec[1:] / vec[0]
    others = np.concatenate((second.T, first),axis=0)
    banana = others @ (others.T @ others) @ others.T
    return banana


def initialize_taus_np_optarm(alg):
    taus = [-1] * alg.arm.shape[0]
    tau_bars = [-1] * alg.arm.shape[0]
    num_pulls = generate_num_pulls(alg)

    optimal_arm = None
    most_pulls = -1
    for i in range(len(alg.arm)):
        if num_pulls[i][alg.T - 1] > most_pulls:
            most_pulls = num_pulls[i][alg.T - 1] 
            optimal_arm = i

    tau_bar = None
    for t, action in reversed(list(enumerate(alg.action_idxs))):
        if action == optimal_arm:
            past_arm = True
            tau_bar = t
        if tau_bar is not None and taus[action] == -1:
            taus[action] = t
            tau_bars[action] = tau_bar
    return optimal_arm, taus, num_pulls, tau_bars



def generate_num_pulls(alg):
    num_pulls = {}
    T = alg.T
    for i in range(len(alg.arm)):
        num_pulls[i] = []

    for t in range(T):
        for key in num_pulls.keys():
            if alg.action_idxs[t] != key:
                if t != 0:
                    num_pulls[key].append(num_pulls[key][ - 1])
                else:
                    num_pulls[key].append(0)
            else:
                if t != 0:
                    num_pulls[key].append(num_pulls[key][ - 1] + 1)
                else:
                    num_pulls[key].append(1)
    return num_pulls