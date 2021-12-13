from test_util import test_Baseline1, test_Baseline2, test_LinUCB, test_UCB
from alg_util import generate_k_dim_vector, generate_random_vec
import numpy as np
import tqdm
import multiprocessing as mp


def calc_errors_and_times(theta, action_set, T, num_arms):
    try:
        ucb_error, ucb_time = test_UCB(theta, action_set, T=T)
        lin_ucb_error, lin_ucb_time = test_LinUCB(theta, action_set, T=T)
        baseline_1_error, baseline_1_time = test_Baseline1(theta, action_set, T=T)
        baseline_2_error, baseline_2_time = test_Baseline2(theta, action_set, T=T)
        return T, num_arms, ucb_error, ucb_time, lin_ucb_error, lin_ucb_time, baseline_1_error, baseline_1_time, baseline_2_error, baseline_2_time
    except:
        pass


def test(name, save=False):
    Ts =  [128, 256, 512, 1024, 2048]
    NumArmss = [2, 8, 16, 32, 64, 128]
    num_epochs = 10

    ucb_errors = {}
    linucb_errors = {}
    baseline_1_errors = {}
    baseline_2_errors = {}
    ucb_times = {}
    linucb_times = {}
    baseline_1_times = {}
    baseline_2_times = {}

    vals = []
    num_skipped = 0

    pool = mp.Pool(10)
    for T in Ts:
        
        ucb_errors[T] = {}
        linucb_errors[T] = {}
        baseline_1_errors[T] = {}
        baseline_2_errors[T] = {}
        
        ucb_times[T] = {}
        linucb_times[T] = {}
        baseline_1_times[T] = {}
        baseline_2_times[T] = {}
        
        for num_arms in NumArmss:      
            ucb_errors[T][num_arms] = []
            linucb_errors[T][num_arms] = []
            baseline_1_errors[T][num_arms] = []
            baseline_2_errors[T][num_arms] = []
            
            ucb_times[T][num_arms] = []
            linucb_times[T][num_arms] = []
            baseline_1_times[T][num_arms] = []
            baseline_2_times[T][num_arms] = []
            
            for _ in range(num_epochs):
                theta = np.matrix(generate_random_vec(dim = 2))
                action_set = np.matrix([generate_random_vec(dim = 2) for _ in range(num_arms)])
                vals.append(pool.apply_async(func=calc_errors_and_times, args=(theta, action_set, T, num_arms)))
    #             calc_errors_and_times(theta, action_set, T, num_arms)
                    
                    
    for val in tqdm(vals):
        final_item = val.get()
        if final_item is None:
            num_skipped += 1
            continue
        T, num_arms, ucb_error, ucb_time, lin_ucb_error, lin_ucb_time, baseline_1_error, baseline_1_time, baseline_2_error, baseline_2_time = final_item
        linucb_errors[T][num_arms].append(lin_ucb_error)
        ucb_errors[T][num_arms].append(ucb_error)
        baseline_1_errors[T][num_arms].append(baseline_1_error)
        baseline_2_errors[T][num_arms].append(baseline_2_error)

        linucb_times[T][num_arms].append(lin_ucb_time)
        ucb_times[T][num_arms].append(ucb_time)
        baseline_1_times[T][num_arms].append(baseline_1_time)
        baseline_2_times[T][num_arms].append(baseline_2_time)
        
    print(num_skipped)

if __name__ == '__main__':
    test()