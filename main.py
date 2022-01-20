from dataset_util import load_battery_dataset, find_mean
from test_util import test_Baseline1, test_Baseline2, test_LinUCB, test_UCB, test_Baseline2_LP
from alg_util import generate_k_dim_vector, generate_random_vec
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")
from sampler import SyntheticSampler, BatterySampler
import os
from plot_util import plot
import pandas as pd
import argparse
from datetime import datetime

def calc_errors_and_times(theta, action_set, T, num_arms, sigma, timelimit):
    ucb_error, ucb_time, is_feasible, is_other_feasible = test_UCB(theta, action_set, sigma, T=T, timelimit=timelimit)
    lin_ucb_error, lin_ucb_time = test_LinUCB(theta, action_set, sigma, T=T, timelimit=timelimit)
    baseline_1_error, baseline_1_time = test_Baseline1(theta, action_set, sigma, T=T)
    baseline_2_error, baseline_2_time = test_Baseline2(theta, action_set, sigma, T=T)
    baseline_2_lp_error, baseline_2_lp_time = test_Baseline2_LP(theta, action_set, sigma, T=T, timelimit=timelimit)
    print(baseline_2_lp_error)
    if lin_ucb_error is None or baseline_2_lp_error is None:
        return None

    # ucb_error = -1
    # ucb_time = -1
    # baseline_1_error = -1
    # baseline_1_time = -1
    # baseline_2_error = -1
    # baseline_2_time = -1
    print("Top")
    print(is_feasible)
    print(is_other_feasible)
    return T, num_arms, ucb_error, ucb_time, lin_ucb_error, lin_ucb_time, baseline_1_error, baseline_1_time, baseline_2_error, baseline_2_time, baseline_2_lp_error, baseline_2_lp_time

def test_battery(name, save=False):
    sampler = BatterySampler()
    theta, action_set, sigma = sampler.sample()
    
    print(calc_errors_and_times(theta, action_set, 1000, len(action_set), sigma))

def test_synthetic(name, dim, ord, timelimit=None, save=False):
    ord = float(ord)
    if not os.path.exists("data/all.csv"):
        df = pd.DataFrame()
    else:
        df = pd.read_csv("data/all.csv")

    Ts =  [1024, 1536, 2048]
    NumArmss = [2,4,32,64]
    num_epochs = 10

    ucb_errors = {}
    linucb_errors = {}
    baseline_1_errors = {}
    baseline_2_errors = {}
    baseline_2_lp_errors = {}
    ucb_times = {}
    linucb_times = {}
    baseline_1_times = {}
    baseline_2_times = {}
    baseline_2_lp_times = {}

    vals = []
    num_skipped = 0

    pool = mp.Pool(10)
    for T in Ts:

        ucb_errors[T] = {}
        linucb_errors[T] = {}
        baseline_1_errors[T] = {}
        baseline_2_errors[T] = {}
        baseline_2_lp_errors[T] = {}

        ucb_times[T] = {}
        linucb_times[T] = {}
        baseline_1_times[T] = {}
        baseline_2_times[T] = {}
        baseline_2_lp_times[T] = {}

        for num_arms in NumArmss:
            ucb_errors[T][num_arms] = []
            linucb_errors[T][num_arms] = []
            baseline_1_errors[T][num_arms] = []
            baseline_2_errors[T][num_arms] = []
            baseline_2_lp_errors[T][num_arms] = []

            ucb_times[T][num_arms] = []
            linucb_times[T][num_arms] = []
            baseline_1_times[T][num_arms] = []
            baseline_2_times[T][num_arms] = []
            baseline_2_lp_times[T][num_arms] = []
            sampler = SyntheticSampler(num_arms, dim=dim, ord=ord)
            for _ in range(num_epochs):
                theta, action_set, sigma = sampler.sample()
                vals.append(pool.apply_async(func=calc_errors_and_times, args=(theta, action_set, T, num_arms, sigma, timelimit)))
                # calc_errors_and_times(theta, action_set, T, num_arms, sigma, timelimit)



    for val in tqdm(vals, smoothing=.05):
        final_item = val.get()
        if final_item is None:
            num_skipped += 1
            continue
        T, num_arms, ucb_error, ucb_time, lin_ucb_error, lin_ucb_time, baseline_1_error, baseline_1_time, baseline_2_error, baseline_2_time, baseline_2_lp_error, baseline_2_lp_time = final_item
        linucb_errors[T][num_arms].append(lin_ucb_error)
        ucb_errors[T][num_arms].append(ucb_error)
        baseline_1_errors[T][num_arms].append(baseline_1_error)
        baseline_2_errors[T][num_arms].append(baseline_2_error)
        baseline_2_lp_errors[T][num_arms].append(baseline_2_lp_error)

        linucb_times[T][num_arms].append(lin_ucb_time)
        ucb_times[T][num_arms].append(ucb_time)
        baseline_1_times[T][num_arms].append(baseline_1_time)
        baseline_2_times[T][num_arms].append(baseline_2_time)
        baseline_2_lp_times[T][num_arms].append(baseline_2_lp_time)

    data_dict = {}

    now = datetime.now()
    folder_name = "images/{}_{}".format(name, now.strftime("%m_%d_%Y_%H_%M_%S"))
    os.mkdir("images/{}_{}".format(name, now.strftime("%m_%d_%Y_%H_%M_%S")))

    data_dict["name"] = name
    data_dict["date"] = now.strftime("%m_%d_%Y_%H_%M_%S")
    data_dict["LinUCB Error"] = find_mean(linucb_errors)
    data_dict["UCB Error"] = find_mean(ucb_errors)
    data_dict["Baseline 1 Error"] = find_mean(baseline_1_errors)
    data_dict["Baseline 2 Error"] = find_mean(baseline_2_errors)
    data_dict["Baseline 2 LP Error"] = find_mean(baseline_2_lp_errors)

    data_dict["Dimension"] = dim
    data_dict["LinUCB Time"] = find_mean(linucb_times)
    data_dict["UCB Time"] = find_mean(ucb_times)
    data_dict["Baseline 1 Time"] = find_mean(baseline_1_times)
    data_dict["Baselien 2 Time"] = find_mean(baseline_2_times)
    data_dict["Baseline 2 LP Time"] = find_mean(baseline_2_lp_times)
    data_dict["Type"] = "Synthetic"
    data_dict["Order"] = ord
    data_dict["Time Limit Reached"] = num_skipped
    df = df.append(data_dict, ignore_index=True)
    df.to_csv("data/all.csv")

    plot(linucb_errors, "LinUCB Error", "linucb_error.png", folder_name)
    plot(ucb_errors, "UCB Error", "ucb_error.png", folder_name)
    plot(baseline_1_errors, "Baseline 1 Error", "baseline_1_error.png", folder_name)
    plot(baseline_2_errors, "Baseline 2 Error", "baseline_2_error.png", folder_name)
    plot(linucb_times, "LinUCB time", "linucb_time.png", folder_name)
    plot(ucb_times, "UCB time", "ucb_time.png", folder_name)
    plot(baseline_1_times, "Baseline 1 time", "baseline_1_time.png", folder_name)
    plot(baseline_2_times, "Baseline 2 time", "baseline_2_time.png", folder_name)
    plot(baseline_2_lp_errors, "Baseline 2 Error", "baseline_2_lp_error.png", folder_name)
    plot(baseline_2_lp_times, "Baseline 2 time", "baseline_2_lp_time.png", folder_name)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--sample', choices=['synthetic', 'battery'],
                    help='sample type')
    parser.add_argument("--dim", type=int, help="Dimension")
    parser.add_argument("--name", type=str, help="Name of experiment")
    parser.add_argument("--timelimit", type=int, default=None, help="Name of experiment")
    parser.add_argument("--ord", choices=['1', '2', 'inf'], help='Type of norm')
    args = parser.parse_args()
    if args.sample == "synthetic":
        test_synthetic(name=args.name, dim=args.dim, ord=args.ord, timelimit=args.timelimit)
    else:
        test_battery(name=args.name)
