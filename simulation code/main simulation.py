# -*- coding: utf-8 -*-
# simulation.py


import numpy as np
import pandas as pd
import time
import sys
from multiprocessing import Pool, cpu_count
import warnings
import os

warnings.filterwarnings('ignore')

# configuration
N = 100000
M = 500
L_case = 2

alpha = 0.4
v_dist = 'pois'
n0 = 200

p_values = [20, 40, 60]
n_values = [200, 400, 600, 800, 1000]
B_values = [1, 5, 10, 15, 20]
ar_values = [0.1, 0.5, 0.9]

methods = ['Full', 'PS', 'RPS', 'Unif-PS']

# 导入函数
sys.path.append('.')
from function import generate_data, repeated_subsample


def run_simulation(params):
    p_val, n_val, B_val, ar_val, method, rep_idx = params

    seed = (p_val * 1000000 + n_val * 1000 + B_val * 10 +
            int(ar_val * 100) + rep_idx)
    np.random.seed(seed)

    # data generation
    t1 = time.process_time()
    X, Y, beta_true = generate_data(N, p_val, ar_val)
    t_data = time.process_time() - t1

    # global estimator
    t1 = time.process_time()
    beta_full = np.linalg.lstsq(X, Y, rcond=None)[0]
    t_full = time.process_time() - t1

    # algorithm execution
    t1 = time.process_time()

    if method == 'Full':
        beta_est = beta_full
    elif method == 'PS':
        beta_est = repeated_subsample(
            X, Y, n_val, B_val, 'ps', alpha=alpha, n0=n0, L_case=L_case)
    elif method == 'RPS':
        beta_est = repeated_subsample(
            X, Y, n_val, B_val, 'rps', v_dist=v_dist)
    elif method == 'Unif':
        beta_est = repeated_subsample(
            X, Y, n_val, B_val, 'unif')
    elif method == 'Unif-PS':
        beta_est = repeated_subsample(
            X, Y, n_val, B_val, 'unif_ps')

    t_algo = t_full if method == 'Full' else time.process_time() - t1

    # compute MSE
    t1 = time.process_time()
    mse_uncond = np.sum((beta_est - beta_true) ** 2)
    t_mse_uncond = time.process_time() - t1

    t1 = time.process_time()
    mse_cond = np.sum((beta_est - beta_full) ** 2)
    t_mse_cond = time.process_time() - t1

    return {
        'p': p_val, 'n': n_val, 'B': B_val, 'ar': ar_val,
        'method': method, 'rep': rep_idx,
        'mse_uncond': mse_uncond, 'mse_cond': mse_cond,
        't_data': t_data, 't_full': t_full, 't_algo': t_algo,
        't_mse_uncond': t_mse_uncond, 't_mse_cond': t_mse_cond}


def create_tasks():
    tasks = []
    for p_val in p_values:
        for n_val in n_values:
            for B_val in B_values:
                for ar_val in ar_values:
                    for rep_idx in range(1, M + 1):
                        for method in methods:
                            tasks.append((p_val, n_val, B_val, ar_val, method, rep_idx))
    return tasks


def main():
    print("Start")
    print(f"parameters: p={p_values}, n={n_values}, B={B_values}, ar={ar_values}, alpha={alpha}, v_dist={v_dist}")
    print(f"method: {methods}")

    output_dir = "simulation_output"
    os.makedirs(output_dir, exist_ok=True)

    tasks = create_tasks()
    total_tasks = len(tasks)
    n_cores = max(1, cpu_count() - 1)

    results = []
    with Pool(processes=n_cores) as pool:
        for i, result in enumerate(pool.imap_unordered(run_simulation, tasks, chunksize=1)):
            results.append(result)
            if (i + 1) % 1000 == 0:
                print(f"Progress: {i + 1}/{total_tasks}")

    df_raw = pd.DataFrame(results)
    group_cols = ['p', 'n', 'B', 'ar', 'method']
    df_summary = df_raw.groupby(group_cols).agg({
        'mse_uncond': 'mean', 'mse_cond': 'mean',
        't_data': 'mean', 't_full': 'mean', 't_algo': 'mean',
        't_mse_uncond': 'mean', 't_mse_cond': 'mean'
    }).reset_index()

    raw_file = os.path.join(output_dir, "simulation_raw.pkl")
    summary_pkl = os.path.join(output_dir, "simulation_summary.pkl")
    summary_csv = os.path.join(output_dir, "simulation_summary.csv")

    df_raw.to_pickle(raw_file)
    df_summary.to_pickle(summary_pkl)
    df_summary.to_csv(summary_csv, index=False)

    return df_summary

if __name__ == "__main__":
    df_result = main()
