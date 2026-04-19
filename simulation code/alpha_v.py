# -*- coding: utf-8 -*-
# exploration.py
# optimal parameters for ps and rps

import numpy as np
import pandas as pd
import time
import sys
import os
from multiprocessing import Pool, cpu_count
import warnings

warnings.filterwarnings('ignore')

exploration_dir = "exploration_output"

# configuration
N = 100000
n0 = 200
M = 200
L_case = 2

alpha_values = np.arange(0, 1.05, 0.05)
v_dist_values = ['pois', 'beta', 'geom', 'exp']

p_values = [20, 40, 60]

alpha_n_values = [1000]
alpha_B_values = [1, 5, 10]

vdist_n_values = [200, 400, 600, 800, 1000]
vdist_B_values = [10]

ar_values = [0.1, 0.5, 0.9]

sys.path.append('.')
from function import generate_data, ols, compute_mse, repeated_subsample


def execute_exploration(params):
    exp_type, param_val, p_val, n_val, B_val, ar_val, rep_idx = params

    seed = (p_val * 1000000 + n_val * 1000 + B_val * 10 +
            int(ar_val * 100) + rep_idx)
    np.random.seed(seed)

    X, Y, beta_true = generate_data(N, p_val, ar_val)
    beta_global = ols(X, Y)
    if exp_type == 'alpha':
        beta_est = repeated_subsample(
            X, Y, n_val, B_val, 'ps', alpha=param_val, n0=n0, L_case=L_case)
    else:
        beta_est = repeated_subsample(
            X, Y, n_val, B_val, 'rps', v_dist=param_val)

    mse_uncond, log_uncond = compute_mse(beta_est, beta_true)
    mse_cond, log_cond = compute_mse(beta_est, beta_global)

    return {
        'exp_type': exp_type,
        'param_value': param_val, 'p': p_val, 'n': n_val, 'B': B_val, 'ar': ar_val, 'rep': rep_idx,
        'mse_unconditional': mse_uncond, 'log_mse_unconditional': log_uncond,
        'mse_conditional': mse_cond, 'log_mse_conditional': log_cond}

def create_tasks(exp_type):
    tasks = []

    if exp_type == 'alpha':
        param_list = alpha_values
        n_list = alpha_n_values
        B_list = alpha_B_values
    else:
        param_list = v_dist_values
        n_list = vdist_n_values
        B_list = vdist_B_values

    for param_val in param_list:
        for p_val in p_values:
            for n_val in n_list:
                for B_val in B_list:
                    for ar_val in ar_values:
                        for rep_idx in range(1, M + 1):
                            tasks.append((exp_type, param_val, p_val, n_val,
                                          B_val, ar_val, rep_idx))
    np.random.shuffle(tasks)
    return tasks


def run_parallel_exploration(exp_type):
    if exp_type == 'alpha':
        print("for alpha")
    else:
        print("for v_dist")

    tasks = create_tasks(exp_type)
    start_time = time.time()

    n_cores = max(1, cpu_count() - 1)
    with Pool(processes=n_cores) as pool:
        results = pool.map(execute_exploration, tasks, chunksize=1)

    total_time = time.time() - start_time
    print(f"完成时间: {total_time:.1f}秒 ({total_time / 60:.1f}分钟)")

    df_raw = pd.DataFrame(results)
    group_cols = ['p', 'n', 'B', 'ar', 'param_value']
    df_summary = df_raw.groupby(group_cols).agg({
        'mse_unconditional': 'mean', 'log_mse_unconditional': 'mean',
        'mse_conditional': 'mean', 'log_mse_conditional': 'mean'
    }).reset_index()

    rename_dict = {
        'mse_unconditional': 'mean_mse_unconditional',
        'log_mse_unconditional': 'mean_log_mse_unconditional',
        'mse_conditional': 'mean_mse_conditional',
        'log_mse_conditional': 'mean_log_mse_conditional'
    }
    df_summary = df_summary.rename(columns=rename_dict)

    # save
    raw_file = os.path.join(exploration_dir, f"{exp_type}_raw.pkl")
    summary_file = os.path.join(exploration_dir, f"{exp_type}_summary.pkl")
    raw_csv = os.path.join(exploration_dir, f"{exp_type}_raw.csv")
    summary_csv = os.path.join(exploration_dir, f"{exp_type}_summary.csv")

    df_raw.to_pickle(raw_file)
    df_summary.to_pickle(summary_file)
    df_raw.to_csv(raw_csv, index=False)
    df_summary.to_csv(summary_csv, index=False)
    return df_summary


def analyze_optimal_params():
    print("\noptimal parameters analysis ...")
    optimal_params = {}
    for exp_type in ['alpha', 'v_dist']:
        summary_file = os.path.join(exploration_dir, f"{exp_type}_summary.pkl")
        df = pd.read_pickle(summary_file)

        param_summary = df.groupby('param_value').agg({
            'mean_mse_unconditional': 'mean',
            'mean_mse_conditional': 'mean'
        }).reset_index()

        idx_cond = param_summary['mean_mse_conditional'].idxmin()
        best_cond = param_summary.loc[idx_cond, 'param_value']
        mse_cond = param_summary.loc[idx_cond, 'mean_mse_conditional']

        idx_uncond = param_summary['mean_mse_unconditional'].idxmin()
        best_uncond = param_summary.loc[idx_uncond, 'param_value']
        mse_uncond = param_summary.loc[idx_uncond, 'mean_mse_unconditional']

        optimal_params[f'{exp_type}_cond'] = best_cond
        optimal_params[f'{exp_type}_uncond'] = best_uncond

        print(f"results of {exp_type.upper()}:")
        print(f"optimal_cond : {best_cond} (MSE={mse_cond:.6f})")
        print(f"optimal_uncond: {best_uncond} (MSE={mse_uncond:.6f})")

    return optimal_params


def main():
    print("Start")
    start_time = time.time()
    alpha_summary = run_parallel_exploration('alpha')
    '''
    vdist_summary = run_parallel_exploration('v_dist')
    '''
    optimal_params = analyze_optimal_params()

    params_file = os.path.join(exploration_dir, "optimal_params.txt")
    with open(params_file, 'w') as f:
        f.write("# optimal parameters\n")
        f.write("\n")

        f.write(f"alpha_opt_cond = {optimal_params['alpha_cond']}\n")
        f.write(f"alpha_opt_uncond = {optimal_params['alpha_uncond']}\n")
        '''
        f.write(f"v_dist_opt_cond = '{optimal_params['v_dist_cond']}'\n")
        f.write(f"v_dist_opt_uncond = '{optimal_params['v_dist_uncond']}'\n")
        '''
    print(f"\nsave to {params_file}")

    total_time = time.time() - start_time

    return optimal_params

if __name__ == "__main__":
    params = main()