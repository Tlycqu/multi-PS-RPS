# -*- coding: utf-8 -*-
# application_origin.py

import numpy as np
import pandas as pd
import time
import os
import sys
from multiprocessing import Pool, cpu_count
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

sys.path.append('.')

from function import ols, repeated_subsample


def evaluate_exp1(params, X_all, y_all, beta_full_all):
    run_id, n, B, method = params
    np.random.seed(42 + run_id * 1000)

    t1 = time.process_time()

    if method == 'PS':
        beta_sub = repeated_subsample(
            X_all, y_all, n, B, 'ps', alpha=0.4, n0=200, L_case=2)
    elif method == 'RPS':
        beta_sub = repeated_subsample(
            X_all, y_all, n, B, 'rps', v_dist='pois')
    elif method == 'Unif':
        beta_sub = repeated_subsample(
            X_all, y_all, n, B, 'unif')
    elif method == 'Unif-PS':
        beta_sub = repeated_subsample(
            X_all, y_all, n, B, 'unif_ps')

    t_algo = time.process_time() - t1

    t1 = time.process_time()
    mse = np.sum((beta_sub - beta_full_all) ** 2)
    t_mse = time.process_time() - t1

    return {
        'run_id': run_id, 'method': method, 'n': n, 'B': B,
        'mse': mse, 'mse_cond': mse, 't_algo': t_algo, 't_mse': t_mse}


def evaluate_exp2(params):
    run_id, n, B, method, X_train, y_train, X_test, y_test, beta_full_train = params

    np.random.seed(42 + run_id * 1000)

    t1 = time.process_time()

    if method == 'Full':
        beta_sub = beta_full_train
    elif method == 'PS':
        beta_sub = repeated_subsample(
            X_train, y_train, n, B, 'ps', alpha=0.4, n0=200, L_case=2)
    elif method == 'RPS':
        beta_sub = repeated_subsample(
            X_train, y_train, n, B, 'rps', v_dist='pois')
    elif method == 'Unif':
        beta_sub = repeated_subsample(
            X_train, y_train, n, B, 'unif')
    elif method == 'Unif-PS':
        beta_sub = repeated_subsample(
            X_train, y_train, n, B, 'unif_ps')

    t_algo = time.process_time() - t1

    t1 = time.process_time()
    y_pred = X_test @ beta_sub
    pe = np.sum((y_test - y_pred) ** 2)
    t_pred = time.process_time() - t1

    return {
        'run_id': run_id, 'method': method, 'n': n, 'B': B,
        'pe': pe, 't_algo': t_algo, 't_pred': t_pred}

def run_parallel(tasks, evaluate_func, n_cores):
    total_tasks = len(tasks)
    results = []
    with Pool(processes=n_cores) as pool:
        for i, result in enumerate(pool.imap_unordered(evaluate_func, tasks)):
            results.append(result)
            if (i + 1) % 1000 == 0:
                print(f"progress: {i + 1}/{total_tasks}")

    return results


def save_results(results, output_dir, filename, add_time=True, t_data=None, t_full=None):
    os.makedirs(output_dir, exist_ok=True)
    df_results = pd.DataFrame(results)
    if add_time and t_data is not None and t_full is not None:
        df_results['t_data'] = t_data / len(results)
        df_results['t_full'] = t_full / len(results)

    pkl_path = f"{output_dir}/{filename}.pkl"
    csv_path = f"{output_dir}/{filename}.csv"
    df_results.to_pickle(pkl_path)
    df_results.to_csv(csv_path, index=False)
    return df_results


def create_summary(df, group_cols, value_cols, output_path):
    agg_dict = {}
    for col in value_cols:
        agg_dict[col] = ['mean', 'std']

    summary = df.groupby(group_cols).agg(agg_dict).reset_index()

    new_columns = []
    for col in summary.columns:
        if isinstance(col, tuple):
            if col[1] == 'mean':
                new_columns.append(f"{col[0]}_mean")
            elif col[1] == 'std':
                new_columns.append(f"{col[0]}_std")
            else:
                new_columns.append(col[0])
        else:
            new_columns.append(col)

    summary.columns = new_columns
    summary.to_csv(output_path, index=False)
    return summary


def run_exp1():
    print("exp1: evaluation of parameter estimation accuracy")

    t1 = time.process_time()

    df = pd.read_csv("train.csv")
    X_raw = df.iloc[:, :-1].values.astype(np.float64)
    y_raw = df.iloc[:, -1].values.astype(np.float64)
    epsilon = 1e-6
    y_all = np.log(y_raw + epsilon)
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_raw)

    t_data = time.process_time() - t1

    t1 = time.process_time()
    beta_full_all = ols(X_all, y_all)
    t_full = time.process_time() - t1

    n_values = [200, 400, 600, 800, 1000]
    B_values = list(range(1, 11))
    methods = ['PS', 'RPS', 'Unif', 'Unif-PS']
    M = 500

    tasks = []
    for run_id in range(M):
        for n in n_values:
            for B in B_values:
                for method in methods:
                    tasks.append((run_id, n, B, method))

    evaluate_func = partial(
        evaluate_exp1,
        X_all=X_all, y_all=y_all, beta_full_all=beta_full_all)

    n_cores = max(1, cpu_count() - 1)
    results = run_parallel(tasks, evaluate_func, n_cores)

    output_dir = "exp1_results"
    df_raw = save_results(results, output_dir, "exp1_raw", True, t_data, t_full)

    if df_raw is not None:
        df_raw['total_time'] = df_raw['t_algo'] + df_raw['t_mse']
        summary = create_summary(
            df_raw, ['method', 'n', 'B'],
            ['mse', 't_algo', 't_mse', 'total_time'],
            f"{output_dir}/exp1_summary.csv")

    return df_raw


def run_exp2():
    print("exp2: evaluation of predictive performance")

    df = pd.read_csv("train.csv")
    X_raw = df.iloc[:, :-1].values.astype(np.float64)
    y_raw = df.iloc[:, -1].values.astype(np.float64)
    epsilon = 1e-6
    y_raw_log = np.log(y_raw + epsilon)

    n_values = [200, 400, 600, 800, 1000]
    B_values = list(range(1, 11))
    methods = ['PS', 'RPS', 'Unif', 'Unif-PS']
    M = 500

    tasks = []
    for run_id in range(M):
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw, y_raw_log, train_size=20000, test_size=1263,
            random_state=run_id, shuffle=True)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        beta_full_train = ols(X_train, y_train)
        tasks.append((run_id, 20000, 1, 'Full', X_train, y_train, X_test, y_test, beta_full_train))

        for n in n_values:
            for B in B_values:
                for method in methods:
                    tasks.append((run_id, n, B, method, X_train, y_train, X_test, y_test, beta_full_train))

    evaluate_func = partial(evaluate_exp2)
    n_cores = max(1, cpu_count() - 1)
    results = run_parallel(tasks, evaluate_func, n_cores)

    output_dir = "exp2_results"
    df_raw = save_results(results, output_dir, "exp2_raw", add_time=False)

    if df_raw is not None:
        df_raw['total_time'] = df_raw['t_algo'] + df_raw['t_pred']
        summary = create_summary(
            df_raw, ['method', 'n', 'B'],
            ['pe', 't_algo', 't_pred', 'total_time'],
            f"{output_dir}/exp2_summary.csv")
    return df_raw


def main():
    print("\n run exp1")
    run_exp1()

    print("\n run exp2")
    run_exp2()

if __name__ == "__main__":
    main()