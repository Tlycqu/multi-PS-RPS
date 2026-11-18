# -*- coding: utf-8 -*-
# function.py


import numpy as np
from scipy.linalg import cholesky
import warnings
warnings.filterwarnings('ignore')


def get_beta_true(p):
    return np.tile([1.0, 1.5], (p + 1) // 2)[:p]

def get_cov_matrix(p, ar=0.5):
    idx = np.arange(p)
    idx_matrix = np.abs(idx - idx.reshape(-1, 1))
    return ar ** idx_matrix

def generate_data(N, p, ar=0.5):
    sigma_X = get_cov_matrix(p, ar)
    beta_true = get_beta_true(p)
    Z = np.random.normal(0, 1, (N, p))
    L = cholesky(sigma_X, lower=True)
    X = Z @ L.T
    eps = np.random.normal(0, 1, N)
    Y = X @ beta_true + eps

    return X, Y, beta_true


def ols(X, Y):
    XTX = X.T @ X
    XTY = X.T @ Y
    return np.linalg.pinv(XTX) @ XTY

def compute_mse(beta_est, beta_ref):
    diff = beta_est - beta_ref
    mse = np.sum(diff ** 2)
    log_mse = np.log(mse)
    return mse, log_mse

def compute_lev(X):
    p = X.shape[1]
    XTX = X.T @ X
    XTX_inv = np.linalg.pinv(XTX)
    lev = np.sum((X @ XTX_inv) * X, axis=1)
    return lev


def pilot_pi(X, Y, n, alpha=0.5, n0=200, optimality="L", L_case=2):
    N, p = X.shape

    # 1. pilot phase
    pi_pilot = n0 / N
    pilot_indices = np.where(np.random.binomial(1, pi_pilot, N) == 1)[0]
    X_pilot = X[pilot_indices]
    Y_pilot = Y[pilot_indices]
    n_pilot_actual = len(pilot_indices)
    beta0 = ols(X_pilot, Y_pilot)

    # 2. calculate optimal sampling probability
    eps0 = Y - X @ beta0
    if optimality == "L":
        Sigma_hat = X_pilot.T @ X_pilot / n_pilot_actual
        Sigma_inv = np.linalg.pinv(Sigma_hat)

        if L_case == 1:
            L = np.eye(p)
            L_Sigma_inv = Sigma_inv
            h = np.abs(eps0) * np.sqrt(np.sum((X @ L_Sigma_inv) ** 2, axis=1))
        elif L_case == 2:
            h = np.abs(eps0) * np.sqrt(np.sum(X ** 2, axis=1))
        elif L_case == 3:
            L = np.ones(p) / np.sqrt(p)
            L_Sigma_inv = L @ Sigma_inv
            L_Sigma_inv_X = X @ L_Sigma_inv
            h = np.abs(eps0) * np.abs(L_Sigma_inv_X)
        elif L_case == 4:
            L = np.zeros(p)
            L[0] = 1
            L_Sigma_inv = L @ Sigma_inv
            L_Sigma_inv_X = X @ L_Sigma_inv
            h = np.abs(eps0) * np.abs(L_Sigma_inv_X)

    gamma0 = np.mean(h[pilot_indices])
    pi_opt = n * h / (N * gamma0)
    pi_alpha = (1 - alpha) * pi_opt + alpha * (n / N)
    pi_tilde = np.minimum(pi_alpha, 1.0)

    return pi_tilde


def rps(X, Y, n, v_dist):
    N, p = X.shape
    pi_n = n / N

    # 1. Bernoulli sampling
    u = np.where(np.random.binomial(1, pi_n, N) == 1)[0]
    X_sub = X[u]
    Y_sub = Y[u]
    n_sub = len(u)

    # 2. perturbation weight
    if v_dist == 'exp':
        v = np.random.exponential(1 / pi_n, n_sub)
    elif v_dist == 'pois':
        v = np.random.poisson(1 / pi_n, n_sub)
    elif v_dist == 'geom':
        p_geom = pi_n
        v = np.random.geometric(p_geom, n_sub)
    elif v_dist == 'beta':
        v = np.random.uniform(0, 2 / pi_n, n_sub)

    # 3. weighted estimation
    w = v
    X_weighted = X_sub * np.sqrt(w).reshape(-1, 1)
    Y_weighted = Y_sub * np.sqrt(w)

    return ols(X_weighted, Y_weighted)

def unif(X, Y, n):
    indices = np.random.choice(X.shape[0], n, replace=True)
    return ols(X[indices], Y[indices])

def unif_ps(X, Y, n):
    N, p = X.shape
    pi_unif = n / N
    sample_indices = np.where(np.random.binomial(1, pi_unif, N) == 1)[0]
    w = 1.0 / np.sqrt(pi_unif)
    X_weighted = X[sample_indices] * w
    Y_weighted = Y[sample_indices] * w

    return ols(X_weighted, Y_weighted)


def repeated_subsample(X, Y, n, B, method='ps', alpha=0.5, v_dist='exp',
                       n0=200, optimality="L", L_case=2):
    p = X.shape[1]
    beta_sum = np.zeros(p)
    if method == 'ps':
        pi_tilde = pilot_pi(X, Y, n, alpha, n0, optimality, L_case)
        N = len(X)
        for i in range(B):
            delta_indices = np.where(np.random.binomial(1, pi_tilde, N) == 1)[0]
            w = 1.0 / pi_tilde[delta_indices]
            X_weighted = X[delta_indices] * np.sqrt(w).reshape(-1, 1)
            Y_weighted = Y[delta_indices] * np.sqrt(w)
            beta_est = ols(X_weighted, Y_weighted)
            beta_sum += beta_est
    else:
        method_functions = {
            'rps': lambda: rps(X, Y, n, v_dist),
            'unif': lambda: unif(X, Y, n),
            'unif_ps': lambda: unif_ps(X, Y, n)
        }

        for i in range(B):
            beta_est = method_functions[method]()
            beta_sum += beta_est

    return beta_sum / B

