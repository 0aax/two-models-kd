import os
import os.path as osp

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import argparse
import numpy as np
import pathlib
from typing import Tuple

from scipy.special import logsumexp
from scipy.optimize import minimize

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def softmax(z: np.ndarray) -> np.ndarray:
    # stable softmax
    return np.exp(z - logsumexp(z, axis=1, keepdims=True))

def nll(theta, X, y, k):
    # p is feature dim, k is number of classes
    n = X.shape[0]
    p = X.shape[1]

    theta_mat = theta.reshape((p, k))
    Z = X @ theta_mat   # n x k
    Z_ = Z - logsumexp(Z, axis=1, keepdims=True)

    return np.sum(-y*Z_) * (1/n)

def grad_nll(theta, X, y, k):
    n = X.shape[0]
    p = X.shape[1]

    theta_mat = theta.reshape((p, k))
    preds = softmax(X @ theta_mat)   # n x k
    grad = X.T @ (preds - y)

    return grad.reshape(-1) * (1/n)

def run_sgd(V: np.ndarray, y: np.ndarray, indices: np.ndarray, alpha_ast, alpha_init, config) -> Tuple[np.ndarray, np.ndarray]:
    steps = config["steps"]
    batch_size = config["batch_size"]

    k = y.shape[1]
    d = V.shape[1]

    eta0 = config["step_size"]
    eta_pow = config["decay_exponent"]

    alpha = alpha_init.copy()   # d x k
    alpha_sum = alpha.copy()
    alpha_error = np.zeros(steps + 1)

    alpha_ast_norm = np.linalg.norm(alpha_ast) ** 2

    # since theta is a matrix, we compute frobenious norm
    alpha_error[0] = np.linalg.norm(alpha - alpha_ast, ord="fro") ** 2 / alpha_ast_norm
    print("initial nll (computed wrt standard or distill labels)", nll(alpha, V, y, k))

    for t in range(1, steps + 1):
        batch_indices = indices[(t - 1) * batch_size: t * batch_size]
        # shift back by 1 since we're including the initial point
        v_t = V[batch_indices].T    # d x batch
        y_t = y[batch_indices].T    # to make is k x batch

        # apply inverse link function
        pred = softmax((alpha.T @ v_t).T).T   # since the softmax implementation reduces over axis=1, so we transpose befor applying
        grad = (1 / batch_size) * v_t @ (pred - y_t).T

        eta = eta0 / (1.0 + 1e-3 * t ** eta_pow)

        alpha = alpha - eta * grad
        alpha_sum += alpha
        alpha_error[t] = np.linalg.norm(alpha_sum / (t + 1) - alpha_ast, ord="fro") ** 2 / alpha_ast_norm

    return alpha_error

def run_experiment(config) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed=config["seed"])

    # first compute theta_hat on the dataset
    print("fetching dataset!")
    X, y_str = fetch_openml(name="Fashion-MNIST", version=1, as_frame=False, return_X_y=True, parser="auto")
    
    # labels from openml are strings
    y_int = y_str.astype(int)

    scaler = StandardScaler(with_mean=True)
    X_scale = scaler.fit_transform(X)
    pca_p = PCA(n_components=None)
    X = pca_p.fit_transform(X_scale) # translate into components

    explain_var_ratio = pca_p.explained_variance_ratio_
    cumulative_var = np.cumsum(explain_var_ratio)
    print("cumulative var", cumulative_var[:config["teacher_dim"]])

    X = X[:, :config["teacher_dim"]]

    n = X.shape[0]
    p = X.shape[1]
    k = len(np.unique(y_int))
    d = config["student_dim"]
    indices = np.arange(n)

    print("new", "n", n, "p", p, "d", d, "k", k)

    y = np.zeros((n, k))
    y[np.arange(n), y_int] = 1.0

    V = X[:, :d]    # takes top d components

    theta_init = rng.standard_normal((p*k, )) / np.sqrt(p)
    init_hat = rng.standard_normal((d*k, )) / np.sqrt(d)

    print("fitting theta_hat")
    nll_theta = (lambda theta: nll(theta, X, y, k))
    grad_theta = (lambda theta: grad_nll(theta, X, y, k))
    theta_hat_path = osp.join(config["optimal_estimator_dir"], f"theta_{n}_{p}.npy")
    
    if osp.exists(theta_hat_path):
        theta_hat = np.load(theta_hat_path) # load if already exists
    else:
        theta_hat = minimize(nll_theta, theta_init.copy(), jac=grad_theta, method="L-BFGS-B")
        theta_hat = theta_hat.x.reshape((p, k))

        np.save(theta_hat_path, theta_hat)

    # use theta_hat to generate teacher labels
    y_hat = softmax(X @ theta_hat)

    # now compute optimal students
    nll_alpha = (lambda alpha: nll(alpha, V, y, k))
    grad_alpha = (lambda alpha: grad_nll(alpha, V, y, k))

    nll_beta = (lambda beta: nll(beta, V, y_hat, k))
    grad_beta = (lambda beta: grad_nll(beta, V, y_hat, k))

    print("fitting alpha_hat")
    alpha_hat_path = osp.join(config["optimal_estimator_dir"], f"alpha_{n}_{d}.npy")
    
    if osp.exists(alpha_hat_path):
        alpha_hat = np.load(alpha_hat_path)
    else:
        alpha_hat = minimize(nll_alpha, init_hat.copy(), jac=grad_alpha, method="L-BFGS-B")
        print(alpha_hat)
        alpha_hat = alpha_hat.x.reshape((d, k))
        np.save(alpha_hat_path, alpha_hat)
    
    print("fitting beta_hat")
    beta_hat_path = osp.join(config["optimal_estimator_dir"], f"beta_{n}_{d}.npy")

    if osp.exists(beta_hat_path):
        beta_hat = np.load(beta_hat_path)
    else:
        beta_hat = minimize(nll_beta, init_hat.copy(), jac=grad_beta, method="L-BFGS-B")
        print(beta_hat)
        beta_hat = beta_hat.x.reshape((d, k))
        np.save(beta_hat_path, beta_hat)

    print("alpha^", np.linalg.norm(alpha_hat))
    print("beta^", np.linalg.norm(beta_hat))
    print("Their difference is:", np.linalg.norm(alpha_hat - beta_hat, ord="fro"))

    trials = config["trials"]
    steps = config["steps"]

    n_checkpoints = np.arange(0, steps, 1)

    for trial in range(trials):
        # sample data points
        trial_indices = rng.choice(indices, steps * config["batch_size"], replace=True)

        # init per trial
        init = rng.standard_normal((d*k, )) / np.sqrt(d)
        init_mat = init.reshape((d, k))

        alpha_error = run_sgd(V, y, trial_indices, alpha_hat.copy(), init_mat.copy(), config)
        beta_error = run_sgd(V, y_hat, trial_indices, beta_hat.copy(), init_mat.copy(), config)

        np.save(osp.join(config["trial_dir"], f"trial_{trial}.npy"), [alpha_error[n_checkpoints], beta_error[n_checkpoints]])

if __name__ == "__main__":
    parser = argparse.ArgumentParser("multinomial")
    parser.add_argument("step_exponent", type=float)
    parser.add_argument("student_dim", type=int)

    args = parser.parse_args()
    step_exponent = args.step_exponent
    student_dim = args.student_dim

    step_size = 10 ** (-step_exponent)

    optimal_estimator_dir = osp.join(os.getcwd(), "data", "optimal_estimators")
    pathlib.Path(optimal_estimator_dir).mkdir(exist_ok=True, parents=True)

    data_dir = osp.join(os.getcwd(), "data", f"d_{student_dim}", f"lr_{step_size}")
    pathlib.Path(data_dir).mkdir(exist_ok=True, parents=True)

    print("step", 10**(-step_exponent))
    print("student dim", student_dim)
    
    # step_sizes = [10**(-1), 10**(-0.75), 10**(-0.5), 10**(-0.25), 10**(0.0)]
    # student_dims = [100, 300, 500]

    config = {
        "seed": 57,
        "steps": 500_000,
        "batch_size": 100,
        "trials": 10,
        "step_size": step_size,
        "decay_exponent": 0.6,
        "teacher_dim": 784,     # fashion mnist is 784-dimensional
        "student_dim": student_dim,
        "optimal_estimator_dir": optimal_estimator_dir,
        "trial_dir": data_dir
    }

    run_experiment(config)