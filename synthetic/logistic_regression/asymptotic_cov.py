import numpy as np
import os
import os.path as osp
import pathlib
from scipy.optimize import minimize
from scipy.special import log_expit, expit

def sigmoid(z: float) -> float:
    return expit(z)

def cov_hat(samples: np.ndarray) -> np.ndarray:
    return np.cov(samples, rowvar=False, bias=False)

def compute_covariance(config, trial_num):    
    rng = np.random.default_rng(seed=(config["seed"] + trial_num))  # set different seed for each trial

    n = config["data_points"]
    p = config["teacher_dim"]
    d = config["student_dim"]

    # sample a big dataset
    theta_true = rng.standard_normal(p)
    theta_true = theta_true / np.linalg.norm(theta_true, ord=2) * config["scale"]   # scale coefficients to obtain higher or lower variance

    X = rng.standard_normal((n, p)) / np.sqrt(p)
    V = X[:, 0:d]
    probs = sigmoid(X @ theta_true)
    label_var = np.sum(probs * (1 - probs)) / n # compute the average label variance

    y = rng.binomial(1, probs)

    theta_init = np.zeros(p)
    init = np.zeros(d)

    # first compute theta_hat (since we're working with a finite dataset)
    nll_theta = (lambda theta: np.sum(- (y * log_expit(X @ theta) + (1 - y) * log_expit(- X @ theta))))
    grad_theta = (lambda theta: X.T @ (sigmoid(X @ theta) - y))
    theta_hat = minimize(nll_theta, theta_init.copy(), jac=grad_theta, method="L-BFGS-B").x

    # use theta_hat to generate teacher labels
    y_hat = sigmoid(X @ theta_hat)

    # now compute optimal students
    nll_alpha = (lambda alpha: np.sum(- (y * log_expit(V @ alpha) + (1 - y) * log_expit(- V @ alpha))))
    nll_beta = (lambda beta: np.sum(- (y_hat * log_expit(V @ beta) + (1 - y_hat) * log_expit(-V @ beta))))

    grad_alpha = (lambda alpha: V.T @ (sigmoid(V @ alpha) - y))
    grad_beta = (lambda beta: V.T @ (sigmoid(V @ beta) - y_hat))

    alpha_hat = minimize(nll_alpha, init.copy(), jac=grad_alpha, method="L-BFGS-B").x
    beta_hat = minimize(nll_beta, init.copy(), jac=grad_beta, method="L-BFGS-B").x

    # compute empirical hessians and gradient noise second moments
    H_alpha = 1/n * V.T @ np.diag(sigmoid(V @ alpha_hat) * (1 - sigmoid(V @ alpha_hat))) @ V
    H_beta = 1/n * V.T @ np.diag(sigmoid(V @ beta_hat) * (1 - sigmoid(V @ beta_hat))) @ V
    
    H_alpha_inv = np.linalg.inv(H_alpha)
    H_beta_inv = np.linalg.inv(H_beta)

    G_alpha = 1/n * V.T @ np.diag((sigmoid(V @ alpha_hat) - y) ** 2) @ V
    G_beta = 1/n * V.T @ np.diag((sigmoid(V @ beta_hat) - y_hat) ** 2) @ V

    cov_alpha = H_alpha_inv @ G_alpha @ H_alpha_inv
    cov_beta = H_beta_inv @ G_beta @ H_beta_inv
    
    return label_var, np.trace(cov_alpha), np.trace(cov_beta)

def run_experiment(config):
    trials = config["trials"]

    cov_alpha_sum = 0.0
    cov_beta_sum = 0.0
    label_var_sum = 0.0

    for trial in range(trials):
        label_var, cov_alpha, cov_beta = compute_covariance(config, trial)
        label_var_sum += label_var
        cov_alpha_sum += cov_alpha
        cov_beta_sum += cov_beta

    label_var_avg = label_var_sum / trials
    cov_alpha_avg = cov_alpha_sum / trials
    cov_beta_avg = cov_beta_sum / trials

    return label_var_avg, cov_alpha_avg, cov_beta_avg

if __name__ == "__main__":
    data_dir = osp.join(os.getcwd(), "asymp_cov")
    pathlib.Path(data_dir).mkdir(exist_ok=True, parents=True)

    config = {
        "seed": 57,
        "trials": 100,
        "data_points": 1_000,
        "teacher_dim": 100
    }

    # scale of coefficients
    scales = np.logspace(-1, 2.5, 20)
    # ratio of student dim to teacher dim
    ratios = [2/10, 4/10, 6/10, 8/10]

    for i, dim_ratio in enumerate(ratios):
        config["student_dim"] = int(config["teacher_dim"] * dim_ratio)
        all_vals = np.zeros((len(scales), 3))

        for j, scale in enumerate(scales):
            config["scale"] = scale
            label_var_avg, cov_alpha_avg, cov_beta_avg = run_experiment(config)
            all_vals[j] = np.array([label_var_avg, cov_alpha_avg, cov_beta_avg])

        student_dimension_fn = osp.join(data_dir, f"d_{config['student_dim']}.npy")
        np.save(student_dimension_fn, all_vals)