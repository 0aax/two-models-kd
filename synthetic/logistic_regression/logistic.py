import numpy as np
import os
import os.path as osp
import pathlib
from scipy.optimize import minimize
from typing import Tuple
import argparse

def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))

def cov_hat(samples: np.ndarray) -> np.ndarray:
    return np.cov(samples, rowvar=False, bias=False)

def run_sgd(V: np.ndarray, y: np.ndarray, alpha_ast, alpha_init, config) -> Tuple[np.ndarray, np.ndarray]:
    steps = config["steps"]
    eta0 = config["step_size"]
    eta_pow = config["decay_exponent"]

    alpha = alpha_init.copy()
    alpha_sum = alpha.copy()
    alpha_error = np.zeros(steps + 1)
    alpha_ast_norm = np.sum(alpha_ast ** 2)

    # initial value
    alpha_error[0] = np.sum((alpha - alpha_ast) ** 2) / alpha_ast_norm

    for t in range(1, steps + 1):
        # shift back by 1 since we're including the initial point
        v_t = V[t - 1]
        y_t = y[t - 1]

        # apply inverse link function
        prob = sigmoid(v_t.T @ alpha)
        grad = (prob - y_t) * v_t
        eta = eta0 / (1 + 1e-3 * t ** (eta_pow))

        alpha = alpha - eta * grad
        alpha_sum += alpha
        alpha_error[t] = np.sum((alpha_sum / (t + 1) - alpha_ast) ** 2) / alpha_ast_norm

    return alpha_error

def run_experiment(config) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed=config["seed"])

    n = config["data_points"]
    p = config["teacher_dim"]
    d = config["student_dim"]

    print("data points", n)
    print("teacher dim", p)
    print("student dim", d)

    trials = config["trials"]
    steps = config["steps"]
    indices = np.arange(n)

    n_checkpoints = np.arange(0, steps, 1)

    all_alpha_errors = np.zeros((trials, len(n_checkpoints)))
    all_beta_errors = np.zeros((trials, len(n_checkpoints)))

    for trial in range(trials):
        # sample a big dataset
        theta_true = rng.standard_normal(p) / np.sqrt(p)

        X = rng.standard_normal((n, p)) / np.sqrt(p)
        V = X[:, 0:d]
        probs = sigmoid(X @ theta_true)
        y = rng.binomial(1, probs)

        theta_init = np.zeros(p)
        init = np.zeros(d)

        # first compute theta_hat (since we're working with a finite dataset)
        nll_theta = (lambda theta: np.sum(- (y * np.log(sigmoid(X @ theta)) + (1 - y) * np.log(1 - sigmoid(X @ theta)))))
        grad_theta = (lambda theta: X.T @ (sigmoid(X @ theta) - y))
        theta_hat = minimize(nll_theta, theta_init.copy(), jac=grad_theta, method="L-BFGS-B").x

        # use theta_hat to generate teacher labels
        y_hat = sigmoid(X @ theta_hat)

        # now compute optimal students
        nll_alpha = (lambda alpha: np.sum(- (y * np.log(sigmoid(V @ alpha)) + (1 - y) * np.log(1 - sigmoid(V @ alpha)))))
        nll_beta = (lambda beta: np.sum(- (y_hat * np.log(sigmoid(V @ beta)) + (1 - y_hat) * np.log(1 - sigmoid(V @ beta)))))

        grad_alpha = (lambda alpha: V.T @ (sigmoid(V @ alpha) - y))
        grad_beta = (lambda beta: V.T @ (sigmoid(V @ beta) - y_hat))

        alpha_hat = minimize(nll_alpha, init.copy(), jac=grad_alpha, method="L-BFGS-B").x
        beta_hat = minimize(nll_beta, init.copy(), jac=grad_beta, method="L-BFGS-B").x

        # sample SGD order of samples
        trial_indices = rng.choice(indices, steps, replace=True)
        V_trial = V[trial_indices]
        y_trial = y[trial_indices]
        y_hat_trial = y_hat[trial_indices]  # this is the teacher labels

        alpha_error = run_sgd(V_trial, y_trial, alpha_hat.copy(), init.copy(), config)
        beta_error = run_sgd(V_trial, y_hat_trial, beta_hat.copy(), init.copy(), config)

        all_alpha_errors[trial, :] = alpha_error[n_checkpoints]
        all_beta_errors[trial, :] = beta_error[n_checkpoints]

    alpha_errors_avg = np.sum(all_alpha_errors, axis=0) / trials
    beta_errors_avg = np.sum(all_beta_errors, axis=0) / trials

    errors_fn = osp.join(config["data_dir"], f"d_{config['student_dim']}", f"step_{step_size}.npy")
    np.save(errors_fn, [alpha_errors_avg, beta_errors_avg])

if __name__ == "__main__":
    parser = argparse.ArgumentParser("logistic")
    parser.add_argument("student_dim", type=int)

    # student_dims = [3, 5, 7, 9]

    args = parser.parse_args()
    student_dim = args.student_dim

    data_dir = osp.join(os.getcwd(), "data")
    pathlib.Path(osp.join(data_dir, f"d_{student_dim}")).mkdir(exist_ok=True, parents=True)

    config = {
        "seed": 57,
        "steps": 10_000,
        "trials": 100,
        "data_points": 1_000,
        "decay_exponent": 0.6,
        "teacher_dim": 10,
        "data_dir": data_dir
    }

    config["student_dim"] = student_dim

    step_sizes = [10**(-3), 10**(-2.5), 10**(-2), 10**(-1.5), 10**(-1), 10**(-0.5), 10**(0.0)]

    for step_size in step_sizes:
        config["step_size"] = step_size
        run_experiment(config)