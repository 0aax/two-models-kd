import numpy as np
import os
import os.path as osp
import pathlib
from tqdm import tqdm
from typing import Tuple
import argparse

def run_gd(M: np.ndarray, x_ast, x_init, config) -> Tuple[np.ndarray, np.ndarray]:
    steps = config["steps"]
    eta = config["step_size"]

    x = x_init.copy()
    x_error = np.zeros(steps + 1)

    # initial error value
    x_error[0] = np.sum((x - x_ast) ** 2)

    for t in range(1, steps + 1):
        grad = (x @ x.T - M) @ x
        x = x - eta * grad
        x_error[t] = np.sum((x - x_ast) ** 2)

    return x_error

def run_experiment(config) -> None:
    rng = np.random.default_rng(seed=config["seed"])

    n = config["full_rank"]
    r = config["teacher_rank"]

    lambda_r = config["lambda_r"]
    lambda_n = config["lambda_n"]

    # make a diagonal matrix
    lambda_i = np.hstack((np.array([1]), np.linspace(0.8, 0.75, r-1), np.linspace(lambda_r, lambda_n, n-r)))
    lambda_hat = np.hstack((lambda_i[:r], np.zeros(n-r)))
    M = np.diag(lambda_i)
    M_hat = np.diag(lambda_hat)

    print("lambdas shape and max/min", lambda_i.shape, np.max(lambda_i), np.min(lambda_i))

    trials = config["trials"]
    steps = config["steps"]

    n_checkpoints = np.arange(0, steps, 1)

    all_w_errors = np.zeros((trials, len(n_checkpoints)))
    all_u_errors = np.zeros((trials, len(n_checkpoints)))

    for trial in tqdm(range(trials)):
        # random init
        init = rng.standard_normal((n, 1)) / np.sqrt(n) * 0.001

        # now define optimal point, setting the sign to be correct
        if init[0, 0] > 0:
            lambda_1 = np.sqrt(lambda_i[0])
        else:
            lambda_1 = -np.sqrt(lambda_i[0])

        # shared global optimum: \pm \sqrt{\lambda_1} * e_1
        x_ast = lambda_1 * np.eye(n, 1)

        w_error = run_gd(M, x_ast.copy(), init.copy(), config)
        u_error = run_gd(M_hat, x_ast.copy(), init.copy(), config)

        all_w_errors[trial, :] = w_error[n_checkpoints]
        all_u_errors[trial, :] = u_error[n_checkpoints]

    w_errors_avg = np.sum(all_w_errors, axis=0) / trials
    u_errors_avg = np.sum(all_u_errors, axis=0) / trials

    errors_fn = osp.join(config["data_dir"], f"{config['step_size']}_distance.npy")
    np.save(errors_fn, [w_errors_avg, u_errors_avg])

if __name__ == "__main__":
    parser = argparse.ArgumentParser("rank1")
    parser.add_argument("lambda_r", type=float)
    parser.add_argument("teacher_rank", type=int)

    # lambda_rs = [0.70, 0.65, 0.6, 0.55]
    # teacher_ranks = [2, 4, 8, 16, 32]

    args = parser.parse_args()
    lambda_r = args.lambda_r
    teacher_rank = args.teacher_rank

    data_dir = osp.join(os.getcwd(), "data", f"rank_{teacher_rank}", f"eigenval_{lambda_r}")
    pathlib.Path(data_dir).mkdir(exist_ok=True, parents=True)

    config = {
        "seed": 57,
        "steps": 200,
        "trials": 100,
        "full_rank": 500,
        "data_dir": data_dir
    }

    config["lambda_r"] = lambda_r
    config["lambda_n"] = lambda_r - 0.05
    config["teacher_rank"] = teacher_rank

    step_sizes = [10 ** (-2), 10 ** (-1.5), 10 ** (-1.0), 10 ** (-0.5), 10 ** (0.0)]

    for step_size in step_sizes:
        config["step_size"] = step_size
        run_experiment(config)