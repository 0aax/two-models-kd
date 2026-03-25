import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
import os.path as osp
from textwrap import wrap

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = "14"

def get_best_step_teacher(teacher_ranks, step_sizes, data_dir):
    # lambda_r = 0.7, teacher rank varies
    best_steps = []

    for i, teacher_rank in enumerate(teacher_ranks):
        final_error = [0 for _ in range(len(step_sizes))]

        for j, step_size in enumerate(step_sizes):
            data_file = osp.join(data_dir, f"rank_{teacher_rank}", "eigenval_0.7", f"{step_size}_distance.npy")
            errors = np.load(data_file)

            # first entry corresponds to standard supervision
            final_error[j] += errors[0][-1]

        best_step_idx = final_error.index(min(final_error))
        best_steps.append(step_sizes[best_step_idx])

    return best_steps

def get_best_step_lambda(lambdas, step_sizes, data_dir):
    # lambda_r varies, teacher rank = 8
    best_steps = []

    for i, lambda_r in enumerate(lambdas):
        final_error = [0 for _ in range(len(step_sizes))]

        for j, step_size in enumerate(step_sizes):
            data_file = osp.join(data_dir, f"rank_8", f"eigenval_{lambda_r}", f"{step_size}_distance.npy")
            errors = np.load(data_file)

            # first entry corresponds to standard supervision
            final_error[j] += errors[0][-1]

        best_step_idx = final_error.index(min(final_error))
        best_steps.append(step_sizes[best_step_idx])

    return best_steps

if __name__ == "__main__":
    data_dir = osp.join(os.getcwd(), "data")
    colors = ["coral", "aqua", "green", "magenta", "pink"]

    step_sizes = [10 ** (-2), 10 ** (-1.5), 10 ** (-1.0), 10 ** (-0.5), 10 ** (0.0)]
    teacher_ranks = [2, 4, 8, 16, 32]
    lambda_rs = [0.70, 0.65, 0.6, 0.55]

    steps = 125     # experiments do 200 GD steps, but at 125 they already close to optimum
    n_checkpoints = np.arange(0, steps, 1)

    teacher_steps = get_best_step_teacher(teacher_ranks, step_sizes, data_dir)
    lambda_steps = get_best_step_lambda(lambda_rs, step_sizes, data_dir)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.75))

    for i, teacher_rank in enumerate(teacher_ranks):
        best_step = teacher_steps[i]
        errors_alpha, errors_beta = np.zeros((len(n_checkpoints), )), np.zeros((len(n_checkpoints), ))

        data_file = osp.join(data_dir, f"rank_{teacher_rank}", "eigenval_0.7", f"{best_step}_distance.npy")
        errors = np.load(data_file)

        errors_alpha += errors[0][:steps]
        errors_beta += errors[1][:steps]

        ax1.plot(n_checkpoints, errors_beta, linestyle="-", color=colors[i], label=r"$r$={teacher_rank}".format(teacher_rank=teacher_rank))
        ax1.plot(n_checkpoints, errors_alpha, linestyle="--", color=colors[i])

    ax1.set_title("\n".join(wrap(r"Varying teacher rank $r$.", width=40)))
    ax1.set_ylabel(r"$|x_t - x^\ast|_2$")
    ax1.set_xlabel(r"Step $(t)$")
    ax1.legend()

    ax1.grid(linestyle='dotted')

    for i, lambda_r in enumerate(lambda_rs):
        lambda_n = lambda_r - 0.05
        best_step = lambda_steps[i]
        errors_alpha, errors_beta = np.zeros((len(n_checkpoints), )), np.zeros((len(n_checkpoints), ))

        data_file = osp.join(data_dir, "rank_8", f"eigenval_{lambda_r}", f"{best_step}_distance.npy")
        errors = np.load(data_file)

        errors_alpha += errors[0][:steps]
        errors_beta += errors[1][:steps]

        if i == 0:  # plot the distill reference
            ax2.plot(n_checkpoints, errors_beta, linestyle="-", color="black", label="Distill")

        ax2.plot(n_checkpoints, errors_alpha, linestyle="--", color=colors[i],
                 label=r"$\lambda_{r+1}$" + r"={lambda_r:.2f}, $\lambda_n$={lambda_n:.2f}".format(lambda_r=lambda_r, lambda_n=lambda_n))

    ax2.set_title("\n".join(wrap(r"Varying eigenvalues $\lambda$.", width=40)))
    ax2.set_ylabel(r"$|x_t - x^\ast|_2$")
    ax2.set_xlabel(r"Step $(t)$")
    legend1 = ax2.legend()

    style_proxies = [
        Line2D([0], [0], color='gray', linestyle='--', label='Baseline'),
        Line2D([0], [0], color='gray', linestyle='-',  label='Distill')
    ]
    legend2 = ax2.legend(handles=style_proxies,
                     frameon=True, facecolor='none', edgecolor='black', loc='upper left', bbox_to_anchor=(1.05, 1))

    ax2.add_artist(legend1)
    ax2.grid(linestyle='dotted')

    plt.tight_layout()
    plt.savefig(f"rank1_param_error.pdf")