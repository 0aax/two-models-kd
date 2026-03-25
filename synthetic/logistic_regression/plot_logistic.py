import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
from textwrap import wrap

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = "14"

def get_best_step(student_dims, step_sizes, data_dir, no_distill=True):
    best_steps = []

    for i, student_dim in enumerate(student_dims):
        final_error = [0 for _ in range(len(step_sizes))]

        for j, step_size in enumerate(step_sizes):
            data_file = osp.join(data_dir, f"d_{student_dim}", f"step_{step_size}.npy")
            errors = np.load(data_file)

            if no_distill:  # first entry corresponds to standard supervision
                final_error[j] += errors[0][-1]
            else:           # second entry corresponds to distillation
                final_error[j] += errors[1][-1]

        best_step_idx = final_error.index(min(final_error))
        best_steps.append(step_sizes[best_step_idx])

    return best_steps

if __name__ == "__main__":
    data_dir = osp.join(os.getcwd(), "data")
    student_dims = [3, 5, 7, 9]
    step_sizes = [10**(-3), 10**(-2.5), 10**(-2), 10**(-1.5), 10**(-1), 10**(-0.5), 10**(0.0)]
    steps = 10_000

    colors = ["coral", "aqua", "green", "magenta"]

    no_distill_steps = get_best_step(student_dims, step_sizes, data_dir, no_distill=True)
    distill_steps = get_best_step(student_dims, step_sizes, data_dir, no_distill=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))

    n_checkpoints = np.arange(0, steps, 1)

    # plot where both methods have the same step size (chosen optimally for standard supervision)
    for i, student_dim in enumerate(student_dims):
        best_step = no_distill_steps[i]

        data_file = osp.join(data_dir, f"d_{student_dim}", f"step_{best_step}.npy")
        errors_step = np.load(data_file)

        errors_alpha = errors_step[0]
        errors_beta = errors_step[1]

        ax1.plot(n_checkpoints, errors_beta, linestyle="-", color=colors[i], label=r"$d$={student_dim}".format(student_dim=student_dim))
        ax1.plot(n_checkpoints, errors_alpha, linestyle="--", color=colors[i])

    ax1.set_title("\n".join(wrap(r"Optimal $\eta_0$ standard supervision.", width=40)))
    ax1.set_ylabel(r"$|\overline{\gamma}_t - \hat{\gamma}|_2/|\hat{\gamma}|_2$")
    ax1.set_xlabel(r"Step $(t)$")

    ax1.grid(linestyle='dotted')
    ax1.set_yscale("log")

    # plot where optimal step size is used for each method
    for i, student_dim in enumerate(student_dims):
        best_no_distill_step = no_distill_steps[i]
        best_distill_step = distill_steps[i]

        data_file_no_distill_step = osp.join(data_dir, f"d_{student_dim}", f"step_{best_no_distill_step}.npy")
        data_file_distill_step = osp.join(data_dir, f"d_{student_dim}", f"step_{best_distill_step}.npy")
        
        errors_no_distill_step = np.load(data_file_no_distill_step)
        errors_distill_step = np.load(data_file_distill_step)

        errors_alpha = errors_no_distill_step[0]
        errors_beta = errors_distill_step[1]

        ax2.plot(n_checkpoints, errors_beta, linestyle="-", color=colors[i], label=r"$d$={student_dim}".format(student_dim=student_dim))
        ax2.plot(n_checkpoints, errors_alpha, linestyle="--", color=colors[i])

    ax2.set_title("\n".join(wrap(r"Optimal $\eta_0$ for both sequences.", width=40)))
    ax2.set_ylabel(r"$|\overline{\gamma}_t - \hat{\gamma}|_2 / |\hat{\gamma}|_2$")
    ax2.set_xlabel(r"Step $(t)$")

    ax2.grid(linestyle='dotted')
    ax2.set_yscale("log")

    ax2.plot([], [], linestyle="--", color="gray", label="Baseline")
    ax2.plot([], [], linestyle="-", color="gray", label="Distill")
    ax2.legend(frameon=True, facecolor='none', edgecolor='black', loc='upper left', bbox_to_anchor=(1.05, 1))

    plt.tight_layout()
    plt.savefig(f"logistic_param_error.pdf")