import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
from textwrap import wrap

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = "18"

def get_best_step(student_dims, step_sizes, data_dir, trials, no_distill=True):
    best_steps = []

    for i, student_dim in enumerate(student_dims):
        final_error = [0 for _ in range(len(step_sizes))]

        for j, step_size in enumerate(step_sizes):
            for trial in range(trials):
                data_file = osp.join(data_dir, f"d_{student_dim}", f"lr_{step_size}", f"trial_{trial}.npy")
                errors = np.load(data_file)

                if no_distill:  # first entry corresponds to standard supervision
                    final_error[j] += errors[0][-1]
                else:           # second entry corresponds to distillation
                    final_error[j] += errors[1][-1]

            final_error[j] /= trials

        print(min(final_error))
        best_step_idx = final_error.index(min(final_error))
        print(best_step_idx)
        print(step_sizes[best_step_idx])
        best_steps.append(step_sizes[best_step_idx])

    return best_steps

if __name__ == "__main__":
    data_dir = osp.join(os.getcwd(), "data")
    student_dims = [100, 300, 500]
    step_sizes = [10**(-1), 10**(-0.75), 10**(-0.5), 10**(-0.25), 10**(0.0)]
    steps = 500_000
    trials = 10

    colors = ["coral", "aqua", "green", "magenta", "pink"]

    no_distill_best_lrs = get_best_step(student_dims, step_sizes, data_dir, trials, no_distill=True)
    distill_best_lrs = get_best_step(student_dims, step_sizes, data_dir, trials, no_distill=False)

    #### Plot best supervised

    fig, ax = plt.subplots(figsize=(6, 5.5))
    
    n_checkpoints = np.arange(0, steps, 1)

    for i, student_dim in enumerate(student_dims):
        best_lr = no_distill_best_lrs[i]
        errors_alpha, errors_beta = np.zeros((len(n_checkpoints), )), np.zeros((len(n_checkpoints), ))

        for trial in range(trials):
            data_file = osp.join(data_dir, f"d_{student_dim}", f"lr_{best_lr}", f"trial_{trial}.npy")
            errors = np.load(data_file)

            errors_alpha += errors[0]
            errors_beta += errors[1]

        errors_alpha /= trials
        errors_beta /= trials

        ax.plot(n_checkpoints, errors_beta, linestyle="-", color=colors[i], label=r"$d$={student_dim}".format(student_dim=student_dim))
        ax.plot(n_checkpoints, errors_alpha, linestyle="--", color=colors[i])

    ax.plot([], [], linestyle="--", color="gray", label="Baseline")
    ax.plot([], [], linestyle="-", color="gray", label="Distill")

    # reorder elements of legend to that the line markers are the bottom row
    handles, labels = ax.get_legend_handles_labels()
    order_handles = [handles[0], handles[3], handles[1], handles[4], handles[2]]
    order_labels = [labels[0], labels[3], labels[1], labels[4], labels[2]]

    ax.set_title("\n".join(wrap(r"Parameter error varying student dimension.", width=45)))
    ax.set_ylabel(r"$|\overline{\gamma}_t - \hat{\gamma}|_2 / |\hat{\gamma}|_2$")
    ax.set_xlabel(r"Step $(t)$")

    ax.grid(linestyle='dotted')
    ax.set_yscale("log")
    ax.legend(order_handles, order_labels, frameon=True, ncol=3, facecolor='none', edgecolor='black', loc='upper center', bbox_to_anchor=(0.5, -0.3))

    plt.subplots_adjust(bottom=0.35)
    plt.tight_layout()
    plt.savefig(f"fmnist_supervised_optimal.pdf")

    #### Plot both optimal step

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, student_dim in enumerate(student_dims):
        best_baseline_lr = no_distill_best_lrs[i]
        best_distill_lr = distill_best_lrs[i]

        errors_alpha, errors_beta = np.zeros((len(n_checkpoints), )), np.zeros((len(n_checkpoints), ))

        for trial in range(trials):
            data_file_baseline = osp.join(data_dir, f"d_{student_dim}", f"lr_{best_baseline_lr}", f"trial_{trial}.npy")
            data_file_distill = osp.join(data_dir, f"d_{student_dim}", f"lr_{best_distill_lr}", f"trial_{trial}.npy")
            
            errors_baseline = np.load(data_file_baseline)
            errors_distill = np.load(data_file_distill)

            errors_alpha += errors_baseline[0]
            errors_beta += errors_distill[1]

        errors_alpha /= trials
        errors_beta /= trials

        ax.plot(n_checkpoints, errors_beta, linestyle="-", color=colors[i], label=r"$d$={student_dim}".format(student_dim=student_dim))
        ax.plot(n_checkpoints, errors_alpha, linestyle="--", color=colors[i])

    ax.plot([], [], linestyle="--", color="gray", label="Baseline")
    ax.plot([], [], linestyle="-", color="gray", label="Distill")

    ax.set_title("\n".join(wrap(r"Parameter error varying student dimension, optimal $\eta_0$ for both sequences.", width=45)))
    ax.set_ylabel(r"$|\overline{\gamma}_t - \hat{\gamma}|_2 / |\hat{\gamma}|_2$")
    ax.set_xlabel(r"Step $(t)$")

    ax.grid(linestyle='dotted')
    ax.set_yscale("log")
    ax.legend(frameon=True, facecolor='none', edgecolor='black', loc='upper left', bbox_to_anchor=(1.05, 1))

    plt.tight_layout()
    plt.savefig(f"fmnist_both_optimal.pdf")
