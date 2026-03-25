import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
from textwrap import wrap

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = "18"

colors = ["coral", "aqua", "green", "magenta"]
ratios = [2/10, 4/10, 6/10, 8/10]

data_dir = osp.join(os.getcwd(), "asymp_cov")

config = {
    "seed": 57,
    "trials": 100,
    "data_points": 1_000,
    "teacher_dim": 100
}

fig, ax = plt.subplots(figsize=(6, 5))

for i, dim_ratio in enumerate(ratios):
    student_dim = int(config["teacher_dim"] * dim_ratio)
    student_dimension_fn = osp.join(data_dir, f"d_{student_dim}.npy")
    all_vals = np.load(student_dimension_fn)

    # first column is the label variance average
    # second column is Tr(cov alpha)
    # third column is Tr(cov beta)
    ax.plot(all_vals[:, 0].T, all_vals[:, 1].T / all_vals[:, 2].T, linestyle="-", marker="s", color=colors[i], label=r"$d$={d}".format(d=student_dim))

plt.ylabel(r"ARE Trace")
plt.xlabel(r"Label variance")
plt.title("\n".join(wrap(r"Asymptotic relative efficiency under increasing label variance.", width=40)))

plt.grid(linestyle='dotted')
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig("asymptotic_rel_efficiency.pdf")