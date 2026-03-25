import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = "14"

lr = 1e-3
rel_file = f"c4_data/rel_{lr}.npy"

index_to_rel = np.load(rel_file)

print(index_to_rel.shape)

fig, ax = plt.subplots(1, 3, figsize=(11, 3), sharex=True)

plotting_fracs = [0.5, 0.75, 1.0]
subset_entropies = [1.985, 2.857, 3.052, 3.233, 3.816]
colors = ["coral", "aqua", "green", "magenta", "pink"]

for frac_id, frac in enumerate(plotting_fracs):
    rel_frac = index_to_rel[:, :, frac_id].T

    for run_count in range(rel_frac.shape[0]):
        ax[frac_id].plot(subset_entropies, rel_frac[run_count], color=colors[run_count], linestyle="--", marker="s")
    
    ax[frac_id].set_xlabel("Per-token entropy")

    ax[frac_id].title.set_text(f"Fraction of training: {frac}")
    ax[frac_id].grid(linestyle='dotted')

ax[0].set_ylabel("Ratio of steps")

plt.tight_layout()
plt.savefig(f"rel_val_loss_individual_{lr}.pdf")