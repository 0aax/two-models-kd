import wandb as wb
import numpy as np
import pathlib
import os
import os.path as osp

api = wb.Api()

runs = {i: {"nd": [], "d": []} for i in range(13)}

lr = 1e-3

runs[0]["nd"] = ["2463s20h", "vesuuh8w", "b4bns7ky", "452x5g4n", "ym55kcpy"]
runs[0]["d"] = ["riwg2emi", "gany6ull", "egjnv84l", "452x5g4n", "ym55kcpy"]

runs[3]["nd"] = ["xzg78zso", "7kyy9alr", "4p5ogwb5", "q6mxi6ni", "wnp2e8lx"]
runs[3]["d"] = ["7mw55exf", "glkleeqx", "bsopmmfn", "hx4d9egt", "b1ul8fdh"]

runs[6]["nd"] = ["phcipyuf", "zts1hp6c", "94fh7jme", "60x1pw2u", "4aahlkdo"]
runs[6]["d"] = ["ji8woqj5", "x3z5oqnu", "05iae9vw", "s4wmvsa9", "9z4wsuzy"]

runs[9]["nd"] = ["kp3pi7x9", "gbf6nvpa", "5g5zrr6u", "b6m5jhwc", "y867pm3c"]
runs[9]["d"] = ["0xk93wb1", "jk2vqpsf", "7rn7slyu", "mtn1yvt3", "9bfy8p9m"]

runs[12]["nd"] = ["ai92fbvf", "lzrpsrj4", "1y3gwesm", "k7t3dyqt", "57n1y5c1"]
runs[12]["d"] = ["6qdshd6s", "q0ej39jp", "b1y30ks0", "z6xlmm3a", "e91r37ys"]

frac_of_training = [0.50, 0.75, 1.0]

all_index_to_frac_d = []
all_index_to_frac_nd = []
all_index_to_rel = []

subset_ids = [0, 3, 6, 9, 12]

for subset_id in subset_ids:
    nd_runs = runs[subset_id]["nd"]
    d_runs = runs[subset_id]["d"]

    total_runs = len(d_runs)

    index_to_frac_nd = np.zeros((total_runs, len(frac_of_training)))
    index_to_frac_d = np.zeros((total_runs, len(frac_of_training)))
    index_to_rel = np.zeros((total_runs, len(frac_of_training)))
    
    for run_count, run_id in enumerate(range(total_runs)):
        run_no_distill = api.run(f"xiea-stanford-university/c4-subset_{subset_id:02d}-05/{nd_runs[run_id]}")
        run_distill = api.run(f"xiea-stanford-university/c4-subset_{subset_id:02d}-05/{d_runs[run_id]}")

        eval_loss_nd = run_no_distill.history(keys=["eval/loss"])
        eval_loss_nd = eval_loss_nd["eval/loss"].to_numpy()
        target_loss = eval_loss_nd[-1]

        eval_loss_d = run_distill.history(keys=["eval/loss"])
        eval_loss_d = eval_loss_d["eval/loss"].to_numpy()

        for i, frac in enumerate(frac_of_training):
            idx_nd_base = int((len(eval_loss_nd) - 1) * frac)

            worse_loss = max(eval_loss_nd[idx_nd_base], eval_loss_d[idx_nd_base])
            
            idx_d = np.where(eval_loss_d <= worse_loss)
            idx_nd = np.where(eval_loss_nd <= worse_loss)

            index_to_frac_nd[run_count, i] += idx_nd_base / len(eval_loss_nd)
            index_to_frac_d[run_count, i] += idx_d[0][0] / len(eval_loss_nd)
            index_to_rel[run_count, i] += idx_d[0][0] / idx_nd[0][0]

    all_index_to_frac_d.append(index_to_frac_d)
    all_index_to_frac_nd.append(index_to_frac_nd)
    all_index_to_rel.append(index_to_rel)

pathlib.Path(osp.join(os.getcwd(), "c4_data")).mkdir(exist_ok=True, parents=True)
np.save(f"c4_data/distill_{lr}.npy", all_index_to_frac_d)
np.save(f"c4_data/no_distill_{lr}.npy", all_index_to_frac_nd)
np.save(f"c4_data/rel_{lr}.npy", all_index_to_rel)