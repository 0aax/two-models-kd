import glob
import numpy as np
import pandas as pd

save_directory = "/c4_train_entropy"

all_idx = []
all_ent = []
for fn in sorted(glob.glob(f"{save_directory}/shard*/rank*/chunk_*.npz")):
    data = np.load(fn)
    all_idx.append(data["index"])
    all_ent.append(data["entropy"])

idx = np.concatenate(all_idx)
ent = np.concatenate(all_ent)

df = pd.DataFrame({"index": idx, "entropy": ent})
df.sort_values(by=["entropy"], inplace=True)
print("rows", len(df))
df.to_csv(f"{save_directory}/entropy_sorted.csv", index=False)