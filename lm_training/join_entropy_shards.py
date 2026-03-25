from datasets import load_dataset
import pathlib
import argparse
import glob

save_path = "/c4_train"

num_train_samples = 5_987_485 # set this so it's equal across subsets
num_test_samples = 29_937
pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

# choose which subset to save
parser = argparse.ArgumentParser("join_dataset")
parser.add_argument("subset_idx", help="Subset index.", type=int)
args = parser.parse_args()

subset_idx = args.subset_idx
print("subset idx", subset_idx)

# gather all files
total_shards = 6
data_files = []

for shard_idx in range(total_shards):
    fn_prefix = f"/c4_train_shards/shard_{shard_idx:02d}/subset_{subset_idx:02d}_train"
    shard_data_files = glob.glob(f"{fn_prefix}/data-*.arrow")
    print(f"data files in {shard_idx:02d}", len(shard_data_files))
    data_files += [f"{fn}" for fn in shard_data_files]

dataset = load_dataset("arrow", data_files=data_files)["train"]
print("dataset", dataset)

train_val = dataset.train_test_split(train_size=num_train_samples, test_size=num_test_samples, shuffle=True, seed=57)

train_dataset = train_val["train"]
eval_dataset = train_val["test"]

print("train", train_dataset)
print("eval", eval_dataset)

train_dataset.save_to_disk(f"{save_path}/subset_{subset_idx:02d}_train")
eval_dataset.save_to_disk(f"{save_path}/subset_{subset_idx:02d}_eval")