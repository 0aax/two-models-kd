from datasets import load_dataset
import pathlib
import pandas as pd
import argparse

# directories to load and save from
entropy_path = "/c4_train_entropy/entropy_sorted_cleaned.csv"
save_path = "/c4_train_shards"
pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

# we load only parts of the dataset
parser = argparse.ArgumentParser("split_dataset")
parser.add_argument("shard_idx", help="Shard index.", type=int)
args = parser.parse_args()

shard_idx = args.shard_idx
print("shard idx", shard_idx)

total_chunks = 876
total_splits = 6
file_range = total_chunks // total_splits
start_file_id = shard_idx * file_range
end_file_id = (shard_idx + 1) * file_range

print("file ranges", start_file_id, end_file_id)

# load only specified range of chunks
data_files = [f"/c4_indexed/data-{id:05d}-of-00876.arrow" for id in range(start_file_id, end_file_id) if id != 291] # seems like 291 got corrupted when I moved data
dataset = load_dataset("arrow", data_files=data_files)["train"]
print("dataset", dataset)

# make a map from the original indices to the current dataset's indicies
available_indices = dataset["index"]
index_to_position = {val: i for i, val in enumerate(available_indices)}

# set up which subset we're going to save
entropy_table = pd.read_csv(entropy_path)
sorted_example_idx = entropy_table["index"].to_list()
total_subsets = 13
subsets = [0, 3, 6, 9, 12]  # choose subsets to save
subset_lengths = len(sorted_example_idx) // total_subsets

for i in subsets:
    print("subset", i)
    start_idx = i * subset_lengths
    end_idx = min((i + 1) * subset_lengths, len(sorted_example_idx) - 1)
    print("start", start_idx, "end", end_idx)

    # indices for this subset
    subset_indices = sorted_example_idx[start_idx: end_idx]
    print("subset size", len(subset_indices))

    # remap indices to correspond to current dataset, also ignore all indices that are not in the current dataset
    mapped_subset_indices = [index_to_position[i] for i in subset_indices if i in index_to_position]
    
    dataset_subset = dataset.select(indices=mapped_subset_indices)
    print("subset in current shard", dataset_subset)
    dataset_subset.save_to_disk(f"{save_path}/shard_{shard_idx:02d}/subset_{i:02d}_train")