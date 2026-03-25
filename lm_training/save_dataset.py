from datasets import load_from_disk
import argparse

parser = argparse.ArgumentParser("save_dataset")
parser.add_argument("shard_idx", help="Shard index.", type=int)
args = parser.parse_args()

shard_idx = args.shard_idx
print("shard idx", shard_idx)

dataset = load_from_disk("/c4_indexed")

print("dataset", dataset)
size_subset = int(7.1e9 / 1024)
num_shards = (len(dataset) // size_subset) * 2
print("total shards", num_shards)

split = dataset.train_test_split(train_size=size_subset, seed=57, shuffle=True)
teacher, remaining = split["train"], split["test"]