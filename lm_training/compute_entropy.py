from accelerate import Accelerator
from datasets import load_dataset
import numpy as np
import pathlib
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.utils.data as torch_data
from transformers import GPT2LMHeadModel, set_seed
import argparse

set_seed(57)

parser = argparse.ArgumentParser("save_dataset")
parser.add_argument("shard_idx", help="Shard index.", type=int)
args = parser.parse_args()

shard_idx = args.shard_idx
print("shard idx", shard_idx)

save_directory = f"/c4_train_entropy/shard_{shard_idx:02d}"
accelerator = Accelerator(mixed_precision="fp16")

total_chunks = 876
total_splits = 12
file_range = total_chunks // total_splits
start_file_id = shard_idx * file_range
end_file_id = (shard_idx + 1) * file_range

print("file ranges", start_file_id, end_file_id)

# load only specified range of chunks
data_files = [f"/c4_indexed/data-{id:05d}-of-00876.arrow" for id in range(start_file_id, end_file_id)]
dataset = load_dataset("arrow", data_files=data_files)
print("dataset", dataset)

dataset.set_format("torch", columns=["input_ids", "attention_mask", "index"])
print("dataset formatted", dataset)
dataset = dataset["train"]

dataloader = torch_data.DataLoader(dataset, batch_size=48, num_workers=8, shuffle=False,
                                   pin_memory=True, persistent_workers=True)

model = GPT2LMHeadModel.from_pretrained("/c4_gpt2_model/gpt2-medium-lr0.0005-seed57/checkpoint-13542")
model.eval()

# setup dataloader to be multi-gpu
model, dataloader = accelerator.prepare(model, dataloader)
rank = accelerator.process_index
pathlib.Path(f"{save_directory}/rank_{rank:02d}").mkdir(parents=True, exist_ok=True)

index_store = []
entropy_store = []

progress_bar = tqdm(dataloader, disable=(not accelerator.is_local_main_process))

buffer_size = 500_000

for i, batch in enumerate(progress_bar):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1) # softmax over vocabulary dimension

        per_token_entropy = torch.sum(-(probs * torch.log(probs)), dim=-1)
        per_token_entropy = per_token_entropy * attention_mask

        valid_tokens = torch.sum(attention_mask, dim=-1)
        avg_entropy = torch.sum(per_token_entropy, dim=-1) / valid_tokens

        index_store.append(batch["index"])
        entropy_store.append(avg_entropy)

    if (i + 1) % buffer_size == 0:
        buffer_index = torch.cat(index_store).cpu().numpy()
        buffer_entropy = torch.cat(entropy_store).cpu().numpy()

        np.savez(f"{save_directory}/rank_{rank:02d}/chunk_{i // buffer_size:04d}.npz",
                 index=buffer_index, entropy=buffer_entropy)

        entropy_store = []
        index_store = []

        print(f"Saved buffer {(i + 1) // buffer_size}")

# remaining samples
if len(index_store) != 0:
    buffer_index = torch.cat(index_store).cpu().numpy()
    buffer_entropy = torch.cat(entropy_store).cpu().numpy()

    np.savez(f"{save_directory}/rank_{rank:02d}/chunk_{(i // buffer_size) + 1:04d}.npz",
             index=buffer_index, entropy=buffer_entropy)

    entropy_store = []
    index_store = []

    print(f"Saved buffer {((i + 1) // buffer_size) + 1}")