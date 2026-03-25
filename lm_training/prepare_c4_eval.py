from transformers import GPT2TokenizerFast, set_seed
from datasets import load_dataset
from itertools import chain

set_seed(57)

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

dataset_name = "c4"
print("Loading dataset")
dataset = load_dataset(f"allenai/{dataset_name}", "en", split="validation[:5%]", num_proc=24)

print("Tokenizing data!")
def tokenize_data(example):
    encoding = tokenizer(example["text"], padding=False, truncation=False)

    input_ids_with_eos = []
    attn_masks_with_eos = []

    for input_ids, attn_mask in zip(encoding["input_ids"], encoding["attention_mask"]):
        input_ids_with_eos.append(input_ids + [tokenizer.eos_token_id])
        attn_masks_with_eos.append(attn_mask + [1])

    return {
        "input_ids": input_ids_with_eos,
        "attention_mask": attn_masks_with_eos
    }

# check the column names and remove everything!! otherwise concat will not work!!
tokenized_dataset = dataset.map(tokenize_data, batched=True, batch_size=5_000, remove_columns=["text", "timestamp", "url"], num_proc=24)
print("tokens", tokenized_dataset)

print("Concatenating data!")
def concat(examples):
    examples["input_ids"] = [list(chain.from_iterable(examples['input_ids']))] # convert chain to list of tokens
    examples["attention_mask"] = [list(chain.from_iterable(examples['attention_mask']))] # convert chain to list of tokens
    return examples

concat_dataset = tokenized_dataset.map(concat, batched=True, batch_size=5_000, num_proc=24)
print("concat", concat_dataset)

print("Chunking data!")
def chunk(examples):
    chunk_size = 1024
    input_ids_truncated = []
    attention_mask_truncated = []

    for input_ids, attention_mask in zip(examples["input_ids"], examples["attention_mask"]):
        for i in range(0, len(input_ids), chunk_size):
            chunk = input_ids[i: i + chunk_size]

            if len(chunk) == chunk_size:
                input_ids_truncated.append(chunk)
                attention_mask_truncated.append(attention_mask[i: i + chunk_size])

    examples["input_ids"] = input_ids_truncated
    examples["attention_mask"] = attention_mask_truncated

    return examples

chunk_dataset = concat_dataset.map(chunk, batched=True, batch_size=1, num_proc=24)
print("chunk", chunk_dataset)

def index(example, idx):
    return {"index": idx}

print("Adding indices")
indexed_dataset = chunk_dataset.map(index, with_indices=True, batched=True, num_proc=24)
indexed_dataset.save_to_disk(f"/{dataset_name}_eval_indexed")
print("indexed", indexed_dataset)