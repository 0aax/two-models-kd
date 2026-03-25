from datasets import load_from_disk
import pandas as pd

entropy_path = "/c4_train_entropy"
entropy_model_data_path = "/c4_train_teacher_7.1B"

dataset = load_from_disk(entropy_model_data_path)
print("dataset", dataset)

used_indices = dataset["index"]

entropy_table = pd.read_csv(f"{entropy_path}/entropy_sorted.csv")
print("original rows", len(entropy_table))
cleaned_table = entropy_table[~entropy_table["index"].isin(used_indices)]

cleaned_table.sort_values(by=["entropy"], inplace=True)
print("cleaned rows", len(cleaned_table))
cleaned_table.to_csv(f"{entropy_path}/entropy_sorted_cleaned.csv", index=False)