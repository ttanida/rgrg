import os

from datasets import Dataset
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, DataCollatorWithPadding


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PERCENTAGE_OF_TRAIN_SET_TO_USE = 0.2
# PERCENTAGE_OF_VAL_SET_TO_USE = 0.5
PERCENTAGE_OF_TRAIN_SET_TO_USE = 0.001
PERCENTAGE_OF_VAL_SET_TO_USE = 0.001

path_chest_imagenome_customized = "/u/home/tanida/datasets/chest-imagenome-dataset-customized-full"

# reduce memory usage by only using necessary columns and selecting appropriate datatypes
usecols = ["phrases"]

datasets_as_dfs = {dataset: os.path.join(path_chest_imagenome_customized, dataset) + ".csv" for dataset in ["train", "valid"]}
datasets_as_dfs = {dataset: pd.read_csv(csv_file_path, usecols=usecols, keep_default_na=False) for dataset, csv_file_path in datasets_as_dfs.items()}


total_num_samples_train = len(datasets_as_dfs["train"])
new_num_samples_train = int(PERCENTAGE_OF_TRAIN_SET_TO_USE * total_num_samples_train)
datasets_as_dfs["train"] = datasets_as_dfs["train"][:new_num_samples_train]

total_num_samples_val = len(datasets_as_dfs["valid"])
new_num_samples_val = int(PERCENTAGE_OF_VAL_SET_TO_USE * total_num_samples_val)
datasets_as_dfs["valid"] = datasets_as_dfs["valid"][:new_num_samples_val]

raw_train_dataset = Dataset.from_pandas(datasets_as_dfs["train"])
raw_val_dataset = Dataset.from_pandas(datasets_as_dfs["valid"])

print()
print("------")
print("------")
print()

print(raw_train_dataset)
print(raw_val_dataset)

print()
print("------")
print("------")
print()

print(raw_train_dataset[0])
print(raw_train_dataset[1])

print()
print("------")
print("------")
print()

checkpoint = "healx/gpt-2-pubmed-medium"
tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token
# TODO: set end of sequence token?


def tokenize_function(example):
    return tokenizer(example["phrases"], truncation=True, max_length=1024)


tokenized_train_dataset = raw_train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = raw_val_dataset.map(tokenize_function, batched=True)

print()
print("------")
print("------")
print()

tokenized_train_dataset = tokenized_train_dataset.remove_columns(["phrases"])
tokenized_train_dataset = tokenized_train_dataset.add_column("labels", tokenized_train_dataset["input_ids"])

tokenized_val_dataset = tokenized_val_dataset.remove_columns(["phrases"])
tokenized_val_dataset = tokenized_val_dataset.add_column("labels", tokenized_val_dataset["input_ids"])

print(tokenized_train_dataset)
print(tokenized_val_dataset)

print()
print("------")
print("------")
print()

print(tokenized_train_dataset[0])
print(type(tokenized_train_dataset[0]["input_ids"]))
print(tokenized_train_dataset[1])

print()
print("------")
print("------")
print()

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest", return_tensors="pt")

samples = tokenized_train_dataset[:2]
batch = data_collator(samples)

print()
print("------")
print("------")
print()

print(batch)

print()
print("------")
print("------")
print()
