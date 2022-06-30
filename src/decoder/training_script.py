import os
from tabnanny import verbose

from datasets import Dataset
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

from custom_image_word_dataset import CustomImageWordDataset

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


checkpoint = "healx/gpt-2-pubmed-medium"
tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token


def tokenize_function(example):
    phrase = example["phrases"]
    if len(phrase) == 0:
        phrase = tokenizer.eos_token  # becomes "<|endoftext|>"
    return tokenizer(phrase, truncation=True, max_length=1024)


tokenized_train_dataset = raw_train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = raw_val_dataset.map(tokenize_function, batched=True )


tokenized_train_dataset = tokenized_train_dataset.remove_columns(["phrases"])
tokenized_val_dataset = tokenized_val_dataset.remove_columns(["phrases"])

# custom dataset also returns image feature vectors
train_dataset_complete = CustomImageWordDataset("train", tokenized_train_dataset)
val_dataset_complete = CustomImageWordDataset("val", tokenized_val_dataset)

# print(train_dataset_complete[0])
# print(train_dataset_complete[1])
# print(train_dataset_complete[2])


def custom_collate_fn(batch):
    # discard sampples from batch where __getitem__ from custom_image_worddataset failed (i.e. return None)
    # otherwise, whole training loop woudl stop
    print(batch)
    batch = list(filter(lambda x: x is not None, batch))

    print(batch)

    input_ids_and_attention_mask = {k: v for k, v in batch.items() if k != "image_hidden_states"}
    padded_input_ids_and_attention_mask = tokenizer.pad(input_ids_and_attention_mask, padding="longest", return_tensors="pt")

    batch["input_ids"] = padded_input_ids_and_attention_mask["input_ids"]
    batch["attention_mask"] = padded_input_ids_and_attention_mask["attention_mask"]
    return torch.utils.data.dataloader.default_collate(batch)

BATCH_SIZE = 64
NUM_WORKERS = 12

train_loader = DataLoader(train_dataset_complete, collate_fn=custom_collate_fn, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset_complete, collate_fn=custom_collate_fn, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

print(next(iter(train_loader)))
