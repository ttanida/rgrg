import os

from datasets import Dataset
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

from custom_image_word_dataset import CustomImageWordDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PERCENTAGE_OF_TRAIN_SET_TO_USE = 0.2
# PERCENTAGE_OF_VAL_SET_TO_USE = 0.5
PERCENTAGE_OF_TRAIN_SET_TO_USE = 0.005
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


# don't set batched=True, otherwise phrases that are empty will not get a eos token for some reason (see tokenize_function)
tokenized_train_dataset = raw_train_dataset.map(tokenize_function)
tokenized_val_dataset = raw_val_dataset.map(tokenize_function)

tokenized_train_dataset = tokenized_train_dataset.remove_columns(["phrases"])
tokenized_val_dataset = tokenized_val_dataset.remove_columns(["phrases"])

# custom dataset also returns image feature vectors
train_dataset_complete = CustomImageWordDataset("train", tokenized_train_dataset)
val_dataset_complete = CustomImageWordDataset("val", tokenized_val_dataset)

print("*******")
print(train_dataset_complete[18])
print(train_dataset_complete[19])
print(train_dataset_complete[20])
print("*******")


class CustomDataCollatorWithPadding:
    def __init__(self, tokenizer, padding):
        self.tokenizer = tokenizer
        self.padding = padding

    def __call__(self, batch: list[dict[str]]):
        # discard samples from batch where __getitem__ from custom_image_word_dataset failed (i.e. returned None)
        # otherwise, whole training loop would stop
        batch = list(filter(lambda x: x is not None, batch))  # filter out samples that are None

        image_hidden_dim = batch[0]["image_hidden_states"].size(0)
        image_hidden_states_batch = torch.empty(size=(len(batch), image_hidden_dim))

        for i, sample in enumerate(batch):
            # remove image_hidden_states vectors from batch and store them in dedicated image_hidden_states_batch matrix
            image_hidden_states_batch[i] = sample.pop("image_hidden_states")

        # batch only contains samples with input_ids and attention_mask keys
        # the tokenizer will turn the batch variable into a single dict with input_ids and attention_mask keys,
        # that map to tensors of shape [batch_size x (longest) seq_len (in batch)] respectively
        batch = self.tokenizer.pad(batch, padding=self.padding, return_tensors="pt")

        # add the image_hidden_states_batch tensor to the dict
        batch["image_hidden_states"] = image_hidden_states_batch

        return batch

BATCH_SIZE = 3
NUM_WORKERS = 12

custom_collate_with_padding = CustomDataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

train_loader = DataLoader(train_dataset_complete, collate_fn=custom_collate_with_padding, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset_complete, collate_fn=custom_collate_with_padding, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
