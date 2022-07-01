from copy import deepcopy
import os

# import evaluate
from datasets import Dataset
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import GPT2Tokenizer
from tqdm import tqdm

from custom_image_word_dataset import CustomImageWordDataset
from gpt2 import DecoderModel

# bertscore_metric = evaluate.load("bertscore")
# bleu_metric = evaluate.load("bleu")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PERCENTAGE_OF_TRAIN_SET_TO_USE = 0.2
PERCENTAGE_OF_VAL_SET_TO_USE = 0.5
# PERCENTAGE_OF_TRAIN_SET_TO_USE = 0.00005
# PERCENTAGE_OF_VAL_SET_TO_USE = 0.0001

path_chest_imagenome_customized = "/u/home/tanida/datasets/chest-imagenome-dataset-customized-full"

# only load the phrases
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

print("\n*******")
print(raw_train_dataset)
print(raw_val_dataset)
print("*******\n")

checkpoint = "healx/gpt-2-pubmed-medium"
tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token


def tokenize_function(example):
    phrase = example["phrases"]
    if len(phrase) == 0:  # if phrase == ""
        phrase = tokenizer.eos_token  # becomes "<|endoftext|>"
    return tokenizer(phrase, truncation=True, max_length=1024)


# don't set batched=True, otherwise phrases that are empty will not get a eos token for some unknown reason
# datasets will consist of the columns "phrases", "input_ids", "attention_mask"
tokenized_train_dataset = raw_train_dataset.map(tokenize_function)
tokenized_val_dataset = raw_val_dataset.map(tokenize_function)

# remove redundant "phrases" column
tokenized_train_dataset = tokenized_train_dataset.remove_columns(["phrases"])
tokenized_val_dataset = tokenized_val_dataset.remove_columns(["phrases"])

# custom dataset will return a dict with keys "input_ids", "attention_mask", "image_hidden_states" when indexed,
# with "image_hidden_states" mapping to a tensor of size 1024 that are the image features of the given bbox image
train_dataset_complete = CustomImageWordDataset("train", tokenized_train_dataset)
val_dataset_complete = CustomImageWordDataset("val", tokenized_val_dataset)


class CustomDataCollatorWithPadding:
    def __init__(self, tokenizer, padding):
        self.tokenizer = tokenizer
        self.padding = padding

    def __call__(self, batch: list[dict[str]]):
        # discard samples from batch where __getitem__ from custom_image_word_dataset failed (i.e. returned None)
        # otherwise, whole training loop would stop
        batch = list(filter(lambda x: x is not None, batch))  # filter out samples that are None

        image_hidden_dim = batch[0]["image_hidden_states"].size(0)

        # initiate a image_hidden_states_batch tensor that will store all image_hidden_states of the batch
        image_hidden_states_batch = torch.empty(size=(len(batch), image_hidden_dim))

        for i, sample in enumerate(batch):
            # remove image_hidden_states vectors from batch and store them in dedicated image_hidden_states_batch tensor
            image_hidden_states_batch[i] = sample.pop("image_hidden_states")

        # batch only contains samples with input_ids and attention_mask keys
        # the tokenizer will turn the batch variable into a single dict with input_ids and attention_mask keys,
        # that map to tensors of shape [batch_size x (longest) seq_len (in batch)] respectively
        batch = self.tokenizer.pad(batch, padding=self.padding, return_tensors="pt")

        # add the image_hidden_states_batch tensor to the dict
        batch["image_hidden_states"] = image_hidden_states_batch

        return batch


BATCH_SIZE = 32
NUM_WORKERS = 12

custom_collate_with_padding = CustomDataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

train_loader = DataLoader(train_dataset_complete, collate_fn=custom_collate_with_padding, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset_complete, collate_fn=custom_collate_with_padding, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)


def train_one_epoch(model, train_dl, optimizer, epoch):
    """
    Train model for 1 epoch.
    Write train loss to tensorboard.

    Args:
        model (nn.Module): The input model to be trained.
        train_dl (torch.utils.data.Dataloder): The train dataloader to train on.
        optimizer (Optimizer): The model's optimizer.
        epoch (int): Current epoch number.

    Returns:
        train_loss (float): Train loss for 1 epoch.
    """
    # training the model on the train set
    model.train()
    train_loss = 0.0

    for batch in tqdm(train_dl):
        # batch is a dict with keys for 'input_ids', 'attention_mask', 'image_hidden_states' (see custom_image_word_dataset)
        input_ids, attention_mask, image_hidden_states = batch.values()

        batch_size = input_ids.size(0)

        input_ids = input_ids.to(device, non_blocking=True)  # shape (batch_size x seq_len)
        attention_mask = attention_mask.to(device, non_blocking=True)  # shape (batch_size x seq_len)
        image_hidden_states = image_hidden_states.to(device, non_blocking=True)  # shape (batch_size x image_hidden_dim) (with image_hidden_dim = 1024)

        # model returns loss and language modeling logits (not needed here), if return_loss=True
        loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, image_hidden_states=image_hidden_states, return_loss=True)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item() * batch_size

    train_loss /= len(train_dl)

    writer.add_scalar("training loss", train_loss, epoch)

    return train_loss


def evaluate_one_epoch(model, val_dl, lr_scheduler, epoch):
    """
    Evaluate model on val set.

    Write to tensorboard:
        - val loss

    Args:
        model (nn.Module): The input model to be evaluated.
        val_dl (torch.utils.data.Dataloder): The val dataloader to evaluate on.
        lr_scheduler (torch.optim.lr_scheduler): The learning rate scheduler to use.
        epoch (int): Current epoch number.

    Returns:
        val_loss (float): Val loss for 1 epoch.
    """
    # evaluating the model on the val set
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_dl):
            input_ids, attention_mask, image_hidden_states = batch.values()

            batch_size = input_ids.size(0)

            input_ids = input_ids.to(device, non_blocking=True)  # shape (batch_size x seq_len)
            attention_mask = attention_mask.to(device, non_blocking=True)  # shape (batch_size x seq_len)
            image_hidden_states = image_hidden_states.to(device, non_blocking=True)  # shape (batch_size x image_hidden_dim) (with image_hidden_dim = 1024)

            # model returns loss and language modeling logits (not needed here), if return_loss=True
            loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, image_hidden_states=image_hidden_states, return_loss=True)

            val_loss += loss.item() * batch_size

    val_loss /= len(val_dl)

    writer.add_scalar("val loss", val_loss, epoch)

    # decrease lr by 1e-1 if val loss has not decreased after certain number of epochs
    lr_scheduler.step(val_loss)

    return val_loss


def print_stats_to_console(
    train_loss,
    val_loss,
    epoch,
):
    print(f"Epoch: {epoch}:")
    print(f"\tTrain loss: {train_loss:.3f}")
    print(f"\tVal loss: {val_loss:.3f}")


def train_model(model, train_dl, val_dl, optimizer, lr_scheduler, epochs, patience, run):
    """
    Train a model on train set and evaluate on validation set.
    Saves best model w.r.t. val loss.

    Parameters
    ----------
    model: nn.Module
        The input model to be trained.
    train_dl: torch.utils.data.Dataloder
        The train dataloader to train on.
    val_dl: torch.utils.data.Dataloder
        The val dataloader to validate on.
    optimizer: Optimizer
        The model's optimizer.
    lr_scheduler: torch.optim.lr_scheduler
        The learning rate scheduler to use.
    epochs: int
        Number of epochs to train for.
    patience: int
        Number of epochs to wait for val loss to decrease.
        If patience is exceeded, then training is stopped early.
    run: int
        Number of current run.

    Returns
    -------
    None, but saves model with the lowest val loss over all epochs.
    """

    lowest_val_loss = np.inf

    # the best_model_state is the one where the val loss is the lowest over all epochs
    best_model_state = None
    num_epochs_without_decrease_val_loss = 0  # parameter to determine early stopping
    num_epochs_without_saving_best_model = 0  # parameter to determine if model should be saved
    save_model_every_k_epochs = 3  # intermittently save the best current model

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_dl, optimizer, epoch)
        val_loss = evaluate_one_epoch(model, val_dl, lr_scheduler, epoch)
        print_stats_to_console(train_loss, val_loss, epoch)

        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            best_epoch = epoch
            best_model_save_path = os.path.join(model_save_path, f"val_loss_{lowest_val_loss:.3f}_epoch_{epoch}.pth")
            best_model_state = deepcopy(model.state_dict())
            num_epochs_without_decrease_val_loss = 0
        else:
            num_epochs_without_decrease_val_loss += 1

        if num_epochs_without_decrease_val_loss >= patience:
            # save the model with the overall lowest val loss
            torch.save(best_model_state, best_model_save_path)
            print(f"\nEarly stopping at epoch ({epoch}/{epochs})!")
            print(f"Lowest overall val loss: {lowest_val_loss} at epoch {best_epoch}")
            return None

        num_epochs_without_saving_best_model += 1

        if num_epochs_without_saving_best_model >= save_model_every_k_epochs:
            torch.save(best_model_state, best_model_save_path)
            num_epochs_without_saving_best_model = 0

    # save the model with the overall lowest val loss
    torch.save(best_model_state, best_model_save_path)
    print("\nFinished training!")
    print(f"Lowest overall val loss: {lowest_val_loss:.3f} at epoch {best_epoch}")
    return None


model_save_path_parent_dir = "/u/home/tanida/weights/decoder_model"

EPOCHS = 30
LR = 1e-4
PATIENCE = 7  # number of epochs to wait before early stopping
PATIENCE_LR_SCHEDULER = 2  # number of epochs to wait for val loss to reduce before lr is reduced by 1e-1

run = 1
model_save_path = os.path.join(model_save_path_parent_dir, f"weights_run_{run}")
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)

model = DecoderModel()
model.to(device, non_blocking=True)
opt = AdamW(model.parameters(), lr=LR)
lr_scheduler = ReduceLROnPlateau(opt, mode="min", patience=PATIENCE_LR_SCHEDULER)
writer = SummaryWriter(log_dir=f"/u/home/tanida/weights/decoder_model/runs/{run}")

print("\nStarting training!\n")

train_model(
    model=model,
    train_dl=train_loader,
    val_dl=val_loader,
    optimizer=opt,
    lr_scheduler=lr_scheduler,
    epochs=EPOCHS,
    patience=PATIENCE,
    run=run
)
