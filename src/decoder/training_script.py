from copy import deepcopy
import logging
import os
import random

import evaluate
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

from custom_collator import CustomCollatorWithPadding
from custom_image_word_dataset import CustomImageWordDataset
from gpt2 import DecoderModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)

# set the seed value for reproducibility
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# define configurations for training run
RUN = 1
# PERCENTAGE_OF_TRAIN_SET_TO_USE = 0.2
# PERCENTAGE_OF_VAL_SET_TO_USE = 0.5
PERCENTAGE_OF_TRAIN_SET_TO_USE = 5e-5
PERCENTAGE_OF_VAL_SET_TO_USE = 0.0001
BATCH_SIZE = 32
NUM_WORKERS = 12
EPOCHS = 30
LR = 1e-4
EVALUATE_EVERY_K_STEPS = 10000  # how often to evaluate the model on the validation set and log metrics to tensorboard (additionally, model will always be evaluated at end of epoch)
PATIENCE = 15  # number of evaluations to wait before early stopping
PATIENCE_LR_SCHEDULER = 5  # number of evaluations to wait for val loss to reduce before lr is reduced by 1e-1
NUM_BEAMS = 4
MAX_NUM_TOKENS_GENERATE = 300
NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE = 3  # save num_batches_of_... worth of generated sentences with their gt reference phrases to a txt file


def write_sentences_to_file(
        gen_and_ref_sentences_to_save_to_file,
        generated_sentences_folder_path,
        overall_steps_taken):
    generated_sentences_txt_file = os.path.join(generated_sentences_folder_path, f"generated_sentences_step_{overall_steps_taken}")

    # generated_sentences is a list of str
    generated_sentences = gen_and_ref_sentences_to_save_to_file["generated_sentences"]

    # reference_sentences is a list of list of str
    reference_sentences = gen_and_ref_sentences_to_save_to_file["reference_sentences"]

    with open(generated_sentences_txt_file, "w") as f:
        for gen_sent, ref_sent in zip(generated_sentences, reference_sentences):
            f.write(f"Generated sentence: {gen_sent}\n")
            f.write(f"Reference sentence: {ref_sent}\n\n")


def evaluate_model_on_metrics(model, val_dl, tokenizer, generated_sentences_folder_path, overall_steps_taken):
    """
    Evaluate model on BLEU_1 - BLEU_4 and BERTScore, and also write a certain number of generated sentences with their ground-truth
    references to a txt file for manual inspection.

    Args:
        model (nn.Module): The input model to be evaluated.
        val_dl (torch.utils.data.Dataloder): The val dataloader to evaluate on.
        tokenizer (GPT2Tokenizer): Tokenizer used to decode generated ids.
        generated_sentences_folder_path (str): Folder that contains all txt files with generated sentences per evaluation.

    Returns:
        metrics_with_scores (dict): Dict that holds the BLEU_1 - BLEU_4 scores as well as BERTScore
    """
    metrics_with_scores = {f"bleu_{i}": evaluate.load("bleu") for i in range(1, 5)}
    metrics_with_scores["bert_score"] = evaluate.load("bertscore")

    gen_and_ref_sentences_to_save_to_file = {
        "generated_sentences": [],
        "reference_sentences": []
    }

    model.eval()
    with torch.no_grad():
        for num_batch, batch in tqdm(enumerate(val_dl)):
            # reference_phrases is a list of list of str
            _, _, image_hidden_states, reference_sentences = batch.values()

            image_hidden_states = image_hidden_states.to(device, non_blocking=True)  # shape (batch_size x image_hidden_dim) (with image_hidden_dim = 1024)

            beam_search_output = model.generate(image_hidden_states, max_length=MAX_NUM_TOKENS_GENERATE, num_beams=NUM_BEAMS, early_stopping=True)

            # generated_sentences is a list of str
            generated_sentences = tokenizer.batch_decode(beam_search_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            if num_batch < NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE:
                gen_and_ref_sentences_to_save_to_file["generated_sentences"].extend(generated_sentences)
                gen_and_ref_sentences_to_save_to_file["reference_sentences"].extend(reference_sentences)

            for score in metrics_with_scores.values():
                score.add_batch(predictions=generated_sentences, references=reference_sentences)

    write_sentences_to_file(gen_and_ref_sentences_to_save_to_file, generated_sentences_folder_path, overall_steps_taken)

    metrics_with_final_scores = {}
    for score_name, score in metrics_with_scores.items():
        if score_name[:4] == "bleu":
            result = score.compute(max_order=int(score_name[-1]))
            metrics_with_final_scores[score_name] = result["bleu"]
        else:  # bert_score
            result = score.compute(lang="en", device=device)
            avg_precision = np.array(result["precision"]).mean()
            avg_recall = np.array(result["recall"]).mean()
            avg_f1 = np.array(result["f1"]).mean()

            metrics_with_final_scores["bertscore_precision"] = avg_precision
            metrics_with_final_scores["bertscore_recall"] = avg_recall
            metrics_with_final_scores["bertscore_f1"] = avg_f1

    return metrics_with_final_scores


def get_val_loss(model, val_dl):
    """
    Args:
        model (nn.Module): The input model to be evaluated.
        val_dl (torch.utils.data.Dataloder): The val dataloader to evaluate on.

    Returns:
        val_loss (float): Val loss for val set.
    """
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_dl):
            input_ids, attention_mask, image_hidden_states, _ = batch.values()

            batch_size = input_ids.size(0)

            input_ids = input_ids.to(device, non_blocking=True)  # shape (batch_size x seq_len)
            attention_mask = attention_mask.to(device, non_blocking=True)  # shape (batch_size x seq_len)
            image_hidden_states = image_hidden_states.to(device, non_blocking=True)  # shape (batch_size x image_hidden_dim) (with image_hidden_dim = 1024)

            # model only returns loss if return_loss=True
            loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_hidden_states=image_hidden_states,
                return_loss=True,
            )

            val_loss += loss.item() * batch_size

    val_loss /= len(val_dl)

    return val_loss


def log_stats_to_console(
    train_loss,
    val_loss,
    epoch,
):
    log.info(f"Epoch: {epoch}:")
    log.info(f"\tTrain loss: {train_loss:.3f}")
    log.info(f"\tVal loss: {val_loss:.3f}")


def train_model(
    model,
    train_dl,
    val_dl,
    optimizer,
    lr_scheduler,
    epochs,
    patience,
    weights_folder_path,
    writer,
    tokenizer,
    generated_sentences_folder_path
):
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
    weights_folder_path: str
        Path to folder where best weights will be saved.
    writer: torch.utils.tensorboard.SummaryWriter
        Writer for logging values to tensorboard.

    Returns
    -------
    None, but saves model with the lowest val loss over all epochs.
    """

    lowest_val_loss = np.inf

    # the best_model_state is the one where the val loss is the lowest over all evaluations
    best_model_state = None

    # parameter to determine early stopping
    num_evaluations_without_decrease_val_loss = 0

    num_epochs_without_saving_best_model = 0  # parameter to determine if model should be saved
    save_model_every_k_epochs = 3  # intermittently save the best current model

    overall_steps_taken = 0  # for logging to tensorboard

    for epoch in range(epochs):
        train_loss = 0.0
        steps_taken = 0
        for num_batch, batch in enumerate(train_dl):
            # batch is a dict with keys for 'input_ids', 'attention_mask', 'image_hidden_states' (see custom_image_word_dataset)
            input_ids, attention_mask, image_hidden_states = batch.values()

            batch_size = input_ids.size(0)

            input_ids = input_ids.to(device, non_blocking=True)  # shape (batch_size x seq_len)
            attention_mask = attention_mask.to(device, non_blocking=True)  # shape (batch_size x seq_len)
            image_hidden_states = image_hidden_states.to(device, non_blocking=True)  # shape (batch_size x image_hidden_dim) (with image_hidden_dim = 1024)

            # model only returns loss, if return_loss=True
            loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_hidden_states=image_hidden_states,
                return_loss=True,
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item() * batch_size
            steps_taken += 1
            overall_steps_taken += 1

            # evaluate every k steps and also at the end of an epoch
            if (steps_taken + 1) >= EVALUATE_EVERY_K_STEPS or (num_batch + 1) == len(train_dl):
                # normalize the train loss by steps_taken
                train_loss /= steps_taken
                val_loss = get_val_loss(model, val_dl)

                writer.add_scalars("loss", {"train_loss": train_loss, "val_loss": val_loss}, overall_steps_taken)

                metrics_with_scores = evaluate_model_on_metrics(model, val_dl, tokenizer, generated_sentences_folder_path, overall_steps_taken)

                for metric_name, score in metrics_with_scores.items():
                    writer.add_scalar(metric_name, score, overall_steps_taken)

                # set the model back to training
                model.train()

                # decrease lr by 1e-1 if val loss has not decreased after certain number of evaluations
                lr_scheduler.step(val_loss)

                if val_loss < lowest_val_loss:
                    num_evaluations_without_decrease_val_loss = 0
                    lowest_val_loss = val_loss
                    best_epoch = epoch
                    best_model_save_path = os.path.join(
                        weights_folder_path, f"val_loss_{lowest_val_loss:.3f}_epoch_{epoch}.pth"
                    )
                    best_model_state = deepcopy(model.state_dict())
                else:
                    num_evaluations_without_decrease_val_loss += 1

                if num_evaluations_without_decrease_val_loss >= patience:
                    # save the model with the overall lowest val loss
                    torch.save(best_model_state, best_model_save_path)
                    log.info(f"\nEarly stopping at epoch ({epoch}/{epochs})!")
                    log.info(f"Lowest overall val loss: {lowest_val_loss:.3f} at epoch {best_epoch}")
                    return None

                # log to console at the end of an epoch
                if (num_batch + 1) == len(train_dl):
                    log_stats_to_console(train_loss, val_loss, epoch)

                # reset values
                train_loss = 0.0
                steps_taken = 0

        num_epochs_without_saving_best_model += 1

        if num_epochs_without_saving_best_model >= save_model_every_k_epochs:
            torch.save(best_model_state, best_model_save_path)
            num_epochs_without_saving_best_model = 0

    # save the model with the overall lowest val loss
    torch.save(best_model_state, best_model_save_path)
    log.info("\nFinished training!")
    log.info(f"Lowest overall val loss: {lowest_val_loss:.3f} at epoch {best_epoch}")
    return None


def get_data_loaders(tokenizer, train_dataset_complete, val_dataset_complete):
    def seed_worker(worker_id):
        """To preserve reproducibility for the randomly shuffled train loader."""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    custom_collate_train = CustomCollatorWithPadding(tokenizer=tokenizer, padding="longest", is_val=False)
    custom_collate_val = CustomCollatorWithPadding(tokenizer=tokenizer, padding="longest", is_val=True)

    g = torch.Generator()
    g.manual_seed(seed_val)

    train_loader = DataLoader(
        train_dataset_complete,
        collate_fn=custom_collate_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset_complete,
        collate_fn=custom_collate_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_tokenized_datasets(tokenizer, raw_train_dataset, raw_val_dataset):
    def tokenize_function(example):
        phrase = example["phrases"]
        bos_token = "<|endoftext|>"  # note: in the GPT2 tokenizer, bos_token = eos_token = "<|endoftext|>"
        eos_token = "<|endoftext|>"
        phrase_with_special_tokens = bos_token + phrase + eos_token
        return tokenizer(phrase_with_special_tokens, truncation=True, max_length=1024)

    # don't set batched=True, otherwise phrases that are empty will not be processed correctly
    # tokenized datasets will consist of the columns "phrases", "input_ids", "attention_mask"
    tokenized_train_dataset = raw_train_dataset.map(tokenize_function)
    tokenized_val_dataset = raw_val_dataset.map(tokenize_function)

    # remove redundant "phrases" column for the train set
    # keep the "phrases" column for the val set, since we need the gt phrases to compute BLEU/BERT scores
    tokenized_train_dataset = tokenized_train_dataset.remove_columns(["phrases"])

    return tokenized_train_dataset, tokenized_val_dataset


def get_tokenizer():
    checkpoint = "healx/gpt-2-pubmed-medium"
    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_datasets_with_phrases(config_file_path):
    # path to the csv files specifying the train, val, test sets
    path_chest_imagenome_customized = "/u/home/tanida/datasets/chest-imagenome-dataset-customized-full"
    datasets_as_dfs = {
        dataset: os.path.join(path_chest_imagenome_customized, dataset) + ".csv" for dataset in ["train", "valid"]
    }

    # only read in the phrases
    usecols = ["phrases"]
    datasets_as_dfs = {
        dataset: pd.read_csv(csv_file_path, usecols=usecols, keep_default_na=False)
        for dataset, csv_file_path in datasets_as_dfs.items()
    }

    total_num_samples_train = len(datasets_as_dfs["train"])
    total_num_samples_val = len(datasets_as_dfs["valid"])

    # compute new number of samples for both train and val
    new_num_samples_train = int(PERCENTAGE_OF_TRAIN_SET_TO_USE * total_num_samples_train)
    new_num_samples_val = int(PERCENTAGE_OF_VAL_SET_TO_USE * total_num_samples_val)

    log.info(f"Train: {new_num_samples_train} phrases")
    log.info(f"Val: {new_num_samples_val} phrases")

    with open(config_file_path, "a") as f:
        f.write(f"\tTRAIN NUM PHRASES: {new_num_samples_train}\n")
        f.write(f"\tVAL NUM PHRASES: {new_num_samples_val}\n")

    # limit the datasets to those new numbers
    datasets_as_dfs["train"] = datasets_as_dfs["train"][:new_num_samples_train]
    datasets_as_dfs["valid"] = datasets_as_dfs["valid"][:new_num_samples_val]

    raw_train_dataset = Dataset.from_pandas(datasets_as_dfs["train"])
    raw_val_dataset = Dataset.from_pandas(datasets_as_dfs["valid"])

    return raw_train_dataset, raw_val_dataset


def create_run_folder():
    """
    Run folder will contain a folder for saving the trained weights, a folder for the tensorboard files
    as well as a config file that specifies the overall parameters used for training.
    """
    run_folder_path_parent_dir = "/u/home/tanida/runs/decoder_model"

    run_folder_path = os.path.join(run_folder_path_parent_dir, f"run_{RUN}")
    weights_folder_path = os.path.join(run_folder_path, "weights")
    tensorboard_folder_path = os.path.join(run_folder_path, "tensorboard")
    generated_sentences_folder_path = os.path.join(run_folder_path, "generated_sentences")

    if os.path.exists(run_folder_path):
        log.error(f"Folder to save run {RUN} already exists at {run_folder_path}.")
        log.error("Delete the folder or change the run number.")
        return None

    os.mkdir(run_folder_path)
    os.mkdir(weights_folder_path)
    os.mkdir(tensorboard_folder_path)
    os.mkdir(generated_sentences_folder_path)

    log.info(f"Run {RUN} folder created at {run_folder_path}.")

    config_file_path = os.path.join(run_folder_path, "run_config.txt")
    config_parameters = {
        "PERCENTAGE_OF_TRAIN_SET_TO_USE": PERCENTAGE_OF_TRAIN_SET_TO_USE,
        "PERCENTAGE_OF_VAL_SET_TO_USE": PERCENTAGE_OF_VAL_SET_TO_USE,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_WORKERS": NUM_WORKERS,
        "EPOCHS": EPOCHS,
        "LR": LR,
        "EVALUATE_EVERY_K_STEPS": EVALUATE_EVERY_K_STEPS,
        "PATIENCE": PATIENCE,
        "PATIENCE_LR_SCHEDULER": PATIENCE_LR_SCHEDULER,
        "NUM_BEAMS": NUM_BEAMS,
        "MAX_NUM_TOKENS_GENERATE": MAX_NUM_TOKENS_GENERATE,
        "NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE": NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE
    }

    with open(config_file_path, "w") as f:
        f.write(f"RUN {RUN}:\n")
        for param_name, param_value in config_parameters.items():
            f.write(f"\t{param_name}: {param_value}\n")

    return weights_folder_path, tensorboard_folder_path, config_file_path, generated_sentences_folder_path


def main():
    weights_folder_path, tensorboard_folder_path, config_file_path, generated_sentences_folder_path = create_run_folder()

    # get the datasets with the raw phrases before tokenization
    raw_train_dataset, raw_val_dataset = get_datasets_with_phrases(config_file_path)

    tokenizer = get_tokenizer()

    # tokenize the raw datasets
    tokenized_train_dataset, tokenized_val_dataset = get_tokenized_datasets(
        tokenizer, raw_train_dataset, raw_val_dataset
    )

    # complete dataset will return a dict with keys "input_ids", "attention_mask", "image_hidden_states" when indexed,
    # with "image_hidden_states" mapping to a tensor of size 1024 that are the image features of the given bbox image

    # validation dataset has an additional key called "phrase" that will return the corresponding ground-truth phrase
    # this is required to compute the BLEU/BERT scores in evaluate_model_on_metrics
    train_dataset_complete = CustomImageWordDataset("train", tokenized_train_dataset)
    val_dataset_complete = CustomImageWordDataset("val", tokenized_val_dataset)

    train_loader, val_loader = get_data_loaders(tokenizer, train_dataset_complete, val_dataset_complete)

    model = DecoderModel()
    model.to(device, non_blocking=True)
    model.train()

    opt = AdamW(model.parameters(), lr=LR)
    lr_scheduler = ReduceLROnPlateau(opt, mode="min", patience=PATIENCE_LR_SCHEDULER)
    writer = SummaryWriter(log_dir=tensorboard_folder_path)

    log.info("\nStarting training!\n")

    train_model(
        model=model,
        train_dl=train_loader,
        val_dl=val_loader,
        optimizer=opt,
        lr_scheduler=lr_scheduler,
        epochs=EPOCHS,
        patience=PATIENCE,
        weights_folder_path=weights_folder_path,
        writer=writer,
        tokenizer=tokenizer,
        generated_sentences_folder_path=generated_sentences_folder_path
    )


if __name__ == "__main__":
    main()
