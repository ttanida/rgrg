from ast import literal_eval
import logging
import os
import random

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from datasets import Dataset
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm

from src.full_model.custom_collator import CustomCollator
from src.full_model.custom_dataset import CustomDataset
from src.full_model.evaluate_full_model.evaluate_model import evaluate_model
from src.full_model.report_generation_model import ReportGenerationModel
from src.full_model.run_configurations import (
    RUN,
    RUN_COMMENT,
    IMAGE_INPUT_SIZE,
    PERCENTAGE_OF_TRAIN_SET_TO_USE,
    PERCENTAGE_OF_VAL_SET_TO_USE,
    BATCH_SIZE,
    NUM_WORKERS,
    EPOCHS,
    LR,
    EVALUATE_EVERY_K_STEPS,
    PATIENCE_LR_SCHEDULER,
    THRESHOLD_LR_SCHEDULER,
    NUM_BEAMS,
    MAX_NUM_TOKENS_GENERATE,
    NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE,
    NUM_SENTENCES_TO_GENERATE_FOR_EVALUATION,
    NUM_IMAGES_TO_PLOT,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)

# set the seed value for reproducibility
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


def train_model(model, train_dl, val_dl, optimizer, lr_scheduler, epochs, weights_folder_path, tokenizer, generated_sentences_folder_path, writer):
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
    weights_folder_path: str
        Path to folder where best weights will be saved.
    tokenizer: transformers.GPT2Tokenizer
        Used for decoding the generated ids into tokens in evaluate_model (more specifically evaluate_language_model)
    generated_sentences_folder_path:
        Path to folder where generated sentences will be saved as a txt file.
    writer: torch.utils.tensorboard.SummaryWriter
        Writer for logging values to tensorboard.

    Returns
    -------
    None, but saves model with the overall lowest val loss at the end of every epoch.
    """
    run_params = {}
    run_params["epochs"] = epochs
    run_params["weights_folder_path"] = weights_folder_path
    run_params["lowest_val_loss"] = np.inf
    run_params["best_epoch"] = None  # the epoch with the lowest val loss overall
    run_params["best_model_state"] = None  # the best_model_state is the one where the val loss is the lowest overall
    run_params["best_model_save_path"] = None
    run_params["overall_steps_taken"] = 0  # for logging to tensorboard

    for epoch in range(epochs):
        run_params["epoch"] = epoch
        log.info(f"\nTraining epoch {epoch}!\n")

        train_losses_dict = {
            "total_loss": 0.0,
            "obj_detector_loss": 0.0,
            "region_selection_loss": 0.0,
            "region_abnormal_loss": 0.0,
            "language_model_loss": 0.0,
        }

        run_params["steps_taken"] = 0  # to know when to evaluate model during epoch and to normalize losses

        for num_batch, batch in tqdm(enumerate(train_dl)):
            images = batch["images"]
            image_targets = batch["image_targets"]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            region_has_sentence = batch["region_has_sentence"]
            region_is_abnormal = batch["region_is_abnormal"]

            batch_size = images.size(0)

            # put all tensors on the GPU
            images = images.to(device, non_blocking=True)
            image_targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in image_targets]
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            region_has_sentence = region_has_sentence.to(device, non_blocking=True)
            region_is_abnormal = region_is_abnormal.to(device, non_blocking=True)

            output = model(images, image_targets, input_ids, attention_mask, region_has_sentence, region_is_abnormal)

            # if something went wrong in the forward pass (see forward method for details)
            if output == -1:
                optimizer.zero_grad()
                continue
            else:
                (
                    obj_detector_loss_dict,
                    classifier_loss_region_selection,
                    classifier_loss_region_abnormal,
                    language_model_loss,
                ) = output

            # sum up all 4 losses from the object detector
            obj_detector_losses = sum(loss for loss in obj_detector_loss_dict.values())

            # sum up the rest of the losses
            total_loss = obj_detector_losses + classifier_loss_region_selection + classifier_loss_region_abnormal + language_model_loss

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            list_of_losses = [
                total_loss,
                obj_detector_losses,
                classifier_loss_region_selection,
                classifier_loss_region_abnormal,
                language_model_loss,
            ]

            # dicts are insertion ordered since Python 3.7
            for loss_type, loss in zip(train_losses_dict, list_of_losses):
                train_losses_dict[loss_type] += loss.item() * batch_size

            run_params["steps_taken"] += 1
            run_params["overall_steps_taken"] += 1

            is_epoch_end = True if (num_batch + 1) == len(train_dl) else False

            # evaluate every k steps and also at the end of an epoch
            if run_params["steps_taken"] >= EVALUATE_EVERY_K_STEPS or is_epoch_end:

                log.info(f"\nEvaluating at step {run_params['overall_steps_taken']}!\n")

                # evaluate the model and write the scores (among other things) to tensorboard
                evaluate_model(model, train_losses_dict, val_dl, lr_scheduler, optimizer, writer, tokenizer, run_params, is_epoch_end, generated_sentences_folder_path, log)

                log.info(f"\nMetrics evaluated at step {run_params['overall_steps_taken']}!\n")

                # reset values for the next evaluation
                for loss_type in train_losses_dict:
                    train_losses_dict[loss_type] = 0.0
                run_params["steps_taken"] = 0

                # set the model back to training
                model.train()

        # save the current best model weights at the end of each epoch
        torch.save(run_params["best_model_state"], run_params["best_model_save_path"])

    log.info("\nFinished training!")
    log.info(f"Lowest overall val loss: {run_params['lowest_val_loss']:.3f} at epoch {run_params['best_epoch']}")
    return None


def get_data_loaders(tokenizer, train_dataset, val_dataset):
    def seed_worker(worker_id):
        """To preserve reproducibility for the randomly shuffled train loader."""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    custom_collate_train = CustomCollator(tokenizer=tokenizer, is_val=False)
    custom_collate_val = CustomCollator(tokenizer=tokenizer, is_val=True)

    g = torch.Generator()
    g.manual_seed(seed_val)

    train_loader = DataLoader(
        train_dataset,
        collate_fn=custom_collate_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        collate_fn=custom_collate_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_transforms(dataset: str):
    # see compute_mean_std_dataset.py in src/dataset
    mean = 0.471
    std = 0.302

    # use albumentations for Compose and transforms
    # augmentations are applied with prob=0.5
    # since Affine translates and rotates the image, we also have to do the same with the bounding boxes, hence the bbox_params arugment
    train_transforms = A.Compose(
        [
            # we want the long edge of the image to be resized to IMAGE_INPUT_SIZE, and the short edge of the image to be padded to IMAGE_INPUT_SIZE on both sides,
            # such that the aspect ratio of the images are kept, while getting images of uniform size (IMAGE_INPUT_SIZE x IMAGE_INPUT_SIZE)
            # LongestMaxSize: resizes the longer edge to IMAGE_INPUT_SIZE while maintaining the aspect ratio
            # INTER_AREA works best for shrinking images
            A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
            A.GaussianBlur(blur_limit=(1, 1)),
            A.ColorJitter(hue=0.0, saturation=0.0),
            A.Sharpen(alpha=(0.1, 0.2), lightness=0.0),
            # randomly (by default prob=0.5) translate and rotate image
            # mode and cval specify that black pixels are used to fill in newly created pixels
            # translate between -2% and 2% of the image height/width, rotate between -2 and 2 degrees
            A.Affine(mode=cv2.BORDER_CONSTANT, cval=0, translate_percent=(-0.02, 0.02), rotate=(-2, 2)),
            A.GaussNoise(),
            # PadIfNeeded: pads both sides of the shorter edge with 0's (black pixels)
            A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    )

    # don't apply data augmentations to val and test set
    val_test_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    )

    if dataset == "train":
        return train_transforms
    else:
        return val_test_transforms


def get_tokenized_datasets(tokenizer, raw_train_dataset, raw_val_dataset):
    def tokenize_function(example):
        phrases = example["bbox_phrases"]  # List[str]
        bos_token = "<|endoftext|>"  # note: in the GPT2 tokenizer, bos_token = eos_token = "<|endoftext|>"
        eos_token = "<|endoftext|>"

        phrases_with_special_tokens = [bos_token + phrase + eos_token for phrase in phrases]

        # the tokenizer will return input_ids of type List[List[int]] and attention_mask of type List[List[int]]
        return tokenizer(phrases_with_special_tokens, truncation=True, max_length=1024)

    tokenized_train_dataset = raw_train_dataset.map(tokenize_function)
    tokenized_val_dataset = raw_val_dataset.map(tokenize_function)

    # tokenized datasets will consist of the columns
    #   - mimic_image_file_path
    #   - bbox_coordinates (List[List[int]])
    #   - bbox_labels (List[int])
    #   - bbox_phrases (List[str])
    #   - input_ids (List[List[int]])
    #   - attention_mask (List[List[int]])
    #   - bbox_phrase_exists (List[bool])
    #   - bbox_is_abnormal (List[bool])

    return tokenized_train_dataset, tokenized_val_dataset


def get_tokenizer():
    checkpoint = "healx/gpt-2-pubmed-medium"
    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_datasets(config_file_path):
    path_dataset_object_detector = "/u/home/tanida/datasets/dataset-for-full-model-original-bbox-coordinates"

    usecols = [
        "mimic_image_file_path",
        "bbox_coordinates",
        "bbox_labels",
        "bbox_phrases",
        "bbox_phrase_exists",
        "bbox_is_abnormal",
    ]

    # all of the columns below are stored as strings in the csv_file
    # however, as they are actually lists, we apply the literal_eval func to convert them to lists
    converters = {
        "bbox_coordinates": literal_eval,
        "bbox_labels": literal_eval,
        "bbox_phrases": literal_eval,
        "bbox_phrase_exists": literal_eval,
        "bbox_is_abnormal": literal_eval,
    }

    datasets_as_dfs = {dataset: os.path.join(path_dataset_object_detector, dataset) + ".csv" for dataset in ["train", "valid"]}

    datasets_as_dfs = {dataset: pd.read_csv(csv_file_path, usecols=usecols, converters=converters) for dataset, csv_file_path in datasets_as_dfs.items()}

    # bbox_phrases is a list of str
    # replace each bbox_phrase that is empty (i.e. "") by "#"
    # this is done such that model learns to generate the "#" symbol instead of "" for empty sentences
    # this is done because generated sentences that are "" (i.e. have len = 0) will cause problems when computing e.g. Bleu scores
    for dataset_df in datasets_as_dfs.values():
        dataset_df["bbox_phrases"] = dataset_df["bbox_phrases"].apply(lambda bbox_phrases: [phrase if len(phrase) != 0 else "#" for phrase in bbox_phrases])

    total_num_samples_train = len(datasets_as_dfs["train"])
    total_num_samples_val = len(datasets_as_dfs["valid"])

    # compute new number of samples for both train and val
    new_num_samples_train = int(PERCENTAGE_OF_TRAIN_SET_TO_USE * total_num_samples_train)
    new_num_samples_val = int(PERCENTAGE_OF_VAL_SET_TO_USE * total_num_samples_val)

    log.info(f"Train: {new_num_samples_train} images")
    log.info(f"Val: {new_num_samples_val} images")

    with open(config_file_path, "a") as f:
        f.write(f"\tTRAIN NUM IMAGES: {new_num_samples_train}\n")
        f.write(f"\tVAL NUM IMAGES: {new_num_samples_val}\n")

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
    run_folder_path_parent_dir = "/u/home/tanida/runs/full_model"

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
        "COMMENT": RUN_COMMENT,
        "IMAGE_INPUT_SIZE": IMAGE_INPUT_SIZE,
        "PERCENTAGE_OF_TRAIN_SET_TO_USE": PERCENTAGE_OF_TRAIN_SET_TO_USE,
        "PERCENTAGE_OF_VAL_SET_TO_USE": PERCENTAGE_OF_VAL_SET_TO_USE,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_WORKERS": NUM_WORKERS,
        "EPOCHS": EPOCHS,
        "LR": LR,
        "EVALUATE_EVERY_K_STEPS": EVALUATE_EVERY_K_STEPS,
        "PATIENCE_LR_SCHEDULER": PATIENCE_LR_SCHEDULER,
        "THRESHOLD_LR_SCHEDULER": THRESHOLD_LR_SCHEDULER,
        "NUM_BEAMS": NUM_BEAMS,
        "MAX_NUM_TOKENS_GENERATE": MAX_NUM_TOKENS_GENERATE,
        "NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE": NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE,
        "NUM_SENTENCES_TO_GENERATE_FOR_EVALUATION": NUM_SENTENCES_TO_GENERATE_FOR_EVALUATION,
        "NUM_IMAGES_TO_PLOT": NUM_IMAGES_TO_PLOT
    }

    with open(config_file_path, "w") as f:
        f.write(f"RUN {RUN}:\n")
        for param_name, param_value in config_parameters.items():
            f.write(f"\t{param_name}: {param_value}\n")

    return weights_folder_path, tensorboard_folder_path, config_file_path, generated_sentences_folder_path


def main():
    (
        weights_folder_path,
        tensorboard_folder_path,
        config_file_path,
        generated_sentences_folder_path,
    ) = create_run_folder()

    # the datasets still contain the untokenized phrases
    raw_train_dataset, raw_val_dataset = get_datasets(config_file_path)

    tokenizer = get_tokenizer()

    # tokenize the raw datasets
    tokenized_train_dataset, tokenized_val_dataset = get_tokenized_datasets(tokenizer, raw_train_dataset, raw_val_dataset)

    train_transforms = get_transforms("train")
    val_transforms = get_transforms("val")

    train_dataset_complete = CustomDataset("train", tokenized_train_dataset, train_transforms)
    val_dataset_complete = CustomDataset("val", tokenized_val_dataset, val_transforms)

    train_loader, val_loader = get_data_loaders(tokenizer, train_dataset_complete, val_dataset_complete)

    model = ReportGenerationModel()
    model.to(device, non_blocking=True)
    model.train()

    opt = AdamW(model.parameters(), lr=LR)
    lr_scheduler = ReduceLROnPlateau(opt, mode="min", patience=PATIENCE_LR_SCHEDULER, threshold=THRESHOLD_LR_SCHEDULER)
    writer = SummaryWriter(log_dir=tensorboard_folder_path)

    log.info("\nStarting training!\n")

    train_model(
        model=model,
        train_dl=train_loader,
        val_dl=val_loader,
        optimizer=opt,
        lr_scheduler=lr_scheduler,
        epochs=EPOCHS,
        weights_folder_path=weights_folder_path,
        tokenizer=tokenizer,
        generated_sentences_folder_path=generated_sentences_folder_path,
        writer=writer,
    )


if __name__ == "__main__":
    main()
