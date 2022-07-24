from ast import literal_eval
from copy import deepcopy
import logging
import os
import random
from typing import List, Dict

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.object_detector.custom_image_dataset_object_detector import CustomImageDataset
from src.object_detector.object_detector import ObjectDetector

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
RUN = 0
PERCENTAGE_OF_TRAIN_SET_TO_USE = 1
PERCENTAGE_OF_VAL_SET_TO_USE = 1
BATCH_SIZE = 16
NUM_WORKERS = 12
EPOCHS = 30
LR = 1e-2
EVALUATE_EVERY_K_STEPS = 3500  # how often to evaluate the model on the validation set and log metrics to tensorboard (additionally, model will always be evaluated at end of epoch)
PATIENCE = 10  # number of evaluations to wait before early stopping
PATIENCE_LR_SCHEDULER = 3  # number of evaluations to wait for val loss to reduce before lr is reduced by 1e-1


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
            images, targets = batch.values()

            batch_size = images.size(0)

            images = images.to(device, non_blocking=True)  # shape (batch_size x 1 x 224 x 224)
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

            # in eval mode, the model returns the losses and detections
            loss_dict, _ = model(images, targets)

            # sum up all 4 losses
            loss = sum(loss for loss in loss_dict.values())

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
    writer
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

    overall_steps_taken = 0  # for logging to tensorboard

    for epoch in range(epochs):
        log.info(f"\nTraining epoch {epoch}!\n")

        train_loss = 0.0
        steps_taken = 0
        for num_batch, batch in tqdm(enumerate(train_dl)):
            # batch is a dict with keys for 'images' and 'targets'
            images, targets = batch.values()

            batch_size = images.size(0)

            images = images.to(device, non_blocking=True)  # shape (batch_size x 1 x 224 x 224)
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            # sum up all 4 losses
            loss = sum(loss for loss in loss_dict.values())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item() * batch_size
            steps_taken += 1
            overall_steps_taken += 1

            # evaluate every k steps and also at the end of an epoch
            if steps_taken >= EVALUATE_EVERY_K_STEPS or (num_batch + 1) == len(train_dl):
                log.info(f"\nEvaluating at step {overall_steps_taken}!\n")

                # normalize the train loss by steps_taken
                train_loss /= steps_taken
                val_loss = get_val_loss(model, val_dl)

                writer.add_scalars("loss", {"train_loss": train_loss, "val_loss": val_loss}, overall_steps_taken)

                log.info(f"\nTrain and val loss evaluated at step {overall_steps_taken}!\n")

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

        # save the current best model weights at the end of each epoch
        torch.save(best_model_state, best_model_save_path)

    # save the model with the overall lowest val loss
    torch.save(best_model_state, best_model_save_path)
    log.info("\nFinished training!")
    log.info(f"Lowest overall val loss: {lowest_val_loss:.3f} at epoch {best_epoch}")
    return None


def collate_fn(batch: List[Dict[str, Tensor]]):
    # each dict in batch is for a single image and has the keys "image", "boxes", "labels"

    # discard images from batch where __getitem__ from custom_image_dataset failed (i.e. returned None)
    # otherwise, whole training loop will stop (even if only 1 image fails to open)
    batch = list(filter(lambda x: x is not None, batch))

    image_shape = batch[0]["image"].size()
    # allocate an empty images_batch tensor that will store all images of the batch
    images_batch = torch.empty(size=(len(batch), *image_shape))

    for i, sample in enumerate(batch):
        # remove image tensors from batch and store them in dedicated images_batch tensor
        images_batch[i] = sample.pop("image")

    # since batch now only contains dicts with keys "boxes" and "labels", rename it as targets
    targets = batch

    # create a new batch variable to store images_batch and targets
    batch_new = {}
    batch_new["images"] = images_batch
    batch_new["targets"] = targets

    return batch_new


def get_transforms(dataset: str):
    # see compute_mean_std_dataset.py in src/dataset_bounding_boxes
    mean = 0.471
    std = 0.302

    # note: transforms are applied to the already resized (to 224x224) and padded images 
    # (see __getitem__ method of custom dataset class)!

    # use albumentations for Compose and transforms
    train_transforms = A.Compose([
        # optionally add augmentation transforms here (but bboxes also have to be transformed in this case!)
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    # don't apply data augmentations to val and test set
    val_test_transforms = A.Compose(
        [
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    if dataset == "train":
        return train_transforms
    else:
        return val_test_transforms


def get_datasets_as_dfs(config_file_path):
    path_dataset_object_detector = "/u/home/tanida/datasets/object-detector-dataset"

    usecols = ["mimic_image_file_path", "bbox_coordinates", "labels"]

    # since bbox_coordinates and labels are stored as strings in the csv_file, we have to apply 
    # the literal_eval func to convert them to python lists
    converters = {"bbox_coordinates": literal_eval, "labels": literal_eval}

    datasets_as_dfs = {dataset: os.path.join(path_dataset_object_detector, f"{dataset}-50") + ".csv" for dataset in ["train", "valid", "test"]}
    datasets_as_dfs = {dataset: pd.read_csv(csv_file_path, usecols=usecols, converters=converters) for dataset, csv_file_path in datasets_as_dfs.items()}

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

    return datasets_as_dfs


def create_run_folder():
    """
    Run folder will contain a folder for saving the trained weights, a folder for the tensorboard files
    as well as a config file that specifies the overall parameters used for training.
    """
    run_folder_path_parent_dir = "/u/home/tanida/runs/object_detector"

    run_folder_path = os.path.join(run_folder_path_parent_dir, f"run_{RUN}")
    weights_folder_path = os.path.join(run_folder_path, "weights")
    tensorboard_folder_path = os.path.join(run_folder_path, "tensorboard")

    if os.path.exists(run_folder_path):
        log.error(f"Folder to save run {RUN} already exists at {run_folder_path}.")
        log.error("Delete the folder or change the run number.")
        return None

    os.mkdir(run_folder_path)
    os.mkdir(weights_folder_path)
    os.mkdir(tensorboard_folder_path)

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
        "PATIENCE_LR_SCHEDULER": PATIENCE_LR_SCHEDULER
    }

    with open(config_file_path, "w") as f:
        f.write(f"RUN {RUN}:\n")
        for param_name, param_value in config_parameters.items():
            f.write(f"\t{param_name}: {param_value}\n")

    return weights_folder_path, tensorboard_folder_path, config_file_path


def main():
    weights_folder_path, tensorboard_folder_path, config_file_path = create_run_folder()

    datasets_as_dfs = get_datasets_as_dfs(config_file_path)

    train_transforms = get_transforms("train")
    val_transforms = get_transforms("val")

    train_dataset = CustomImageDataset(datasets_as_dfs["train"], train_transforms)
    val_dataset = CustomImageDataset(datasets_as_dfs["valid"], val_transforms)

    train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = ObjectDetector()
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
        writer=writer
    )


if __name__ == "__main__":
    main()
