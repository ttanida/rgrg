import logging
import os
import random

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.encoder.classification_model import ClassificationModel
from src.dataset.constants import ANATOMICAL_REGIONS
from src.encoder.custom_image_dataset import CustomImageDataset

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
RUN = 3
# can be useful to add additional information to run_config.txt file
RUN_COMMENT = """Train ResNet-50 classification model only on classifying regions (i.e. not on classifying normal/abnormal). As dataset use chest-imagenome-dataset-customized-10%-of-full-dataset"""
IMAGE_INPUT_SIZE = 512
PERCENTAGE_OF_TRAIN_SET_TO_USE = 1.0
PERCENTAGE_OF_VAL_SET_TO_USE = 1.0
BATCH_SIZE = 64
NUM_WORKERS = 12
EPOCHS = 20
LR = 1e-3
EVALUATE_EVERY_K_STEPS = 500  # how often to evaluate the model on the validation set and log metrics to tensorboard (additionally, model will always be evaluated at end of epoch)
PATIENCE_LR_SCHEDULER = 5  # number of evaluations to wait for val loss to reduce before lr is reduced by 1e-1
THRESHOLD_LR_SCHEDULER = 1e-3  # threshold for measuring the new optimum, to only focus on significant changes


def write_scores_to_tensorboard(writer, train_loss, val_stats, overall_steps_taken):
    val_loss, precision_score_bboxes, recall_score_bboxes, f1_score_bboxes, f1_scores_per_bbox_class = val_stats

    writer.add_scalars("_loss", {"train_loss": train_loss, "val_loss": val_loss}, overall_steps_taken)

    writer.add_scalar("precision", precision_score_bboxes, overall_steps_taken)
    writer.add_scalar("recall", recall_score_bboxes, overall_steps_taken)
    writer.add_scalar("f1_score", f1_score_bboxes, overall_steps_taken)

    for i, bbox_name in enumerate(ANATOMICAL_REGIONS):
        writer.add_scalar(f"f1_score region: {bbox_name}", f1_scores_per_bbox_class[i], overall_steps_taken)


def evaluate_model(model, val_dl):
    """
    Evaluate model on val set.

    Args:
        model (nn.Module): The input model to be evaluated.
        val_dl (torch.utils.data.Dataloder): The val dataloader to evaluate on.

    Returns:
        val_loss (float): Val loss for 1 epoch.
        f1_scores_is_abnormal (float): Average f1 score for is_abnormal variable for 1 epoch.
        f1_scores_bboxes (float): Average global f1 score for bboxes for 1 epoch.
        f1_scores_bboxes_class (list[float]): Average f1 score for each bbox class for 1 epoch.
        precision_is_abnormal (float): Average precision for is_abnormal variable for 1 epoch.
        recall_is_abnormal (float): Average recall for is_abnormal variable for 1 epoch.
    """
    # evaluating the model on the val set
    model.eval()
    val_loss = 0.0

    num_classes = len(ANATOMICAL_REGIONS)

    # lists collect the global precision/recall/f1-scores of bboxes calculated for each batch
    precision_scores_bboxes = []
    recall_scores_bboxes = []
    f1_scores_bboxes = []

    # list of list where inner list collects the f1-scores calculated for each bbox class for each batch
    f1_scores_bboxes_class = [[] for _ in range(num_classes)]

    with torch.no_grad():
        for batch in tqdm(val_dl):
            batch_images, bbox_targets = batch.values()

            batch_size = batch_images.size(0)

            batch_images = batch_images.to(device, non_blocking=True)  # shape: (BATCH_SIZE, 1, IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE), with IMAGE_INPUT_SIZE usually 512
            bbox_targets = bbox_targets.to(device, non_blocking=True)  # shape: (BATCH_SIZE), integers between 0 and 35 specifying the class for each bbox image

            # logits has output shape: (BATCH_SIZE, 36)
            logits = model(batch_images)

            total_loss = cross_entropy(logits, bbox_targets)

            val_loss += total_loss.item() * batch_size

            bbox_targets = bbox_targets.cpu()
            preds_bbox = torch.argmax(logits, dim=1).cpu()

            # average='micro': calculate metrics globally by counting the total true positives, false negatives and false positives
            precision_score_bbox_globally_current_batch = precision_score(bbox_targets, preds_bbox, average="micro")  # single float value
            recall_score_bbox_globally_current_batch = recall_score(bbox_targets, preds_bbox, average="micro")  # single float value
            f1_score_bbox_globally_current_batch = f1_score(bbox_targets, preds_bbox, average="micro")  # single float value

            precision_scores_bboxes.append(precision_score_bbox_globally_current_batch)
            recall_scores_bboxes.append(recall_score_bbox_globally_current_batch)
            f1_scores_bboxes.append(f1_score_bbox_globally_current_batch)

            # average=None: f1-score for each class are returned
            f1_scores_per_bbox_class_current_batch = f1_score(
                bbox_targets, preds_bbox, average=None, labels=[i for i in range(num_classes)]
            )  # list of 36 f1-scores (float values) for 36 regions

            for i in range(num_classes):
                f1_scores_bboxes_class[i].append(f1_scores_per_bbox_class_current_batch[i])

    val_loss /= len(val_dl)

    precision_score_bboxes = np.array(precision_scores_bboxes).mean()
    recall_score_bboxes = np.array(recall_scores_bboxes).mean()
    f1_score_bboxes = np.array(f1_scores_bboxes).mean()
    f1_scores_per_bbox_class = [np.array(list_).mean() for list_ in f1_scores_bboxes_class]

    return (
        val_loss,
        precision_score_bboxes,
        recall_score_bboxes,
        f1_score_bboxes,
        f1_scores_per_bbox_class,
    )


def train_model(
    model,
    train_dl,
    val_dl,
    optimizer,
    lr_scheduler,
    epochs,
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
    weights_folder_path: str
        Path to folder where best weights will be saved.
    writer: torch.utils.tensorboard.SummaryWriter
        Writer for logging values to tensorboard.

    Returns
    -------
    None, but saves model with the overall lowest val loss at the end of every epoch.
    """
    lowest_val_loss = np.inf

    overall_steps_taken = 0  # for logging to tensorboard

    for epoch in range(epochs):
        log.info(f"\nTraining epoch {epoch}!\n")

        train_loss = 0.0
        steps_taken = 0
        for num_batch, batch in tqdm(enumerate(train_dl)):
            # batch is a dict with keys for 'image', 'bbox_target' (see custom_image_dataset)
            batch_images, bbox_targets = batch.values()

            batch_size = batch_images.size(0)

            batch_images = batch_images.to(device, non_blocking=True)  # shape: (BATCH_SIZE, 1, IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE), with IMAGE_INPUT_SIZE usually 512
            bbox_targets = bbox_targets.to(device, non_blocking=True)  # shape: (BATCH_SIZE), integers between 0 and 35 specifying the class for each bbox image

            # logits has output shape: (BATCH_SIZE, 36)
            logits = model(batch_images)

            # compute the (multi-class) cross entropy loss
            total_loss = cross_entropy(logits, bbox_targets)

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += total_loss.item() * batch_size
            steps_taken += 1
            overall_steps_taken += 1

            # evaluate every k steps and also at the end of an epoch
            if steps_taken >= EVALUATE_EVERY_K_STEPS or (num_batch + 1) == len(train_dl):
                log.info(f"\nEvaluating at step {overall_steps_taken}!\n")

                # normalize the train loss by steps_taken
                train_loss /= steps_taken

                val_stats = evaluate_model(model, val_dl, lr_scheduler, epoch)

                write_scores_to_tensorboard(writer, train_loss, val_stats, overall_steps_taken)

                log.info(f"\nMetrics evaluated at step {overall_steps_taken}!\n")

                # decrease lr by 1e-1 if val loss has not decreased after certain number of evaluations
                val_loss = val_stats[0]
                lr_scheduler.step(val_loss)

                if val_loss < lowest_val_loss:
                    lowest_val_loss = val_loss
                    best_epoch = epoch
                    best_model_save_path = os.path.join(weights_folder_path, f"val_loss_{lowest_val_loss:.3f}_overall_step_{overall_steps_taken}.pth")

                    torch.save(model.state_dict(), best_model_save_path)

                # set the model back to training
                model.train()

                # reset values
                train_loss = 0.0
                steps_taken = 0

    log.info("\nFinished training!")
    log.info(f"Lowest overall val loss: {lowest_val_loss:.3f} at epoch {best_epoch}")
    return None


def collate_fn(batch):
    # discard images from batch where __getitem__ from custom_image_dataset failed (i.e. returned None)
    # otherwise, whole training loop will stop (even if only 1 image fails to open)
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def get_data_loaders(train_dataset, val_dataset):
    def seed_worker(worker_id):
        """To preserve reproducibility for the randomly shuffled train loader."""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed_val)

    train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, worker_init_fn=seed_worker, generator=g, pin_memory=True)
    val_loader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader


def get_transforms(dataset: str):
    # see compute_mean_std_dataset.py in src/dataset
    mean = 0.471
    std = 0.302

    # note: transforms are applied to the already cropped images (see __getitem__ method of CustomImageDataset class)!

    # use albumentations for Compose and transforms
    # augmentations are applied with prob=0.5
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
            ToTensorV2()
        ]
    )

    # don't apply data augmentations to val and test set
    val_test_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    if dataset == "train":
        return train_transforms
    else:
        return val_test_transforms


def get_datasets_as_dfs(config_file_path):
    path_dataset_classification_model = "/u/home/tanida/datasets/chest-imagenome-dataset-customized-10%-of-full-dataset"

    # reduce memory usage by only using necessary columns and selecting appropriate datatypes
    usecols = ["mimic_image_file_path", "bbox_name", "x1", "y1", "x2", "y2"]
    dtype = {"x1": "int16", "x2": "int16", "y1": "int16", "y2": "int16", "bbox_name": "category"}

    datasets_as_dfs = {dataset: os.path.join(path_dataset_classification_model, dataset) + ".csv" for dataset in ["train-600000", "valid-90000"]}
    datasets_as_dfs = {dataset: pd.read_csv(csv_file_path, usecols=usecols, dtype=dtype) for dataset, csv_file_path in datasets_as_dfs.items()}

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

    return datasets_as_dfs


def create_run_folder():
    """
    Run folder will contain a folder for saving the trained weights, a folder for the tensorboard files
    as well as a config file that specifies the overall parameters used for training.
    """
    run_folder_path_parent_dir = "/u/home/tanida/runs/classification_model"

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
        "THRESHOLD_LR_SCHEDULER": THRESHOLD_LR_SCHEDULER
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

    train_dataset = CustomImageDataset(dataset_df=datasets_as_dfs["train"], transforms=train_transforms)
    val_dataset = CustomImageDataset(dataset_df=datasets_as_dfs["valid"], transforms=val_transforms)

    train_loader, val_loader = get_data_loaders(train_dataset, val_dataset)

    model = ClassificationModel()
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
        writer=writer
    )


if __name__ == "__main__":
    main()
