import os

import albumentations as A
from albumentations.pytorch import ToTensorV2
from copy import deepcopy
import cv2
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, precision_score, recall_score

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm

from classification_model import ClassificationModel
from custom_image_dataset import CustomImageDataset
from src.dataset.constants import ANATOMICAL_REGIONS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"We use: {device}")

path_chest_imagenome_customized = "/u/home/tanida/datasets/chest-imagenome-dataset-customized-full"

# reduce memory usage by only using necessary columns and selecting appropriate datatypes
usecols = ["mimic_image_file_path", "bbox_name", "x1", "y1", "x2", "y2", "is_abnormal"]
dtype = {"x1": "int16", "x2": "int16", "y1": "int16", "y2": "int16", "bbox_name": "category"}

datasets_as_dfs = {dataset: os.path.join(path_chest_imagenome_customized, dataset) + ".csv" for dataset in ["train", "valid", "test"]}
datasets_as_dfs = {dataset: pd.read_csv(csv_file_path, usecols=usecols, dtype=dtype) for dataset, csv_file_path in datasets_as_dfs.items()}

PERCENTAGE_OF_TRAIN_SET_TO_USE = 0.2
total_num_samples_train = len(datasets_as_dfs["train"])

new_num_samples_train = int(PERCENTAGE_OF_TRAIN_SET_TO_USE * total_num_samples_train)
datasets_as_dfs["train"] = datasets_as_dfs["train"][:new_num_samples_train]

PERCENTAGE_OF_VAL_SET_TO_USE = 0.5
total_num_samples_val = len(datasets_as_dfs["valid"])

new_num_samples_val = int(PERCENTAGE_OF_VAL_SET_TO_USE * total_num_samples_val)
datasets_as_dfs["valid"] = datasets_as_dfs["valid"][:new_num_samples_val]

# see compute_mean_std_dataset.py in src/dataset
mean = 0.471
std = 0.302

# pre-trained DenseNet121 model expects images to be of size 224x224
IMAGE_INPUT_SIZE = 224

# note: transforms are applied to the already cropped images (see __getitem__ method of CustomImageDataset class)!

# use albumentations for Compose and transforms
train_transforms = A.Compose([
    # we want the long edge of the image to be resized to IMAGE_INPUT_SIZE, and the short edge of the image to be padded to IMAGE_INPUT_SIZE on both sides,
    # such that the aspect ratio of the images are kept (i.e. a resized image of a lung is not distorted),
    # while getting images of uniform size (IMAGE_INPUT_SIZE x IMAGE_INPUT_SIZE)
    A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),  # resizes the longer edge to IMAGE_INPUT_SIZE while maintaining the aspect ratio (INTER_AREA works best for shrinking images)
    A.RandomBrightnessContrast(),  # randomly (by default prob=0.5) change brightness and contrast (by a default factor of 0.2)
    # randomly (by default prob=0.5) translate and rotate image
    # mode and cval specify that black pixels are used to fill in newly created pixels
    # translate between -2% and 2% of the image height/width, rotate between -2 and 2 degrees
    A.Affine(mode=cv2.BORDER_CONSTANT, cval=0, translate_percent=(-0.02, 0.02), rotate=(-2, 2)),
    A.GaussianBlur(),  # randomly (by default prob=0.5) blur the image
    A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),  # pads both sides of the shorter edge with 0's (black pixels)
    A.Normalize(mean=mean, std=std),
    ToTensorV2()
])

# don't apply data augmentations to val and test set
val_test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),
        A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]
)

train_dataset = CustomImageDataset(dataset_df=datasets_as_dfs["train"], transforms=train_transforms)
val_dataset = CustomImageDataset(dataset_df=datasets_as_dfs["valid"], transforms=val_test_transforms)
test_dataset = CustomImageDataset(dataset_df=datasets_as_dfs["test"], transforms=val_test_transforms)


def collate_fn(batch):
    # discard images from batch where __getitem__ from custom_image_dataset failed (i.e. returned None)
    # otherwise, whole training loop will stop (even if only 1 image fails to open)
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


BATCH_SIZE = 64
NUM_WORKERS = 12

train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

model_save_path = "/u/home/tanida/weights/classification_model"


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
        # batch is a dict with keys for 'image', 'bbox_target', 'is_abnormal_target' (see custom_image_dataset)
        batch_images, bbox_targets, is_abnormal_targets = batch.values()

        batch_size = batch_images.size(0)

        batch_images = batch_images.to(device, non_blocking=True)  # shape: (BATCH_SIZE, 1, IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE), with IMAGE_INPUT_SIZE usually 224
        bbox_targets = bbox_targets.to(device, non_blocking=True)  # shape: (BATCH_SIZE), integers between 0 and 35 specifying the class for each bbox image
        is_abnormal_targets = is_abnormal_targets.to(device, non_blocking=True)  # shape: (BATCH_SIZE), floats that are either 0. (normal) or 1. (abnormal) specifying if bbox image is normal/abnormal

        # logits has output shape: (BATCH_SIZE, 37)
        logits = model(batch_images)

        # use the first 36 columns as logits for bbox classes, shape: (BATCH_SIZE, 36)
        bbox_class_logits = logits[:, :36]

        # use the last column (i.e. 37th column) as logits for the is_abnormal binary class, shape: (BATCH_SIZE)
        abnormal_logits = logits[:, -1]

        # compute the (multi-class) cross entropy loss
        cross_entropy_loss = cross_entropy(bbox_class_logits, bbox_targets)

        # compute the binary cross entropy loss, use pos_weight to adding weights to positive samples (i.e. abnormal samples)
        # since we have around 7.6x more normal bbox images than abnormal bbox images (see compute_stats_dataset.py),
        # we set pos_weight=7.6 to put 7.6 more weight on the loss of abnormal images
        pos_weight = torch.tensor([7.6]).to(device, non_blocking=True)
        binary_cross_entropy_loss = binary_cross_entropy_with_logits(abnormal_logits, is_abnormal_targets, pos_weight=pos_weight)

        # total loss is weighted 1:1 between cross_entropy_loss and binary_cross_entropy_loss
        total_loss = cross_entropy_loss + binary_cross_entropy_loss

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += total_loss.item() * batch_size

    train_loss /= len(train_dl)

    writer.add_scalar("training loss", train_loss, epoch)

    return train_loss


def evaluate_one_epoch(model, val_dl, lr_scheduler, epoch):
    """
    Evaluate model on val set.

    Write to tensorboard:
        - val loss
        - val f1_score is_abnormal
        - val precision is_abnormal
        - val recall is_abnormal
        - val f1_score bboxes
        - val f1_score bbox for 36 bbox regions

    Args:
        model (nn.Module): The input model to be evaluated.
        val_dl (torch.utils.data.Dataloder): The val dataloader to evaluate on.
        lr_scheduler (torch.optim.lr_scheduler): The learning rate scheduler to use.
        epoch (int): Current epoch number.

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

    # list collects the f1-scores of is_abnormal variables calculated for each batch
    f1_scores_is_abnormal = []

    # list collects the global f1-scores of bboxes calculated for each batch
    f1_scores_bboxes = []

    # list of list where inner list collects the f1-scores calculated for each bbox class for each batch
    f1_scores_bboxes_class = [[] for _ in range(num_classes)]

    # list collects the precision of is_abnormal variables calculated for each batch
    precision_is_abnormal = []

    # list collects the recall of is_abnormal variables calculated for each batch
    recall_is_abnormal = []

    for batch in tqdm(val_dl):
        batch_images, bbox_targets, is_abnormal_targets = batch.values()

        batch_size = batch_images.size(0)

        batch_images = batch_images.to(device, non_blocking=True)  # shape: (BATCH_SIZE, 1, IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE), with IMAGE_INPUT_SIZE usually 224
        bbox_targets = bbox_targets.to(device, non_blocking=True)  # shape: (BATCH_SIZE), integers between 0 and 35 specifying the class for each bbox image
        is_abnormal_targets = is_abnormal_targets.to(device, non_blocking=True)  # shape: (BATCH_SIZE), floats that are either 0. (normal) or 1. (abnormal) specifying if bbox image is normal/abnormal

        # logits has output shape: (BATCH_SIZE, 37)
        logits = model(batch_images)

        # use the first 36 columns as logits for bbox classes, shape: (BATCH_SIZE, 36)
        bbox_class_logits = logits[:, :36]

        # use the last column (i.e. 37th column) as logits for the is_abnormal binary class, shape: (BATCH_SIZE)
        abnormal_logits = logits[:, -1]

        cross_entropy_loss = cross_entropy(bbox_class_logits, bbox_targets)
        pos_weight = torch.tensor([7.6]).to(device, non_blocking=True)  # we have 7.6x more normal bbox images than abnormal ones
        binary_cross_entropy_loss = binary_cross_entropy_with_logits(abnormal_logits, is_abnormal_targets, pos_weight=pos_weight)

        total_loss = cross_entropy_loss + binary_cross_entropy_loss

        val_loss += total_loss.item() * batch_size

        preds_bbox = torch.argmax(bbox_class_logits, dim=1)
        preds_is_abnormal = abnormal_logits > 0

        # f1-score uses average='binary' by default
        is_abnormal_targets = is_abnormal_targets.cpu()
        preds_is_abnormal = preds_is_abnormal.cpu()
        f1_score_is_abnormal_current_batch = f1_score(is_abnormal_targets, preds_is_abnormal)  # single float value
        f1_scores_is_abnormal.append(f1_score_is_abnormal_current_batch)

        # average='micro': calculate metrics globally by counting the total true positives, false negatives and false positives
        f1_score_bbox_globally_current_batch = f1_score(bbox_targets.cpu(), preds_bbox.cpu(), average="micro")  # single float value
        f1_scores_bboxes.append(f1_score_bbox_globally_current_batch)

        # average=None: f1-score for each class are returned
        f1_scores_per_bbox_class_current_batch = f1_score(
            bbox_targets.cpu(), preds_bbox.cpu(), average=None, labels=[i for i in range(num_classes)]
        )  # list of 36 f1-scores (float values) for 36 regions

        for i in range(num_classes):
            f1_scores_bboxes_class[i].append(f1_scores_per_bbox_class_current_batch[i])

        # precision_score uses average='binary' by default
        precision_is_abnormal_current_batch = precision_score(is_abnormal_targets, preds_is_abnormal)
        precision_is_abnormal.append(precision_is_abnormal_current_batch)

        # recall_score uses average='binary' by default
        recall_is_abnormal_current_batch = recall_score(is_abnormal_targets, preds_is_abnormal)
        recall_is_abnormal.append(recall_is_abnormal_current_batch)

    val_loss /= len(val_dl)

    f1_score_is_abnormal = np.array(f1_scores_is_abnormal).mean()
    f1_score_bboxes = np.array(f1_scores_bboxes).mean()
    f1_scores_per_bbox_class = [np.array(list_).mean() for list_ in f1_scores_bboxes_class]

    precision_is_abnormal = np.array(precision_is_abnormal).mean()
    recall_is_abnormal = np.array(recall_is_abnormal).mean()

    writer.add_scalar("val loss", val_loss, epoch)
    writer.add_scalar("val f1_score is_abnormal", f1_score_is_abnormal, epoch)
    writer.add_scalar("val f1_score bboxes", f1_score_bboxes, epoch)
    writer.add_scalar("val precision is_abnormal", precision_is_abnormal, epoch)
    writer.add_scalar("val recall is_abnormal", recall_is_abnormal, epoch)

    for i, bbox_name in enumerate(ANATOMICAL_REGIONS):
        writer.add_scalar(f"valid f1_score bbox: {bbox_name}", f1_scores_per_bbox_class[i], epoch)

    # decrease lr by 1e-1 if val loss has not decreased after certain number of epochs
    lr_scheduler.step(val_loss)

    return (
        val_loss,
        f1_score_is_abnormal,
        f1_score_bboxes,
        f1_scores_per_bbox_class,
        precision_is_abnormal,
        recall_is_abnormal,
    )


def print_stats_to_console(
    train_loss,
    val_loss,
    f1_score_is_abnormal,
    f1_score_bboxes,
    f1_scores_per_bbox_class,
    precision_is_abnormal,
    recall_is_abnormal,
    epoch,
):
    print(f"Epoch: {epoch}:")
    print(f"\tTrain loss: {train_loss:.3f}")
    print(f"\tVal loss: {val_loss:.3f}")
    print(f"\tVal precision is_abnormal: {precision_is_abnormal:.3f}")
    print(f"\tVal recall is_abnormal: {recall_is_abnormal:.3f}")
    print(f"\tVal f1_score is_abnormal: {f1_score_is_abnormal:.3f}")
    print(f"\tVal f1_score bboxes: {f1_score_bboxes:.3f}")

    # only print f1_score per bbox class every 5 epochs
    if epoch % 5 == 0:
        print()
        for i, bbox_name in enumerate(ANATOMICAL_REGIONS):
            print(f"\tVal f1_score bbox '{bbox_name}': {f1_scores_per_bbox_class[i]:.3f}")


def train_model(model, train_dl, val_dl, optimizer, lr_scheduler, epochs, patience):
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
        val_stats = evaluate_one_epoch(model, val_dl, lr_scheduler, epoch)
        print_stats_to_console(train_loss, *val_stats, epoch)

        val_loss = val_stats[0]

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


EPOCHS = 30
LR = 1e-4
PATIENCE = 7  # number of epochs to wait before early stopping
PATIENCE_LR_SCHEDULER = 2  # number of epochs to wait for val loss to reduce before lr is reduced by 1e-1

model = ClassificationModel()
model.to(device, non_blocking=True)
opt = AdamW(model.parameters(), lr=LR)
lr_scheduler = ReduceLROnPlateau(opt, mode="min", patience=PATIENCE_LR_SCHEDULER)
writer = SummaryWriter(log_dir="/u/home/tanida/weights/classification_model/runs/2")
train_model(
    model=model,
    train_dl=train_loader,
    val_dl=val_loader,
    optimizer=opt,
    lr_scheduler=lr_scheduler,
    epochs=EPOCHS,
    patience=PATIENCE,
)
