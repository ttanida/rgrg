from ast import literal_eval
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.object_detector.custom_image_dataset_object_detector import CustomImageDataset

# define configurations for training run
RUN = 0
PERCENTAGE_OF_TRAIN_SET_TO_USE = 0.08
PERCENTAGE_OF_VAL_SET_TO_USE = 0.1
BATCH_SIZE = 32
NUM_WORKERS = 12
EPOCHS = 30
LR = 1e-2
EVALUATE_EVERY_K_STEPS = 3500  # how often to evaluate the model on the validation set and log metrics to tensorboard (additionally, model will always be evaluated at end of epoch)
PATIENCE = 10  # number of evaluations to wait before early stopping
PATIENCE_LR_SCHEDULER = 3  # number of evaluations to wait for val loss to reduce before lr is reduced by 1e-1

def get_datasets_as_dfs():
    path_dataset_object_detector = "/u/home/tanida/datasets/dataset-for-full-model-50"

    usecols = ["mimic_image_file_path", "bbox_coordinates", "bbox_labels"]

    # since bbox_coordinates and labels are stored as strings in the csv_file, we have to apply 
    # the literal_eval func to convert them to python lists
    converters = {"bbox_coordinates": literal_eval, "bbox_labels": literal_eval}

    datasets_as_dfs = {dataset: os.path.join(path_dataset_object_detector, f"{dataset}-50") + ".csv" for dataset in ["train", "valid", "test"]}
    datasets_as_dfs = {dataset: pd.read_csv(csv_file_path, usecols=usecols, converters=converters) for dataset, csv_file_path in datasets_as_dfs.items()}

    total_num_samples_train = len(datasets_as_dfs["train"])
    total_num_samples_val = len(datasets_as_dfs["valid"])

    # compute new number of samples for both train and val
    new_num_samples_train = int(PERCENTAGE_OF_TRAIN_SET_TO_USE * total_num_samples_train)
    new_num_samples_val = int(PERCENTAGE_OF_VAL_SET_TO_USE * total_num_samples_val)

    # limit the datasets to those new numbers
    datasets_as_dfs["train"] = datasets_as_dfs["train"][:new_num_samples_train]
    datasets_as_dfs["valid"] = datasets_as_dfs["valid"][:new_num_samples_val]

    return datasets_as_dfs

datasets_as_dfs = get_datasets_as_dfs()

def get_transforms(dataset: str):
    # see compute_mean_std_dataset.py in src/dataset_bounding_boxes
    mean = 0.471
    std = 0.302

    # note: transforms are applied to the already resized (to 224x224) and padded images 
    # (see __getitem__ method of custom dataset class)!

    # use albumentations for Compose and transforms
    train_transforms = A.Compose([
        # A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
        # A.GaussianBlur(blur_limit=(1, 1)),
        # A.ColorJitter(),
        # A.Sharpen(alpha=(0.1, 0.2), lightness=0.0),
        # A.Affine(mode=cv2.BORDER_CONSTANT, cval=0, translate_percent=(-0.02, 0.02), rotate=(-20, 20), p=1.0),
        A.Affine(mode=cv2.BORDER_CONSTANT, cval=0, translate_percent=(-0.30, 0), p=1.0),
        # A.GaussNoise(),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=['class_labels']))

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


train_transforms = get_transforms("train")
val_transforms = get_transforms("val")

train_dataset = CustomImageDataset(datasets_as_dfs["train"], train_transforms)
val_dataset = CustomImageDataset(datasets_as_dfs["valid"], val_transforms)

bboxes = train_dataset[0]["boxes"]
