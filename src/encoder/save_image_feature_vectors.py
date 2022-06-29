import os

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from classification_model import ClassificationModel
from custom_image_dataset import CustomImageDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_to_best_weights = "/u/home/tanida/weights/classification_model/weight_runs_2/val_loss_53.536_epoch_11.pth"
path_to_parent_folder = "/u/home/tanida/image_feature_vectors"


def save_image_features(dataset_name: str, dataset: CustomImageDataset, model):
    path_to_dataset_folder = os.path.join(path_to_parent_folder, dataset_name)

    # use batching to speed up!

    with torch.no_grad():
        for i, sample in tqdm(enumerate(dataset)):
            image_tensor = sample["image"].to(device)
            image_tensor = torch.unsqueeze(image_tensor, dim=0)  # add a batch dimension
            image_feature_vector = model(image_tensor).cpu().numpy()
            image_save_path = os.path.join(path_to_dataset_folder, f"{i}_image_feature")
            np.save(file=image_save_path, arr=image_feature_vector)

# ask Philip about file size
# for around 1.000.000 bbox feature vectors, it would be 8GB
#
#


def create_datasets(datasets_as_dfs):
    # see compute_mean_std_dataset.py in src/dataset
    mean = 0.471
    std = 0.302

    # pre-trained DenseNet121 model expects images to be of size 224x224
    IMAGE_INPUT_SIZE = 224

    # note: transforms are applied to the already cropped images (see __getitem__ method of CustomImageDataset class)!

    # don't apply data augmentations to val and test set
    transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    train_dataset = CustomImageDataset(dataset_df=datasets_as_dfs["train"], transforms=transforms)
    val_dataset = CustomImageDataset(dataset_df=datasets_as_dfs["valid"], transforms=transforms)

    return train_dataset, val_dataset


def get_datasets_as_dfs():
    path_chest_imagenome_customized = "/u/home/tanida/datasets/chest-imagenome-dataset-customized-full"
    PERCENTAGE_OF_TRAIN_SET_TO_USE = 0.2
    PERCENTAGE_OF_VAL_SET_TO_USE = 0.5

    # reduce memory usage by only using necessary columns and selecting appropriate datatypes
    usecols = ["mimic_image_file_path", "bbox_name", "x1", "y1", "x2", "y2", "is_abnormal"]
    dtype = {"x1": "int16", "x2": "int16", "y1": "int16", "y2": "int16", "bbox_name": "category"}

    datasets_as_dfs = {dataset: os.path.join(path_chest_imagenome_customized, dataset) + ".csv" for dataset in ["train", "valid", "test"]}
    datasets_as_dfs = {dataset: pd.read_csv(csv_file_path, usecols=usecols, dtype=dtype) for dataset, csv_file_path in datasets_as_dfs.items()}

    total_num_samples_train = len(datasets_as_dfs["train"])
    new_num_samples_train = int(PERCENTAGE_OF_TRAIN_SET_TO_USE * total_num_samples_train)
    datasets_as_dfs["train"] = datasets_as_dfs["train"][:new_num_samples_train]

    total_num_samples_val = len(datasets_as_dfs["valid"])
    new_num_samples_val = int(PERCENTAGE_OF_VAL_SET_TO_USE * total_num_samples_val)
    datasets_as_dfs["valid"] = datasets_as_dfs["valid"][:new_num_samples_val]

    return datasets_as_dfs


def main():
    datasets_as_dfs = get_datasets_as_dfs()
    train_dataset, val_dataset = create_datasets(datasets_as_dfs)

    model = ClassificationModel(return_feature_vectors=True)
    model.load_state_dict(torch.load(path_to_best_weights))
    model.eval()
    model.to(device)

    save_image_features("train", train_dataset, model)
    save_image_features("val", val_dataset, model)


if __name__ == "__main__":
    main()
