import os

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.classification_model.classification_model import ClassificationModel
from src.classification_model.custom_image_dataset import CustomImageDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_to_best_weights = "/u/home/tanida/weights/classification_model/weight_runs_2/val_loss_53.536_epoch_11.pth"
path_to_parent_folder = "/u/home/tanida/image_feature_vectors"

BATCH_SIZE = 128
NUM_WORKERS = 12
IMAGE_HIDDEN_DIM = 1024


def save_image_features(dataset_name: str, dataloader, model):
    file_saving_path = os.path.join(path_to_parent_folder, dataset_name, f"image_features_{dataset_name}")

    size_of_overall_tensor = (BATCH_SIZE * len(dataloader), IMAGE_HIDDEN_DIM)
    image_features = torch.empty(size=size_of_overall_tensor, dtype=torch.float32, device=torch.device("cpu"))

    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            batch_images, _, _ = batch.values()
            batch_images = batch_images.to(device, non_blocking=True)
            batch_image_features = model(batch_images).cpu()  # shape (batch_size, image_hidden_dimension)
            try:
                image_features[i * BATCH_SIZE: i * BATCH_SIZE + BATCH_SIZE] = batch_image_features
            except RuntimeError:  # since the last batch can be smaller than BATCH_SIZE
                print(f"Last batch number: {i} (out of {len(dataloader)} batches)")
                last_batch_size = batch_image_features.size(0)
                print(f"Number of samples in last batch: {last_batch_size}")
                image_features[i * BATCH_SIZE: i * BATCH_SIZE + last_batch_size] = batch_image_features

    print(f"Saving image feature tensor of shape {image_features.shape}")
    torch.save(image_features, file_saving_path)


def create_dataloaders(datasets_as_dfs):
    # see compute_mean_std_dataset.py in src/dataset
    mean = 0.471
    std = 0.302

    # pre-trained DenseNet121 model expects images to be of size 224x224
    IMAGE_INPUT_SIZE = 224

    # note: transforms are applied to the already cropped images (see __getitem__ method of CustomImageDataset class)!
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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader


def get_datasets_as_dfs():
    path_chest_imagenome_customized = "/u/home/tanida/datasets/chest-imagenome-dataset-customized-full"
    PERCENTAGE_OF_TRAIN_SET_TO_USE = 0.2
    PERCENTAGE_OF_VAL_SET_TO_USE = 0.5

    # reduce memory usage by only using necessary columns and selecting appropriate datatypes
    usecols = ["mimic_image_file_path", "bbox_name", "x1", "y1", "x2", "y2", "is_abnormal"]
    dtype = {"x1": "int16", "x2": "int16", "y1": "int16", "y2": "int16", "bbox_name": "category"}

    datasets_as_dfs = {dataset: os.path.join(path_chest_imagenome_customized, dataset) + ".csv" for dataset in ["train", "valid"]}
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
    train_dataloader, val_dataloader = create_dataloaders(datasets_as_dfs)

    model = ClassificationModel(return_feature_vectors=True)
    model.load_state_dict(torch.load(path_to_best_weights))
    model.eval()
    model.to(device, non_blocking=True)

    save_image_features("train", train_dataloader, model)
    save_image_features("val", val_dataloader, model)


if __name__ == "__main__":
    main()
