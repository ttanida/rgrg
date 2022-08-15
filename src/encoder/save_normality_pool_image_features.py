import os

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.constants import ANATOMICAL_REGIONS
from src.encoder.classification_model import ClassificationModel
from src.encoder.custom_image_dataset import CustomImageDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
NUM_WORKERS = 12
IMAGE_INPUT_SIZE = 512
IMAGE_HIDDEN_DIM = 2048
NORMALITY_POOL_SIZE = 5

path_normality_pool_csv = f"/u/home/tanida/datasets/normality-pool/normality-pool-{NORMALITY_POOL_SIZE}.csv"
path_to_best_weights = "/u/home/tanida/weights/classification_model/weight_runs_.../..."
path_to_parent_folder = "/u/home/tanida/normality_pool_image_features"


def save_normality_image_features(model, normality_dl_for_every_region):
    for region, region_dl in normality_dl_for_every_region:
        file_saving_path = os.path.join(path_to_parent_folder, f"normality_image_features_{region}")

        size_of_overall_tensor = (NORMALITY_POOL_SIZE, IMAGE_HIDDEN_DIM)
        image_features = torch.empty(size=size_of_overall_tensor, dtype=torch.float32, device=torch.device("cpu"))

        with torch.no_grad():
            for i, batch in tqdm(enumerate(region_dl)):
                batch_images, _ = batch.values()
                batch_images = batch_images.to(device, non_blocking=True)
                batch_image_features = model(batch_images).cpu()  # shape (batch_size, image_hidden_dimension)
                try:
                    image_features[i * BATCH_SIZE: i * BATCH_SIZE + BATCH_SIZE] = batch_image_features
                except RuntimeError:  # since the last batch can be smaller than BATCH_SIZE
                    print(f"Last batch number: {i} (out of {len(region_dl)} batches)")
                    last_batch_size = batch_image_features.size(0)
                    print(f"Number of samples in last batch: {last_batch_size}")
                    image_features[i * BATCH_SIZE: i * BATCH_SIZE + last_batch_size] = batch_image_features

        print(f"Saving image feature tensor of shape {image_features.shape}")
        torch.save(image_features, file_saving_path)


def create_dataloader(normality_dataset_for_every_region):
    dataloaders = {region: DataLoader(region_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True) for region, region_dataset in normality_dataset_for_every_region.items()}

    return dataloaders


def create_dataset(normality_df_for_every_region):
    # see compute_mean_std_dataset.py in src/dataset
    mean = 0.471
    std = 0.302

    # note: transforms are applied to the already cropped images (see __getitem__ method of CustomImageDataset class)!
    transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    dataset = {region: CustomImageDataset(dataset_df=region_df, transforms=transforms) for region, region_df in normality_df_for_every_region.items()}

    return dataset


def get_normality_df_for_every_region():
    # reduce memory usage by only using necessary columns and selecting appropriate datatypes
    usecols = ["mimic_image_file_path", "bbox_name", "x1", "y1", "x2", "y2"]
    dtype = {"bbox_name": "category", "x1": "int16", "x2": "int16", "y1": "int16", "y2": "int16"}

    df = pd.read_csv(path_normality_pool_csv, usecols=usecols, dtype=dtype)
    normality_df_for_every_region = {region: df[df["bbox_name"] == region] for region in ANATOMICAL_REGIONS}

    return normality_df_for_every_region


def main():
    normality_df_for_every_region = get_normality_df_for_every_region()
    normality_dataset_for_every_region = create_dataset(normality_df_for_every_region)
    normality_dl_for_every_region = create_dataloader(normality_dataset_for_every_region)

    model = ClassificationModel(return_feature_vectors=True)
    model.load_state_dict(torch.load(path_to_best_weights))
    model.eval()
    model.to(device, non_blocking=True)

    save_normality_image_features(model, normality_dl_for_every_region)


if __name__ == "__main__":
    main()
