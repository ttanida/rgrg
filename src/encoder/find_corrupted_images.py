import os

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from custom_image_dataset import CustomImageDataset

path_chest_imagenome_customized = "/u/home/tanida/datasets/chest-imagenome-dataset-customized-full"

# reduce memory usage by only using necessary columns and selecting appropriate datatypes
usecols = ["mimic_image_file_path", "bbox_name", "x1", "y1", "x2", "y2", "is_abnormal"]
dtype = {"x1": "int16", "x2": "int16", "y1": "int16", "y2": "int16", "bbox_name": "category"} 

datasets_as_dfs = {dataset: os.path.join(path_chest_imagenome_customized, dataset) + ".csv" for dataset in ["train", "valid", "test"]}
datasets_as_dfs = {dataset: pd.read_csv(csv_file_path, usecols=usecols, dtype=dtype) for dataset, csv_file_path in datasets_as_dfs.items()}

# constants for image transformations

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
    A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),  # pads both sides of the shorter edge with 0's (black pixels)
    # A.HueSaturationValue(),
    # A.Affine(mode=cv2.BORDER_CONSTANT),
    # A.GaussianBlur(),
    A.Normalize(mean=mean, std=std),
    ToTensorV2()
])

# don't apply data augmentations to val and test set
val_test_transforms = A.Compose([
    A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),
    A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),
    A.Normalize(mean=mean, std=std),
    ToTensorV2()
])

train_dataset = CustomImageDataset(dataset_df=datasets_as_dfs["train"], transforms=train_transforms)
val_dataset = CustomImageDataset(dataset_df=datasets_as_dfs["valid"], transforms=val_test_transforms)
test_dataset = CustomImageDataset(dataset_df=datasets_as_dfs["test"], transforms=val_test_transforms)

BATCH_SIZE = 64
NUM_WORKERS = 48

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# print("Val:")
# try:
#     for i, batch in tqdm(enumerate(val_loader)):
#         continue
# except Exception:
#     print(f"Raised exception for batch {i}")

# print("Test:")
# try:
#     for i, batch in tqdm(enumerate(test_loader)):
#         continue
# except Exception:
#     print(f"Raised exception for batch {i}")

# print("Train:")
# try:
#     for i, batch in tqdm(enumerate(train_loader)):
#         continue
# except Exception:
#     print(f"Raised exception for batch {i}")

# print("Train:")
# for i in range(59000, 59500):
#     try:
#         train_dataset[i]
#     except Exception:
#         print(i)

print(train_dataset[59379])