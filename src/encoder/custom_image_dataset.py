import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

from src.dataset.constants import ANATOMICAL_REGIONS


class CustomImageDataset(Dataset):
    def __init__(self, dataset_df, transforms):
        super().__init__()
        self.dataset_df = dataset_df
        self.transform = transforms

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, index):
        img_path = self.dataset_df.iloc[index, 0]  # mimic_image_file_path is the 1st column of the dataframes
        image = read_image(img_path)

        bbox_coordinates = self.dataset_df.iloc[index, 2:6].tolist()  # bbox_coordinates = [x1, y1, x2, y2]

        # TODO: crop image

        cropped_image = self.transform(cropped_image)


        bbox_class_index = self.dataset_df.iloc[index, 1]  # 2nd column is bbox_name

        labels = self.dataset_df.iloc[index, [1, 6]].tolist()  # 2nd column is bbox_name, 7th column is is_abnormal boolean variable
        return cropped_image, labels
