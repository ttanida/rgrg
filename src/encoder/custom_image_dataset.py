import cv2
import torch
from torch.utils.data import Dataset

from src.dataset.constants import ANATOMICAL_REGIONS


class CustomImageDataset(Dataset):
    def __init__(self, dataset_df, transforms):
        super().__init__()
        self.dataset_df = dataset_df
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, index):
        # mimic_image_file_path is the 1st column of the dataframes
        image_path = self.dataset_df.iloc[index, 0]

        # cv2.imread by default loads an image with 3 channels
        # since we have grayscale images, we only have 1 channel and thus use cv2.IMREAD_UNCHANGED to read in the 1 channel
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        # get the coordinates of the bbox
        # x1 and y1 are for the top left corner and x2 and y2 are for the bottom right corner
        x1, y1, x2, y2 = self.dataset_df.iloc[index, 2:6]

        # crop the image (which is a np array at this point)
        cropped_image = image[y1:y2, x1:x2]  # cropped_image = image[Y:Y+H, X:X+W]

        # apply transformations
        # albumentations transforms return a dict, which is why key "image" has to be selected
        cropped_image = self.transforms(image=cropped_image)["image"]

        # get the bbox_name (2nd column of df) and convert it into corresponding class index
        bbox_class_index = ANATOMICAL_REGIONS[self.dataset_df.iloc[index, 1]]

        # get the is_abnormal boolean variable (7th column of df) and convert it into integer
        is_abnormal_int = int(self.dataset_df.iloc[index, 6])

        labels = torch.Tensor([bbox_class_index, is_abnormal_int])

        return cropped_image, labels
