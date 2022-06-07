import cv2
import torch
from torch.utils.data import Dataset

from src.dataset.constants import ANATOMICAL_REGIONS


class CustomImageDataset(Dataset):
    def __init__(self, dataset_df, transforms):
        super().__init__()
        self.dataset_df = dataset_df
        self.transform = transforms

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, index):
        # mimic_image_file_path is the 1st column of the dataframes
        image_path = self.dataset_df.iloc[index, 0]
        image = cv2.imread(image_path)

        # by default OpenCV uses BGR color space for color images, so we need to convert the image to RGB color space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # get the coordinates of the bbox
        # x1 and y1 are for the top left corner and x2 and y2 are for the bottom right corner
        x1, y1, x2, y2 = self.dataset_df.iloc[index, 2:6]

        # crop the image (which is a np array at this point)
        cropped_image = image[y1:y2, x1:x2]  # cropped_image = image[Y:Y+H, X:X+W]

        # apply resize, pad, data augmentation transformations, normalize, toTensor
        cropped_image = self.transform(image=cropped_image)["image"]

        # get the bbox_name (2nd column of df) and convert it into corresponding class index
        bbox_class_index = ANATOMICAL_REGIONS[self.dataset_df.iloc[index, 1]]

        # get the is_abnormal boolean variable (7th column of df) and convert it into integer
        is_abnormal_int = int(self.dataset_df.iloc[index, 6])

        labels = torch.Tensor([bbox_class_index, is_abnormal_int])

        return cropped_image, labels
