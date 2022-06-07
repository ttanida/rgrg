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

        bbox_class_index = ANATOMICAL_REGIONS[self.dataset_df.iloc[index, 1]]  # get the bbox_name (2nd column of df) and convert it into corresponding class index
        is_abnormal_int = int(self.dataset_df.iloc[index, 6])  # get the is_abnormal boolean variable (7th column of df) and convert it into integer

        labels = torch.Tensor([bbox_class_index, is_abnormal_int])
        return cropped_image, labels
