import cv2
import torch
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, dataset_df, transforms):
        super().__init__()
        self.dataset_df = dataset_df
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, index):
        # if something in __get__item fails, then return None
        # collate_fn in dataloader filters out None values
        try:
            # mimic_image_file_path is the 1st column of the dataframes
            image_path = self.dataset_df.iloc[index, 0]

            # cv2.imread by default loads an image with 3 channels
            # since we have grayscale images, we only have 1 channel and thus use cv2.IMREAD_UNCHANGED to read in the 1 channel
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

            # resize image to 224x224 (since bbox-coordinates are also set for 224x224 images)
            resized_image = self._resize_pad_image(image, width=224)

            # apply transformations
            # albumentations transforms return a dict, which is why key "image" has to be selected
            resized_image_tensor = self.transforms(image=resized_image)["image"]

            # bbox_coordinates (List[List[int]]) is the 2nd column of the dataframes
            bbox_coordinates = self.dataset_df.iloc[index, 1]

            # labels (List[int]) is the 3rd column of the dataframes
            labels = self.dataset_df.iloc[index, 2]

            sample = {
                "image": resized_image_tensor,
                "boxes": torch.tensor(bbox_coordinates, dtype=torch.float),
                "labels": torch.tensor(labels, dtype=torch.int64),
            }
        except Exception:
            return None

        return sample

    def _resize_pad_image(self, image, width):
        """
        Resize and pad all images to 224x224, such that the bbox coordinates match (which are specified for 224x224 images).
        The implementation is the same as the one in the chest-imagenome folder, i.e. function resize_pad under utils/visualization.ipynb/
        """
        old_size = image.shape[:2]  # old_size is in (height, width) format

        ratio = float(width) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        # new_size should be in (width, height) format

        im = cv2.resize(image, (new_size[1], new_size[0]), interpolation=cv2.INTER_NEAREST)

        delta_w = width - new_size[1]
        delta_h = width - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        color = [0, 0, 0]
        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        return new_im
