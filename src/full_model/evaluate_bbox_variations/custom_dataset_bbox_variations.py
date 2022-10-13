import cv2
# import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, dataset_as_df, transforms, log):
        super().__init__()
        self.dataset_as_df = dataset_as_df
        self.transforms = transforms
        self.log = log

    def __len__(self):
        return len(self.dataset_as_df)

    def __getitem__(self, index):
        # get the image_path for potential logging in except block
        image_path = self.dataset_as_df.iloc[index]["mimic_image_file_path"]

        # if something in __get__item fails, then return None
        # collate_fn in dataloader filters out None values
        try:
            bbox_coordinates_varied = self.dataset_as_df.iloc[index]["bbox_coordinates_varied"]  # List[List[int]]
            bbox_labels = self.dataset_as_df.iloc[index]["bbox_labels"]  # List[int]
            bbox_phrases = self.dataset_as_df.iloc[index]["bbox_phrases"]  # List[str]
            bbox_phrase_exists = self.dataset_as_df.iloc[index]["bbox_phrase_exists"]  # List[bool]

            # cv2.imread by default loads an image with 3 channels
            # since we have grayscale images, we only have 1 channel and thus use cv2.IMREAD_UNCHANGED to read in the 1 channel
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

            # apply transformations to image, bbox_coordinates and bbox_labels
            transformed = self.transforms(image=image, bboxes=bbox_coordinates_varied, class_labels=bbox_labels)

            transformed_image = transformed["image"]
            transformed_bbox_coordinates = transformed["bboxes"]

            sample = {
                "image": transformed_image,
                "bbox_coordinates": transformed_bbox_coordinates,
                "bbox_phrases": bbox_phrases,
                "bbox_phrase_exists": bbox_phrase_exists,
            }

        except Exception as e:
            self.log.error(f"__getitem__ failed for: {image_path}")
            self.log.error(f"Reason: {e}")
            return None

        return sample
