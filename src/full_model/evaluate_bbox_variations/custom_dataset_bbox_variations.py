import cv2
import torch
from torchvision.transforms.functional import crop
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

            transformed_image = transformed["image"]  # torch.tensor of shape 1 x 512 x 512 and dtype=float
            transformed_bbox_coordinates = transformed["bboxes"]  # List[List[float]]
            transformed_bbox_coordinates = self._change_bbox_coords_to_int(transformed_bbox_coordinates)  # List[List[int]]

            # List[torch.tensor], i.e. list of len 29 where each tensor is of shape 1 x bbox_coord_height x bbox_coord_width
            bboxes = self._get_cropped_bboxes(transformed_image, transformed_bbox_coordinates)

            sample = {
                "bboxes": bboxes,  # List[torch.tensor]
                "bbox_phrases": bbox_phrases,  # List[str]
                "bbox_phrase_exists": bbox_phrase_exists,  # List[bool]
            }

        except Exception as e:
            self.log.error(f"__getitem__ failed for: {image_path}")
            self.log.error(f"Reason: {e}")
            return None

        return sample

    def _change_bbox_coords_to_int(self, bbox_coords_float: list[list[float]]):
        bbox_coords_int = []
        for bbox_coords in bbox_coords_float:
            x1, y1, x2, y2 = bbox_coords
            bbox_coords_int.append([int(x1), int(y1), int(x2), int(y2)])

        return bbox_coords_int

    def _get_cropped_bboxes(self, image: torch.tensor, bbox_coords: list[list[int]]):
        bboxes = []
        for coords in bbox_coords:
            x1, y1, x2, y2 = coords
            width = x2 - x1
            height = y2 - y1

            bbox_tensor = crop(img=image, top=y1, left=x1, height=height, width=width)
            bboxes.append(bbox_tensor)

        return bboxes
