import cv2
from torch.utils.data import Dataset


class CustomImageWordDatasetFullModel(Dataset):
    def __init__(self, tokenized_dataset, transforms):
        super().__init__()
        self.tokenized_dataset = tokenized_dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.tokenized_dataset)

    def _get_cropped_image_tensor(self, index):
        # mimic_image_file_path
        image_path = self.tokenized_dataset[index]["mimic_image_file_path"]

        # cv2.imread by default loads an image with 3 channels
        # since we have grayscale images, we only have 1 channel and thus use cv2.IMREAD_UNCHANGED to read in the 1 channel
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        # get the coordinates of the bbox
        # x1 and y1 are for the top left corner and x2 and y2 are for the bottom right corner
        x1 = self.tokenized_dataset[index]["x1"]
        x2 = self.tokenized_dataset[index]["x2"]
        y1 = self.tokenized_dataset[index]["y1"]
        y2 = self.tokenized_dataset[index]["y2"]

        # crop the image (which is a np array at this point)
        cropped_image = image[y1:y2, x1:x2]  # cropped_image = image[Y:Y+H, X:X+W]

        # apply transformations
        # albumentations transforms return a dict, which is why key "image" has to be selected
        cropped_image_tensor = self.transforms(image=cropped_image)["image"]

        return cropped_image_tensor  # of shape [1, 224, 224]

    def __getitem__(self, index):
        # if something in __get__item fails, then return None
        # collate_fn in dataloader filters out None values
        try:
            cropped_image_tensor = self._get_cropped_image_tensor(index)

            sample = {
                "image": cropped_image_tensor,
                "input_ids": self.tokenized_dataset[index]["input_ids"],
                "attention_mask": self.tokenized_dataset[index]["attention_mask"],
            }

            # val set will have an additional column called "phrases", which are the reference phrases for the image regions
            # these are necessary to compute the BLEU/BERTscore during evaluation
            reference_phrase = self.tokenized_dataset[index].get("phrases")
            if isinstance(reference_phrase, str):
                # replace empty reference phrase with hash symbol
                sample["reference_phrase"] = reference_phrase if len(reference_phrase) > 0 else "#"

            # val set may have an additional column called "is_abnormal", which is a boolean variable that indicates if
            # a region is described as abnormal or not by a reference phrase (e.g. "There is pneumothorax.") or not (e.g. "There is no pneumothorax.")
            # this variable helps to compute BLEU/BERTscore for reference phrases that have abnormal findings and those that don't
            # -> allows to draw comparisons between those 2 cases
            is_abnormal = self.tokenized_dataset[index].get("is_abnormal")
            if isinstance(is_abnormal, bool):
                sample["is_abnormal"] = is_abnormal

        except Exception:
            return None

        return sample
