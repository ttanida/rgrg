import cv2
from torch.utils.data import Dataset


class CustomImageWordDataset(Dataset):
    def __init__(self, dataset_name: str, tokenized_dataset, transforms):
        super().__init__()
        self.dataset_name = dataset_name
        self.tokenized_dataset = tokenized_dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.tokenized_dataset)

    def __getitem__(self, index):
        # if something in __get__item fails, then return None
        # collate_fn in dataloader filters out None values
        try:
            image_path = self.tokenized_dataset[index]["mimic_image_file_path"]
            bbox_coordinates = self.tokenized_dataset[index]["bbox_coordinates"]
            bbox_labels = self.tokenized_dataset[index]["bbox_labels"]
            input_ids = self.tokenized_dataset[index]["input_ids"]
            attention_mask = self.tokenized_dataset[index]["attention_mask"]
            bbox_phrase_exists = self.tokenized_dataset[index]["bbox_phrase_exists"]
            bbox_is_abnormal = self.tokenized_dataset[index]["bbox_is_abnormal"]

            # we only need the reference phrases when computing the BLEU/BERTScore during evaluation with val and test set
            if self.dataset_name != "train":
                bbox_phrases = self.tokenized_dataset[index]["bbox_phrases"]

            # cv2.imread by default loads an image with 3 channels
            # since we have grayscale images, we only have 1 channel and thus use cv2.IMREAD_UNCHANGED to read in the 1 channel
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

            # apply transformations to image, bboxes and label
            transformed = self.transforms(image=image, bboxes=bbox_coordinates, class_labels=bbox_labels)

            transformed_image = transformed["image"]
            transformed_bbox_coordinates = transformed["bboxes"]
            transformed_bbox_labels = transformed["class_labels"]









            sample = {
                "input_ids": self.tokenized_dataset[index]["input_ids"],
                "attention_mask": self.tokenized_dataset[index]["attention_mask"],
                "image_hidden_states": self.image_features[index]
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
