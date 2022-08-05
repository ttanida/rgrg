import os

import torch
from torch.utils.data import Dataset


class CustomImageWordDataset(Dataset):
    def __init__(self, dataset_name: str, tokenized_dataset):
        super().__init__()
        path_to_image_feature_vectors = os.path.join("/u/home/tanida/image_feature_vectors", dataset_name, f"image_features_{dataset_name}")

        self.image_features = torch.load(path_to_image_feature_vectors, map_location=torch.device('cpu'))
        self.tokenized_dataset = tokenized_dataset

    def __len__(self):
        return len(self.tokenized_dataset)

    def __getitem__(self, index):
        # if something in __get__item fails, then return None
        # collate_fn in dataloader filters out None values
        try:
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
