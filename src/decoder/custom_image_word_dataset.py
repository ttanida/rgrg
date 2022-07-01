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
        except Exception:
            return None

        return sample
