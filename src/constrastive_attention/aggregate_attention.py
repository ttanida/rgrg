import os

import torch
import torch.nn as nn

from src.full_model.run_configurations import NORMALITY_POOL_SIZE, AGGREGATE_ATTENTION_NUM


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AggregateAttention(nn.Module):
    def __init__(self):
        super().__init__()
        path_to_normality_pool_image_features = os.path.join("/u/home/tanida/normality_pool_image_features", f"normality_image_features_pool_size_{NORMALITY_POOL_SIZE}")

        # shape [36 x NORMALITY_POOL_SIZE x 2048]
        self.normality_pool_image_features = torch.load(path_to_normality_pool_image_features, map_location=device)

        self.wx = 

    def forward(
        self,
        top_region_features: torch.FloatTensor  # shape [batch_size x 36 x 2048]
    ):
        pass
