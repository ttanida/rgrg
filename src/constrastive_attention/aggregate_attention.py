import torch
import torch.nn as nn


class AggregateAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        top_region_features: torch.FloatTensor  # shape [batch_size x 36 x 2048]
    ):
        pass
