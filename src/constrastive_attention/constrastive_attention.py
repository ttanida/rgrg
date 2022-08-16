import torch
import torch.nn as nn

from src.constrastive_attention.aggregate_attention import AggregateAttention
from src.constrastive_attention.differentiate_attention import DifferentiateAttention


class ConstrastiveAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.aggregate_attention = AggregateAttention()
        self.differentiate_attention = DifferentiateAttention()

    def forward(
        self,
        top_region_features: torch.FloatTensor  # shape [batch_size x 36 x 2048]
    ):
        # closest_normal_region_features of shape [batch_size x 36 x AGGREGATE_ATTENTION_NUM x 2048], with AGGREGATE_ATTENTION_NUM most likely 6
        closest_normal_region_features = self.aggregate_attention(top_region_features)

        # top_region_features_with_contrastive_information of shape  [batch_size x 36 x 2048]
        top_region_features_with_contrastive_information = self.differentiate_attention(closest_normal_region_features, top_region_features)

        return top_region_features_with_contrastive_information
