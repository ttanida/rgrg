import torch
import torch.nn as nn

from src.constrastive_attention.aggregate_attention import AggregateAttention


class ConstrastiveAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.aggregate_attention = AggregateAttention()
        self.differentiate_attention = DifferentiateAttention()

    def forward(
        self,
        top_region_features: torch.FloatTensor  # shape [batch_size x 36 x 2048]
    ):
        closest_normal_region_features = self.aggregate_attention(top_region_features)  # shape [batch_size x 36 x 6 x 2048]
        top_region_features_with_contrastive_information = self.differentiate_attention(closest_normal_region_features, top_region_features)

        return top_region_features_with_contrastive_information  # shape [batch_size x 36 x 2048]
