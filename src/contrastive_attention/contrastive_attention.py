import torch
import torch.nn as nn

from src.contrastive_attention.aggregate_attention import AggregateAttention
from src.contrastive_attention.differentiate_attention import DifferentiateAttention
from src.full_model.run_configurations import NORMALITY_POOL_SIZE


class ContrastiveAttention(nn.Module):
    def __init__(self):
        super().__init__()
        NUM_REGIONS = 29
        INPUT_HIDDEN_DIM = 2048
        CA_HIDDEN_DIM = 512

        self.aggregate_attention = AggregateAttention()
        self.differentiate_attention = DifferentiateAttention()

        # normality_pool_image_features of shape [NUM_REGIONS x NORMALITY_POOL_SIZE x INPUT_HIDDEN_DIM]
        # normality pool is registered as a buffer, meaning it's saved and restored in the model's state_dict, but not trained by the optimizer
        # it's important that the normality pool is part of the model's state_dict, since we need it at inference time
        self.register_buffer("normality_pool_image_features", torch.empty(NUM_REGIONS, NORMALITY_POOL_SIZE, INPUT_HIDDEN_DIM))

        # reduce dimensionality as recommended in equation 2 of paper
        self.dim_reduction = nn.Linear(INPUT_HIDDEN_DIM, CA_HIDDEN_DIM)

    def update_normality_pool(self, current_normality_pool):
        # overwrite the values of the last normality pool with the newly calculated values
        self.normality_pool_image_features = current_normality_pool

    def forward(
        self,
        top_region_features: torch.FloatTensor  # shape [batch_size x NUM_REGIONS x INPUT_HIDDEN_DIM]
    ):
        top_region_features = self.dim_reduction(top_region_features)
        normality_pool_image_features = self.dim_reduction(self.normality_pool_image_features)

        # closest_normal_region_features of shape [batch_size x NUM_REGIONS x AGGREGATE_ATTENTION_NUM x CA_HIDDEN_DIM], with AGGREGATE_ATTENTION_NUM most likely 6
        closest_normal_region_features = self.aggregate_attention(top_region_features, normality_pool_image_features)

        # top_region_features_with_contrastive_information of shape [batch_size x NUM_REGIONS x OUTPUT_HIDDEN_DIM], with OUTPUT_HIDDEN_DIM most likely 1024
        top_region_features_with_contrastive_information = self.differentiate_attention(closest_normal_region_features, top_region_features)

        return top_region_features_with_contrastive_information
