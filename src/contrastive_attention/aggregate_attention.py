import math
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

        NUM_REGIONS = 36
        HIDDEN_DIM = 2048

        self.wx = nn.Parameter(torch.empty(AGGREGATE_ATTENTION_NUM, NUM_REGIONS, HIDDEN_DIM, HIDDEN_DIM), requires_grad=True)
        self.wy = nn.Parameter(torch.empty(NUM_REGIONS, HIDDEN_DIM, HIDDEN_DIM), requires_grad=True)

        self.wx_bias = nn.Parameter(torch.empty(AGGREGATE_ATTENTION_NUM, NUM_REGIONS, 1, HIDDEN_DIM), requires_grad=True)
        self.wy_bias = nn.Parameter(torch.empty(NUM_REGIONS, 1, HIDDEN_DIM), requires_grad=True)

        self.reset_parameters(self.wx, self.wx_bias)
        self.reset_parameters(self.wy, self.wy_bias)

    def reset_parameters(self, weight, bias):
        """Use same weight initialization as for Linear layers (https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear)"""
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(bias, -bound, bound)

    def forward(
        self,
        top_region_features: torch.FloatTensor
    ):
        """
        Forward method implements equations 4 - 6 in paper.

        b = batch_size
        r = region_number (i.e. 36)
        a = aggregate_attention_num (i.e. num of closest images by aggregation, most likely 6)
        d = dimensionality (i.e. 2048)
        n = normality_pool_size (most likely 100 or 1000)
        """
        # top_region_features of shape [b x r x d]
        # wx of shape [a x r x d x d]

        # to multiply the two, unsqueeze the necessary dimensions in top_region_features
        # top_region_features now of shape [b x 1 x r x 1 x d]
        top_region_features = top_region_features.unsqueeze(2).unsqueeze(1)

        # x_wx of shape [b x a x r x 1 x d]
        x_wx = torch.matmul(top_region_features, self.wx) + self.wx_bias

        # normality_pool_image_features of shape [r, n, d]
        # wy of shape [r, d, d]
        # y_wy of shape [r, n, d]
        y_wy = torch.matmul(self.normality_pool_image_features, self.wy) + self.wy_bias

        # M are the attention weights (see equation 4 of paper) of shape [b x a x r x 1 x n]
        M = torch.matmul(x_wx, y_wy.transpose(-1, -2))

        # scale attention weights
        M = M / (top_region_features.size(-1) ** 0.5)

        M = nn.functional.softmax(M, dim=-1)

        # closest_normal_region_features of shape [b, a, r, 1, d]
        closest_normal_region_features = torch.matmul(M, self.normality_pool_image_features)

        # return the AGGREGATE_ATTENTION_NUM closest normal image features for all 36 regions of a single image
        return closest_normal_region_features
