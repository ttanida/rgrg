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

        self.wx = nn.Parameter(torch.empty(AGGREGATE_ATTENTION_NUM, 2048, 2048), requires_grad=True)
        self.wy = nn.Parameter(torch.empty(AGGREGATE_ATTENTION_NUM, 2048, 2048), requires_grad=True)

        self.wx_bias = nn.Parameter(torch.empty(2048), requires_grad=True)
        self.wy_bias = nn.Parameter(torch.empty(2048), requires_grad=True)

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
        top_region_features: torch.FloatTensor  # shape [batch_size x 36 x 2048]
    ):
        """Forward method implements equations 4 - 6 in paper."""
        # b ^= batch_size
        # r ^= region_number (i.e. 36)
        # a ^= aggregate_attention_num (most likely 6)
        # d ^= dimensionality (i.e. 2048)
        # n ^= normality_pool_size (most likely 100 or 1000)
        x_wx = torch.einsum("brd, add, d -> brad", top_region_features, self.wx, self.wx_bias)
        y_wy = torch.einsum("rnd, add, d -> nrad", self.normality_pool_image_features, self.wy, self.wy_bias)

        # M are the attention weights (see equation 4 of paper)
        M = torch.einsum("brad, nrad -> bran", x_wx, y_wy)

        # scale attention weights
        M = M / (top_region_features.size(-1) ** 0.5)

        M = nn.functional.softmax(M, dim=-1)

        closest_normal_region_features = torch.einsum("bran, rnd -> brad", M, self.normality_pool_image_features)

        # return the AGGREGATE_ATTENTION_NUM closest normal image features for all 36 regions of a single image
        return closest_normal_region_features  # shape [batch_size x 36 x AGGREGATE_ATTENTION_NUM x 2048]
