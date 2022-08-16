import math
import os

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DifferentiateAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        top_region_features: torch.FloatTensor  # shape [batch_size x 36 x 2048]
    ):
        """Forward method implements equations 7 - 10 in paper."""
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
