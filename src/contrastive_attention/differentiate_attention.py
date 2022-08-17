import math

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DifferentiateAttention(nn.Module):
    def __init__(self):
        super().__init__()

        # wx and wy are the weights in the attention operator
        self.wx = nn.Parameter(torch.empty(2048, 2048), requires_grad=True)
        self.wy = nn.Parameter(torch.empty(2048, 2048), requires_grad=True)

        self.wx_bias = nn.Parameter(torch.empty(2048), requires_grad=True)
        self.wy_bias = nn.Parameter(torch.empty(2048), requires_grad=True)

        # w is the weight for the final linear transformation
        # (note that final hidden_dim is 1024, since this is what binary classifiers and language model expect)
        self.w = nn.Parameter(torch.empty(1024, 2048 * 2), requires_grad=True)
        self.w_bias = nn.Parameter(torch.empty(1024), requires_grad=True)

        self.reset_parameters(self.wx, self.wx_bias)
        self.reset_parameters(self.wy, self.wy_bias)
        self.reset_parameters(self.w, self.w_bias)

    def reset_parameters(self, weight, bias):
        """Use same weight initialization as for Linear layers (https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear)"""
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(bias, -bound, bound)

    def forward(
        self,
        closest_normal_region_features: torch.FloatTensor,  # shape [batch_size x 36 x AGGREGATE_ATTENTION_NUM x 2048]
        top_region_features: torch.FloatTensor  # shape [batch_size x 36 x 2048]
    ):
        """
        Forward method implements equations 7 - 10 in paper.

        subscripts for einsum notation:

        b = batch_size
        r = region_number (i.e. 36)
        a = aggregate_attention_num (i.e. num of closest images by aggregation, most likely 6)
        d = dimensionality (i.e. 2048)
        """
        top_region_features = torch.unsqueeze(top_region_features, dim=2)

        # v_P of shape [batch_size x 36 x (1 + AGGREGATE_ATTENTION_NUM) x 2048]
        v_P = torch.cat([top_region_features, closest_normal_region_features], dim=2)

        x_wx = torch.einsum("brad, dd, d -> brad", v_P, self.wx, self.wx_bias)
        y_wy = torch.einsum("brad, dd, d -> brad", v_P, self.wy, self.wy_bias)

        # M of shape [batch_size x 36 x (1 + AGGREGATE_ATTENTION_NUM) x (1 + AGGREGATE_ATTENTION_NUM)]
        M = torch.matmul(x_wx, y_wy.transpose(-1, -2))

        # scale attention weights
        M = M / (top_region_features.size(-1) ** 0.5)

        M = nn.functional.softmax(M, dim=-1)

        # attn_output of shape [batch_size x 36 x (1 + AGGREGATE_ATTENTION_NUM) x 2048]
        attn_output = torch.einsum("braa, brad -> brad", M, v_P)

        # common_information of shape [batch_size x 36 x 2048]
        # avg pooling in the 2nd dimension
        common_information = torch.mean(attn_output, dim=2, keepdim=False)

        top_region_features = torch.squeeze(top_region_features)

        contrastive_information = top_region_features - common_information

        # concat_information of shape [batch_size x 36 x (2048 * 2)]
        concat_information = torch.cat([top_region_features, contrastive_information], dim=2)

        top_region_features_with_contrastive_information = F.relu(F.linear(concat_information, self.w, self.w_bias))

        return top_region_features_with_contrastive_information  # shape [batch_size x 36 x 1024]
