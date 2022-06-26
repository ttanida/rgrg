import torch
from torch import nn
from torchinfo import summary
from transformers import GPT2Config
from transformers.pytorch_utils import Conv1D
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

d_model = 1024
conv1d = Conv1D(3 * d_model, d_model)
x = torch.rand(
    2, 4, d_model
)  # represents a sequence of batch_size=1, seq_len=4 and embedding_sz=768, something like "Hello how are you"
x = conv1d(x)
# print(x.shape)

query, key, value = x.split(d_model, dim=-1)
# print(query.shape)


class Conv1DWithTrainedWeights(nn.Module):
    """
    Same functionality as Conv1D class of transformers.pytorch_utils but allows initialization with trained weights.

    Conv1D has the same functionality as a linear layer.
    It transforms the inputted hidden_states from shape (batch x sequence_len x hidden_dim) to (batch x sequence_len x 3*hidden_dim),
    thus allowing the retrieval of the query, key and value matrices
    """
    def __init__(self, trained_weight, trained_bias):
        super(Conv1DWithTrainedWeights, self).__init__()
        self.weight = nn.Parameter(trained_weight)  # of shape (1024 x 3072)
        self.bias = nn.Parameter(trained_bias)  # of shape (3072)

    def forward(self, x):  # x has shape (batch x sequence_len x 1024)
        size_out = x.size()[:-1] + (self.weight.size(-1),)  # size_out has shape (batch x sequence_len x 3072)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x  # x has shape (batch x sequence_len x 3072)


class GPT2PseudoAttention(GPT2Attention):
    def __init__(
        self,
        c_attn_weights_and_bias: tuple[torch.FloatTensor],  # pre-trained weights and bias
        c_proj_weights_and_bias: tuple[torch.FloatTensor],  # pre-trained weights and bias
        is_cross_attention=False,
        layer_idx=None,
    ):
        config = GPT2Config(
            vocab_size=50257,
            n_positions=1024,
            n_embd=1024,
            n_layer=24,
            n_head=16,
        )
        super().__init__(config, is_cross_attention, layer_idx)
        self.c_attn = Conv1DWithTrainedWeights(trained_weight=c_attn_weights_and_bias[0], trained_bias=c_attn_weights_and_bias[1])
        self.c_proj = Conv1DWithTrainedWeights(trained_weight=c_proj_weights_and_bias[0], trained_bias=c_proj_weights_and_bias[1])


# c_attn_weights_and_bias = (torch.ones(1024, 3072)*5, torch.ones(3072))
# c_proj_weights_and_bias = (torch.zeros(1024, 3072), torch.zeros(3072))
# PSA = GPT2PseudoAttention(c_attn_weights_and_bias=c_attn_weights_and_bias, c_proj_weights_and_bias=c_proj_weights_and_bias)
# for param in PSA.named_parameters():
#     print(param)

# print(PSA)

# weight = torch.ones(1024, 3072)
# bias = torch.ones(3072)
# my_tuple = (weight, bias)
# conv1d = Conv1DWithTrainedWeights(trained_weight=my_tuple[0], trained_bias=my_tuple[1])
# for param in conv1d.named_parameters():
#     print(param)
