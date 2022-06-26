import torch
from transformers import GPT2Config
from transformers.pytorch_utils import Conv1D
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

d_model = 1024
conv1d  = Conv1D(3 * d_model, d_model)
x = torch.rand(2, 4, d_model)  # represents a sequence of batch_size=1, seq_len=4 and embedding_sz=768, something like "Hello how are you"
x = conv1d(x)
print(x.shape)

query, key, value = x.split(d_model, dim=-1)
print(query.shape)


class GPT2PseudoAttention(GPT2Attention):
    def __init__(self, is_cross_attention=False, layer_idx=None):
        config = GPT2Config(
            vocab_size=50257,
            n_positions=1024,
            n_embd=1024,
            n_layer=24,
            n_head=16,
        )
        super().__init__(config, is_cross_attention, layer_idx)


PSA = GPT2PseudoAttention()
print(PSA)
