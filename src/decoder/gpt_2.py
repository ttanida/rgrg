from numpy import dtype
import torch
import torch.nn as nn
from torchinfo import summary
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel


class Decoder(nn.Module):
    """
    GPT2 model's high level structure (i.e. children):
    (0): (wte): word embedding layer (maps each id in the vocab to a vector of dimension 1024)
    (1): (wpe): positional encoding (maps each of the position in the input to a positional encoding vector also of dimension 1024)
    note: there are overall 1024 positions, since the context size (n_ctx, see model config) is 1024
    (2): (ModuleList): a list of 24 GPT2Blocks (since GPT2 medium has 24 stacked decoder layers) and a LayerNorm at the end
    (3): (lm_head): languaging modeling head, which is a linear layer that maps from the hidden dimension of 1024 to the vocab dimension of 50257
    -> the next word can be predicted by taking the argmax of the 50257-dimensional vector and selecting the corresponding word as the next word

    Each GPT2Block has the following structure:
    (0): (ln_1): LayerNorm
    (1): (attn): masked multi-head self-attention
    (2): (ln_2):LayerNorm
    (3): (mlp): feed-forward neural network

    Each (attn) self-attention block consists of:
    (0): (c_attn):
    (1): (c_proj):
    (2): (attn_dropout): Dropout layer
    (3): (resid_dropout): Dropout layer
    """
    def __init__(self):
        super().__init__()
        self.checkpoint = "healx/gpt-2-pubmed-medium"

        # use GPT2 model with language modeling head, since we want to generate phrases
        self.pretrained_model = GPT2LMHeadModel.from_pretrained(self.checkpoint)

        # freeze all parameters of the GPT2 model
        for param in self.pretrained_model.parameters():
            param.requires_grad = False



model = GPT2LMHeadModel.from_pretrained("healx/gpt-2-pubmed-medium")
# model.lm_head = nn.Linear(1024, 2048)
# summary(model)
for child in model.named_children():
    print(child)
# for param in model.named_parameters():
#     print(param)
#     print()
# print(model)















###################
###################

# # model_path = "/u/home/tanida/gpt-2-pubmed-medium"
# # tokenizer = AutoTokenizer.from_pretrained(model_path)

# checkpoint = "stanford-crfm/pubmed_gpt"
checkpoint = "healx/gpt-2-pubmed-medium"
tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)

# setting `pad_token_id` to `eos_token_id`:50256 for open-end generation
tokenizer.pad_token = tokenizer.eos_token

# the trained model uses <|endoftext|> as its start token (i.e. 50256)
print(tokenizer.bos_token_id)

# raw_inputs = [
#     "I've been waiting for a HuggingFace course my whole life.",
#     "You hate this so much!",
#     ""
# ]


# sequence = "I've been waiting for a HuggingFace course my whole life mediastinum"

# model_inputs = tokenizer(sequence)

# print(tokenizer.decode(model_inputs["input_ids"]))


# inputs = tokenizer(raw_inputs, padding="longest", truncation=True, max_length=1024, return_tensors="pt")
# print(inputs.keys())
# print('input ids: ', inputs['input_ids'])
# print('attention mask: ', inputs['attention_mask'])
# print('shape: ', inputs['input_ids'].shape)

# for _, output in inputs.items():
#     print(list(output.size()))

# model = AutoModel.from_pretrained(model_path)

# setting `pad_token_id` to `eos_token_id`:50256 for open-end generation
# model = GPT2Model.from_pretrained(checkpoint, pad_token_id=tokenizer.eos_token_id)
# print()
# model = GPT2LMHeadModel.from_pretrained(checkpoint, pad_token_id=tokenizer.eos_token_id)
# output = model.generate(inputs=torch.tensor([[50256]], dtype=torch.int), max_length=100)
# print(tokenizer.decode(output[0], skip_special_tokens=True))
# print(model)
# print()
# summary(model)
# print(model)

# print(model.config)

# print(model)
# summary(model, input_data=inputs)

# outputs = model(**inputs)
# print(type(outputs))
# print(outputs.last_hidden_state.shape)  # (batch_size x num_tokens x d_hidden)

