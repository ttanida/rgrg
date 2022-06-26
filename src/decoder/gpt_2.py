import torch
import torch.nn as nn
from torchinfo import summary
from transformers import GPT2Config
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention


class Conv1DWithTrainedWeights(nn.Module):
    """
    Same functionality as Conv1D class of transformers.pytorch_utils but allows initialization with trained weights.

    Conv1D has the same functionality as a linear layer.
    It transforms the inputted hidden_states from shape (batch x sequence_len x hidden_dim) to (batch x sequence_len x 3*hidden_dim),
    thus allowing the retrieval of the query, key and value matrices
    """

    def __init__(self, trained_weight, trained_bias):
        super(Conv1DWithTrainedWeights, self).__init__()
        self.weight = nn.Parameter(trained_weight, requires_grad=False)  # of shape (1024 x 3072)
        self.bias = nn.Parameter(trained_bias, requires_grad=False)  # of shape (3072)

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
        self.c_attn = Conv1DWithTrainedWeights(
            trained_weight=c_attn_weights_and_bias[0],
            trained_bias=c_attn_weights_and_bias[1],
        )
        self.c_proj = Conv1DWithTrainedWeights(
            trained_weight=c_proj_weights_and_bias[0],
            trained_bias=c_proj_weights_and_bias[1],
        )


class DecoderModel(nn.Module):
    """
    GPT2 model's high level structure (i.e. children):
    (0): (wte): word embedding layer (maps each id in the vocab to an embedding vector of dimension 1024)
    (1): (wpe): positional encoding (maps each of the position in the input to a positional encoding vector also of dimension 1024)
    note: there are overall 1024 positions, see n_positions in model.config
    (2): (ModuleList): a list of 24 GPT2Blocks (since GPT2 medium has 24 stacked decoder layers) and a LayerNorm at the end
    (3): (lm_head): languaging modeling head, which is a linear layer that maps from the hidden dimension of 1024 to the vocab dimension of 50257
    -> the next word can be predicted by taking the argmax of the 50257-dimensional vector and selecting the corresponding word as the next word

    Each GPT2Block has the following structure:
    (0): (ln_1): LayerNorm
    (1): (attn): masked multi-head self-attention
    (2): (ln_2): LayerNorm
    (3): (mlp): feed-forward neural network

    Each (attn) self-attention block consists of:
    (0): (c_attn): Conv1D(3 * 1024, 1024) layer, which is a sort of linear layer that transforms an input tensor of the shape
    (batch_size, seq_len, hidden_dim) to (batch_size, seq_len, 3 * hidden_dim) to retrieve the query, key, value matrices

    note: the Conv1D layer is implemented in Huggingface (transformers.pytorch_utils.py) and is not to be confused with
    the PyTorch implementation (torch.nn.modules) that has a lowercase d (i.e. Conv1d)

    (1): (c_proj): Conv1D(1024, 1024) layer
    (2): (attn_dropout): Dropout layer
    (3): (resid_dropout): Dropout layer
    """

    def __init__(self):
        super().__init__()
        self.checkpoint = "healx/gpt-2-pubmed-medium"

        # use GPT2 model with language modeling head, since we want to generate phrases
        self.gpt_with_lm_head = GPT2LMHeadModel.from_pretrained(self.checkpoint)

        # freeze all parameters of the model
        for param in self.gpt_with_lm_head.parameters():
            param.requires_grad = False

        # replace normal attention layers by pseudo attention layers
        self._replace_attention_by_pseudo_attention()

        # divide model into GPT part and language modeling head part
        self.gpt = self.gpt_with_lm_head.transformer
        self.lm_head = self.gpt_with_lm_head.lm_head

        # divide GPT part into word and positional embedding, gpt2 blocks and final layernorm
        self.word_and_positional_embedding = nn.Sequential(*list(self.gpt.children())[:3])
        self.gpt2_blocks = list(self.gpt.children())[3]  # type: nn.ModuleList
        self.final_layernorm = list(self.gpt.children())[4]

        # convert each individual gpt2_block into a nn.ModuleList
        self.gpt2_blocks = nn.ModuleList(nn.ModuleList(gpt2_block.children()) for gpt2_block in self.gpt2_blocks)

        # small neural network to transform embeddings coming from the image feature space into embeddings in the text feature space
        self.feature_space_transformation_nn = nn.Sequential(
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=1024)
        )

    def _replace_attention_by_pseudo_attention(self):
        GPT2PSA_list = []

        for gpt2_block in self.gpt_with_lm_head.transformer.h:
            # extract trained weights and biases
            attn = gpt2_block.attn
            c_attn_weights = attn.c_attn.weight.detach()
            c_attn_bias = attn.c_attn.bias.detach()
            c_proj_weights = attn.c_proj.weight.detach()
            c_proj_bias = attn.c_proj.bias.detach()

            # initialize GPT2PseudoAttention module
            GPT2PSA = GPT2PseudoAttention(
                c_attn_weights_and_bias=(c_attn_weights, c_attn_bias),
                c_proj_weights_and_bias=(c_proj_weights, c_proj_bias),
            )

            GPT2PSA_list.append(GPT2PSA)

        for i, GPT2PSA in enumerate(GPT2PSA_list):
            self.gpt_with_lm_head.transformer.h[i].attn = GPT2PSA

    def forward(self, image_features, text_features):
        # transform image_features from image feature space to text feature space
        image_features = self.feature_space_transformation_nn(image_features)

        text_features = self.word_and_positional_embedding(text_features)
        for gpt2_block in self.gpt2_blocks:
            for i, gpt2_block_module in enumerate(gpt2_block):
                if i == 1:  # the second module is the pseudo self-attention module
                    text_features = gpt2_block_module(image_features, text_features)
                else:
                    text_features = gpt2_block_module(text_features)

        text_features = self.final_layernorm(text_features)
        text_features = self.lm_head(text_features)

        return text_features




# c_attn_weights_and_bias = (torch.ones(1024, 3072) * 5, torch.ones(3072))
# c_proj_weights_and_bias = (torch.zeros(1024, 3072), torch.zeros(3072))
# PSA = GPT2PseudoAttention(
#     c_attn_weights_and_bias=c_attn_weights_and_bias,
#     c_proj_weights_and_bias=c_proj_weights_and_bias,
# )

model = DecoderModel()
for name, module in model.named_modules():
    print(name)


# print(model.pretrained_model.transformer.h[0].attn.use_cache)

# model = GPT2LMHeadModel.from_pretrained("healx/gpt-2-pubmed-medium")
# my_model = nn.Sequential(*list(model.transformer.modules())[:2])

# print(my_model)

# gpt = model.transformer
# gpt2_blocks = list(gpt.children())[3]

# gpt2_blocks = nn.ModuleList(nn.ModuleList(gpt2_block.children()) for gpt2_block in gpt2_blocks)
# gpt2_block = gpt2_blocks[0]
# print(gpt2_block)




# print(len(list(model.transformer.children())))
# for i, child in enumerate(model.transformer.children()):
#     if i == 3:
#         print(child)




# for child in model.children():
#     print(child)


# my_dict = {}
# for name, module in model.named_modules():
#     if isinstance(module, GPT2Attention):
#         my_dict[name] = PSA

# for k, v in my_dict.items():
#     setattr(model, k, v)

# gpt2_block = model.transformer.h[1]
# print(gpt2_block.attn)

# model.transformer.h[0] = nn.Linear(1024, 1024)

# for name, module in model.named_modules():
#     print(name)

# first_GPT2_attention_module = model.transformer.h[0].attn
# for name, param in first_GPT2_attention_module.named_parameters():
#     print(name, param)



# print(first_GPT2_attention_module)
# c_attn_weights = first_GPT2_attention_module.c_attn.weight.detach()
# print(c_attn_weights.shape)

# for name_module, module in model.named_modules():
#     if isinstance(module, GPT2Attention):
#         print(name_module)

# print(model)

# i = 0
# for gpt2_block in model.transformer.h.children():
#     print(i)
#     for child in gpt2_block.children():
#         print(child)
#     i += 1
# print(f"{child[0]}: {child[1].shape}")

# print()
# summary(model.transformer.h[0].attn)
# print(model.transformer.h[0].attn)
# summary(torch.nn.Conv1d(in_channels=1024, out_channels=3072, kernel_size=1))
# for param in model.named_parameters():
#     print(param)
#     print()
# print(model)


###################
###################

# # model_path = "/u/home/tanida/gpt-2-pubmed-medium"
# # tokenizer = AutoTokenizer.from_pretrained(model_path)

# checkpoint = "stanford-crfm/pubmed_gpt"
# checkpoint = "healx/gpt-2-pubmed-medium"
# tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)

# setting `pad_token_id` to `eos_token_id`:50256 for open-end generation
# tokenizer.pad_token = tokenizer.eos_token

# the trained model uses <|endoftext|> as its start token (i.e. 50256)
# print(tokenizer.bos_token_id)

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
