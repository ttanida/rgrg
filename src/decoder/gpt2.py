import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torchinfo import summary
from transformers import GPT2Tokenizer, GPT2LMHeadModel


class Conv1DWithTrainedWeights(nn.Module):
    """
    Same functionality as Conv1D class of transformers.pytorch_utils but allows initialization with trained weights.

    Conv1D has the same functionality as a linear layer.
    It transforms the inputted hidden_states from shape (batch x sequence_len x hidden_dim) to (batch x sequence_len x 3*hidden_dim),
    thus allowing the retrieval of the query, key and value matrices
    """

    def __init__(self, trained_weight, trained_bias):
        super(Conv1DWithTrainedWeights, self).__init__()
        self.weight = nn.Parameter(trained_weight, requires_grad=False)  # of shape (hidden_dim x 3*hidden_dim)
        self.bias = nn.Parameter(trained_bias, requires_grad=False)  # of shape (3 * hidden_dim)

    def forward(self, x):  # x has shape (batch x sequence_len x hidden_dim)
        size_out = x.size()[:-1] + (self.weight.size(-1),)  # size_out has shape (batch x sequence_len x 3*hidden_dim)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x  # x has shape (batch x sequence_len x 3*hidden_dim)


class GPT2PseudoAttention(nn.Module):
    def __init__(
        self,
        c_attn_weights_and_bias: tuple[torch.FloatTensor],  # pre-trained weights and bias for retrieving query, key, value matrices
        c_proj_weights_and_bias: tuple[torch.FloatTensor],  # pre-trained weights and bias for projecting concatenated heads to original hidden dim
    ):

        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.c_attn = Conv1DWithTrainedWeights(
            trained_weight=c_attn_weights_and_bias[0],
            trained_bias=c_attn_weights_and_bias[1],
        )
        self.c_proj = Conv1DWithTrainedWeights(
            trained_weight=c_proj_weights_and_bias[0],
            trained_bias=c_proj_weights_and_bias[1],
        )

        self.embed_dim = 1024
        self.num_heads = 16
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim

        self.attn_dropout = nn.Dropout(p=0.1)
        self.resid_dropout = nn.Dropout(p=0.1)

        # seq_len can maximally be 1024 tokens
        max_positions = 1024

        # create a causal mask for masking out attention weights in the masked self-attention operator (masking out weights of tokens that lie ahead of the attended token)
        # first create a lower triangular matrix
        lower_triangular_matrix = torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8))
        # then save lower_triangular_matrix (with additional dimensions for batch_size and num_heads) in a buffer
        # (to make sure the causal mask does not get updated during backprop)
        self.register_buffer("causal_mask", lower_triangular_matrix.view(1, 1, max_positions, max_positions))

        # value for masking out attention weights
        self.register_buffer("mask_out_value", torch.tensor(-1e4))

        # matrices for getting key and value matrices for image hidden states
        self.uk = nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim)
        self.uv = nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim)

    def _split_heads(self, tensor, num_heads, head_dim):
        """
        Splits hidden_dim (i.e. 1024) into num_heads (i.e. 16) and head_dim (i.e. 64)
        """
        new_shape = tensor.size()[:-1] + (num_heads, head_dim)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)

    def _attn(self, query_word, key_image_word, value_image_word, attention_mask):
        attn_weights = torch.matmul(query_word, key_image_word.transpose(-1, -2))  # shape (batch_size x num_heads x seq_len x 1+seq_len)

        # scale attention weights
        attn_weights = attn_weights / (value_image_word.size(-1) ** 0.5)

        # create and apply the final causal mask to weights
        query_length, key_length = query_word.size(-2), key_image_word.size(-2)

        # note that this causal mask has a shape of seq_len x 1+seq_len (in the last 2 dims),
        # with the first column of the mask only consisting of True boolean values
        # meaning attention weights corresponding to images (which are stored in the first column) are not masked out!
        causal_mask = self.causal_mask[:, :, key_length - query_length: key_length, :key_length].to(torch.bool)

        # select the attention weights where the causal mask has True values, select -1e4 where the causal mask has False values
        attn_weights = torch.where(causal_mask, attn_weights, self.mask_out_value.to(attn_weights.dtype))

        # apply the attention mask of shape (batch_size, 1, 1, 1+seq_len) for masking out padding tokens
        # there is an additional column of zeros for the attention weights corresponding to the image,
        # such that these are not masked out
        attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # downcast (if necessary) back to V's dtype (if in mixed-precision) -- no-op otherwise
        attn_weights = attn_weights.type(value_image_word.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value_image_word)  # shape (batch_size x num_heads x seq_len x head_dim)

        return attn_output

    def _merge_heads(self, tensor, num_heads, head_dim):
        """
        Merges num_heads (i.e. 16) and head_dim (i.e. 64) into hidden_dim (i.e. 1024)
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * head_dim,)
        return tensor.view(new_shape)

    def forward(self,
                word_hidden_states,  # shape (batch_size x seq_len x hidden_dim)
                image_hidden_states,  # shape (batch_size x hidden_dim)
                attention_mask):  # shape (batch_size, 1, 1, 1+seq_len)

        # query, key, value matrices each have shape (batch_size x seq_len x hidden_dim)
        query_word, key_word, value_word = self.c_attn(word_hidden_states).split(self.split_size, dim=2)

        # add an addition dimension to the image_hidden_states
        image_hidden_states = image_hidden_states[:, None, :]  # shape (batch_size x 1 x hidden_dim)

        key_image = self.uk(image_hidden_states)  # shape (batch_size x 1 x hidden_dim)
        value_image = self.uv(image_hidden_states)  # shape (batch_size x 1 x hidden_dim)

        key_image_word = torch.cat((key_image, key_word), dim=1)  # shape (batch_size x 1+seq_len x hidden_dim)
        value_image_word = torch.cat((value_image, value_word), dim=1)  # shape (batch_size x 1+seq_len x hidden_dim)

        query_word = self._split_heads(query_word, self.num_heads, self.head_dim)  # shape (batch_size x num_heads x seq_len x head_dim)
        key_image_word = self._split_heads(key_image_word, self.num_heads, self.head_dim)  # shape (batch_size x num_heads x 1+seq_len x head_dim)
        value_image_word = self._split_heads(value_image_word, self.num_heads, self.head_dim)  # shape (batch_size x num_heads x 1+seq_len x head_dim)

        attn_output = self._attn(query_word, key_image_word, value_image_word, attention_mask)  # shape (batch_size x num_heads x seq_len x head_dim)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)  # shape (batch_size x seq_len x hidden_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output  # shape (batch_size x seq_len x hidden_dim)


class DecoderModel(nn.Module):
    """
    GPT2 model with a language modeling head and pseudo self-attention.

    Pseudo self-attention is based on the papar Encoder-Agnostic Adaptation for Conditional Language Generation (https://arxiv.org/abs/1908.06938).
    It is a technique to condition a pretrained language model to arbitrary conditional input (in my case features of chest x-ray images).

    The code is largely the same as the GPT2 implementation by Huggingface (https://github.com/huggingface/transformers/blob/d0acc9537829e7d067edbb791473bbceb2ecf056/src/transformers/models/gpt2/modeling_gpt2.py),
    except for the custom GPT2PseudoAttention class replacing the GPT2Attention class.

    Recommended reading to understand the GPT2 source code: https://amaarora.github.io/2020/02/18/annotatedGPT2.html
    """

    def __init__(self):
        super().__init__()
        self.checkpoint = "healx/gpt-2-pubmed-medium"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        # divide GPT part into word embedding layer, positional embedding layer, dropout layer, gpt2 blocks and final layernorm
        gpt_children = list(self.gpt.children())
        self.wte = gpt_children[0]  # word embedding layer
        self.wpe = gpt_children[1]  # positional embedding layer
        self.drop = gpt_children[2]  # dropout layer
        self.gpt2_blocks = gpt_children[3]  # type: nn.ModuleList
        self.final_layernorm = gpt_children[4]

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

    def forward(self,
                input_ids: torch.LongTensor,  # shape (batch_size x seq_len)
                attention_mask: torch.FloatTensor,   # shape (batch_size x seq_len)
                image_hidden_states: torch.FloatTensor,   # shape (batch_size x image_hidden_dim) (with image_hidden_dim = 1024, so same as word_hidden_dim)
                return_loss: bool = False
                ):
        """
        If return_loss is False, returns language modeling logits (of shape batch_size x seq_len x vocab_size).
        If return_loss is True, returns a tuple of the cross entropy loss and language modeling logits.

        To compute the loss, the input_ids are used as labels, by shifting them by one position
        (see shift_logits and shift_labels variables towards the end of the forward method).
        """

        # transform image_hidden_states from image feature space to text feature space
        image_hidden_states = self.feature_space_transformation_nn(image_hidden_states)  # shape (batch_size x word_hidden_dim)

        # from now, word_hidden_dim will just be called hidden_dim

        # pass the token ids through the word embedding layer to get the word embeddings
        inputs_embeds = self.wte(input_ids)  # shape (batch_size x seq_len x hidden_dim)
        seq_len = inputs_embeds.size(1)

        # position_ids is a tensor that specifies the position of each token in the input (necessary to create positional embeddings)
        device = input_ids.device
        position_ids = torch.arange(start=0, end=seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_len)  # shape (1 x seq_len)

        # pass the position ids through the positional embedding layer to get the positional embeddings
        position_embeds = self.wte(position_ids)  # shape (1 x seq_len x hidden_dim)

        # addition is broadcasted around batch_size dimension
        word_hidden_states = inputs_embeds + position_embeds  # shape (batch_size x seq_len x hidden_dim)

        word_hidden_states = self.drop(word_hidden_states)

        # we change the attention_mask shape to (batch_size, 1, 1, seq_len), since the attention_mask is later applied to the last dimension of
        # the attention weights that are of shape (batch_size x num_heads x seq_len x 1+seq_len)
        attention_mask = attention_mask[:, None, None, :]

        # since we have 1 additional column in the attention weights (i.e. 1+seq_len in the last dimension) due to the additional concatenated key matrix
        # of the image hidden states (see forward method of GPT2PseudoAttention), we have to shift the attention mask "one to the right" and add a column of ones
        # to the left such that the attention weights corresponding to the image are not masked out
        attention_mask_size = attention_mask.size()
        ones_column = torch.ones(attention_mask_size[:-1] + (1,), dtype=torch.int64).to(self.device)  # shape (batch_size, 1, 1, 1)
        attention_mask = torch.cat((ones_column, attention_mask), dim=-1)  # shape (batch_size, 1, 1, 1+seq_len)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely
        attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # dtype should be either torch.float32 or torch.float16
        attention_mask = (1.0 - attention_mask) * -10000.0

        for gpt2_block in self.gpt2_blocks:
            layer_norm_1 = gpt2_block[0]
            pseudo_self_attention = gpt2_block[1]
            layer_norm_2 = gpt2_block[2]
            mlp = gpt2_block[3]

            residual = word_hidden_states
            word_hidden_states = layer_norm_1(word_hidden_states)
            word_hidden_states = pseudo_self_attention(word_hidden_states, image_hidden_states, attention_mask)

            # residual connection
            word_hidden_states = word_hidden_states + residual

            residual = word_hidden_states
            word_hidden_states = layer_norm_2(word_hidden_states)
            word_hidden_states = mlp(word_hidden_states)

            # residual connection
            word_hidden_states = word_hidden_states + residual

        word_hidden_states = self.final_layernorm(word_hidden_states)

        lm_logits = self.lm_head(word_hidden_states)  # shape (batch_size x seq_len x vocab_size), with vocab_size = 50257

        if return_loss:
            # use input_ids as ground_truth labels
            labels = input_ids

            # shift the tokens, i.e. discard the last token in the sequence for the logits,
            # and discard the first token in the sequence for the labels

            # this way, the logits of the first token are "aligned" with the second token label,
            # the logits of the second token are "aligned" with the third token label, and so on...
            # since the previous token should predict the next token

            # only exception is if seq_len == 1, since this means that all the sequences in the batch only consist
            # of the eos token, meaning all of them were originally empty phrases (i.e. "")
            # in this case, we don't shift the logits/labels, because the single logit should predict end of sentence
            # (theoretically, it could be possible that the batch consists of batch_size sequences of exactly 1 eos or non-eos token,
            # but this would be too improbable)

            if seq_len == 1:
                shift_logits = lm_logits  # shape (batch_size x 1 x vocab_size)
                shift_labels = labels  # shape (batch_size x 1)
            else:
                shift_logits = lm_logits[:, :-1, :].contiguous()  # shape (batch_size x seq_len-1 x vocab_size)
                shift_labels = labels[:, 1:].contiguous()  # shape (batch_size x seq_len-1)

            # flatten the tokens
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))  # shape (batch_size*seq_len-1 x vocab_size)
            shift_labels = shift_labels.view(-1)  # shape (batch_size * seq_len-1)

            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits, shift_labels)

        return (loss, lm_logits) if return_loss else lm_logits


def print_model_summary(verbose):
    checkpoint = "healx/gpt-2-pubmed-medium"
    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    # use a batch of 3 phrases
    raw_inputs = [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
        ""]

    inputs = tokenizer(raw_inputs, padding="longest", truncation=True, max_length=1024, return_tensors="pt")

    # add a batch of 3 image hidden states
    inputs["image_hidden_states"] = torch.rand(3, 1024)

    if verbose > 0:
        print(inputs.keys())
        print('input ids: ', inputs['input_ids'])
        print('attention mask: ', inputs['attention_mask'])
        print('image_hidden_states mask: ', inputs['image_hidden_states'])
        print('shape: ', inputs['input_ids'].shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DecoderModel()
    model.to(device)

    inputs = inputs.to(device)

    if verbose == 0:
        summary(model)
    else:
        summary(model, input_data=dict(inputs), verbose=verbose)


# choose between:
# verbose = 0 (only model params)
# verbose = 1 (model params and output shape of batch)
# verbose = 2 (model params and output shape of batch, more detailed)
# print_model_summary(verbose=1)

# TODO: Implement generate function for DecoderModel

# checkpoint = "healx/gpt-2-pubmed-medium"
# tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
# tokenizer.pad_token = tokenizer.eos_token
# print(tokenizer("<|endoftext|>", truncation=True, max_length=1024))

# phrase = "I love huggingface"
# if len(phrase) == 0:
#     print(tokenizer.eos_token)
#     print(tokenizer(tokenizer.eos_token))
# else:
#     print(tokenizer(phrase))

# input = tokenizer.encode("", return_tensors="pt", add_special_tokens=True)
# print(input)
# print(tokenizer.bos_token)

# # use a batch of 3 phrases
# raw_inputs = [
#     "I've been waiting for a HuggingFace course my whole life.",
#     "I hate this so much!",
#     ""]

# inputs = tokenizer(raw_inputs, padding="longest", truncation=True, max_length=1024, return_tensors="pt")

# # add a batch of 3 image hidden states
# inputs["image_hidden_states"] = torch.rand(3, 1024)
# inputs["labels"] = torch.randint(low=0, high=10, size=(3, 14))

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = GPT2LMHeadModel.from_pretrained(checkpoint)
# input = tokenizer.encode(f"{tokenizer.bos_token}", return_tensors="pt")
# print(input)
# output = model.generate(input)
# print(output)
# dec_output_without_special_tokens = tokenizer.decode(output[0], skip_special_tokens=True)
# dec_output_with_special_tokens = tokenizer.decode(output[0], skip_special_tokens=False)
# print(dec_output_without_special_tokens)
# print(dec_output_with_special_tokens)


# inputs = inputs.to(device)

# output = model(**inputs)
# print(output)
# print(output[0])
# print(output[1])
# print(output[0].shape)
# print(output[1].shape)
