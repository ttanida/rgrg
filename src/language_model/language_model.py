from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torchinfo import summary
from transformers import GPT2LMHeadModel
from transformers.generation_beam_search import BeamSearchScorer


class Conv1DWithTrainedWeights(nn.Module):
    """
    Same functionality as Conv1D class of transformers.pytorch_utils but allows initialization with trained weights.

    Conv1D has the same functionality as a linear layer.
    It transforms the inputted hidden_states from shape [batch x sequence_len x hidden_dim] to [batch x sequence_len x 3*hidden_dim],
    thus allowing the retrieval of the query, key and value matrices
    """

    def __init__(self, trained_weight, trained_bias):
        super(Conv1DWithTrainedWeights, self).__init__()
        self.weight = nn.Parameter(trained_weight, requires_grad=False)  # of shape [hidden_dim x 3*hidden_dim] for c_attn, of shape [hidden_dim x hidden_dim] for c_proj
        self.bias = nn.Parameter(trained_bias, requires_grad=False)  # of shape [3 * hidden_dim] for c_attn, of shape [hidden_dim] for c_proj

    def forward(self, x):  # x has shape [batch x sequence_len x hidden_dim]
        size_out = x.size()[:-1] + (self.weight.size(-1),)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x  # x has shape [batch x sequence_len x 3*hidden_dim] for c_attn, shape [batch x sequence_len x hidden_dim] for c_proj


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
        attn_weights = torch.matmul(query_word, key_image_word.transpose(-1, -2))  # shape [batch_size x num_heads x seq_len x 1+seq_len]

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

        # apply the attention mask of shape [batch_size, 1, 1, 1+seq_len] for masking out padding tokens
        # there is an additional column of zeros for the attention weights corresponding to the image,
        # such that these are not masked out
        attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # downcast (if necessary) back to V's dtype (if in mixed-precision) -- no-op otherwise
        attn_weights = attn_weights.type(value_image_word.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value_image_word)  # shape [batch_size x num_heads x seq_len x head_dim]

        return attn_output

    def _merge_heads(self, tensor, num_heads, head_dim):
        """
        Merges num_heads (i.e. 16) and head_dim (i.e. 64) into hidden_dim (i.e. 1024)
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * head_dim,)
        return tensor.view(new_shape)

    def forward(self,
                word_hidden_states,  # shape [batch_size x seq_len x hidden_dim]
                image_hidden_states,  # shape [batch_size x hidden_dim]
                attention_mask,  # shape [batch_size, 1, 1, 1+seq_len]
                layer_past,
                use_cache):

        # query, key, value matrices each have shape [batch_size x seq_len x hidden_dim]
        q_word, k_word, v_word = self.c_attn(word_hidden_states).split(self.split_size, dim=2)

        # if layer_past is None, we are either training the model or generating the first token in text generation mode
        if layer_past is None:
            # add an addition dimension to the image_hidden_states
            image_hidden_states = image_hidden_states[:, None, :]  # shape [batch_size x 1 x hidden_dim]

            # get the key and value matrices for the image hidden states
            k_image = self.uk(image_hidden_states)  # shape [batch_size x 1 x hidden_dim]
            v_image = self.uv(image_hidden_states)  # shape [batch_size x 1 x hidden_dim]

            # if the batch_size is different, then we are in beam search generation mode (adjust k and v image matrices accordingly)
            if k_image.size(0) != k_word.size(0):
                num_beams = k_word.size(0) // k_image.size(0)
                k_image = k_image.repeat_interleave(num_beams, dim=0)
                v_image = v_image.repeat_interleave(num_beams, dim=0)

            k_image_word = torch.cat((k_image, k_word), dim=1)  # shape [batch_size x 1+seq_len x hidden_dim]
            v_image_word = torch.cat((v_image, v_word), dim=1)  # shape [batch_size x 1+seq_len x hidden_dim]

            q_word = self._split_heads(q_word, self.num_heads, self.head_dim)  # shape [batch_size x num_heads x seq_len x head_dim]
            k_image_word = self._split_heads(k_image_word, self.num_heads, self.head_dim)  # shape [batch_size x num_heads x 1+seq_len x head_dim]
            v_image_word = self._split_heads(v_image_word, self.num_heads, self.head_dim)  # shape [batch_size x num_heads x 1+seq_len x head_dim]

            if use_cache is True:
                present = (k_image_word, v_image_word)
            else:
                present = None

            attn_output = self._attn(q_word, k_image_word, v_image_word, attention_mask)  # shape [batch_size x num_heads x seq_len x head_dim]
        else:
            # if there is a layer_past (which stores key and value tensors of past tokens), then this means we are in text generation mode
            q_word = self._split_heads(q_word, self.num_heads, self.head_dim)  # shape [batch_size x num_heads x 1 x head_dim]
            k_word = self._split_heads(k_word, self.num_heads, self.head_dim)  # shape [batch_size x num_heads x 1 x head_dim]
            v_word = self._split_heads(v_word, self.num_heads, self.head_dim)  # shape [batch_size x num_heads x 1 x head_dim]

            past_key, past_value = layer_past
            k = torch.cat((past_key, k_word), dim=-2)
            v = torch.cat((past_value, v_word), dim=-2)

            present = (k, v)

            attn_output = self._attn(q_word, k, v, attention_mask)  # shape [batch_size x num_heads x seq_len x head_dim]

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)  # shape [batch_size x seq_len x hidden_dim]
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)  # shape [batch_size x seq_len x hidden_dim]

        return attn_output, present


class LanguageModel(nn.Module):
    """
    GPT2 model with a language modeling head and pseudo self-attention.

    Pseudo self-attention is based on the papar "Encoder-Agnostic Adaptation for Conditional Language Generation" (https://arxiv.org/abs/1908.06938).
    It is a technique to condition a pretrained language model to arbitrary conditional input (in my case features of chest x-ray images).

    The code is largely the same as the GPT2 implementation by Huggingface (https://github.com/huggingface/transformers/blob/d0acc9537829e7d067edbb791473bbceb2ecf056/src/transformers/models/gpt2/modeling_gpt2.py),
    except for the custom GPT2PseudoAttention class replacing the GPT2Attention class.

    Recommended reading to understand the GPT2 source code: https://amaarora.github.io/2020/02/18/annotatedGPT2.html
    """

    def __init__(self):
        super().__init__()
        self.checkpoint = "healx/gpt-2-pubmed-medium"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bos_token_id = 50256
        self.eos_token_id = 50256
        self.pad_token_id = 50256

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
                input_ids: torch.LongTensor,  # shape [batch_size x seq_len]
                attention_mask: torch.FloatTensor,  # shape [batch_size x seq_len]
                image_hidden_states: torch.FloatTensor,  # shape [batch_size x image_hidden_dim] (with image_hidden_dim = 1024)
                return_loss: bool = False,
                past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                position_ids: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = False
                ):
        """
        If return_loss is True, returns the language modeling loss.
        If return_loss is False (in which we are in text generation mode and use_cache will be True), returns the language modeling logits (of shape batch_size x seq_len x vocab_size)
        as well as the so-called presents (which store the key and value tensors of the previous tokens, such that they don't have to be recomputed every time during text generation).

        To compute the loss, the input_ids are used as labels.
        To prevent padding tokens from counting towards the loss, the attention_mask is transformed to a boolean mask and inverted.
        Then this inverted boolean mask is used to set all padding token ids to -100.
        In the cross entropy loss, the ignore_index is set to -100, such that padding token ids are ignored as targets.

        Furthermore, the label at the first position of the sequence is discarded and the labels are shifted accordingly (i.e. one to the left),
        such that the language modeling logits align with the labels that they are trying to predict.
        """
        # get a boolean copy of the attention_mask and invert it
        mask_to_ignore_padding_tokens_for_loss_computation = ~(attention_mask.to(torch.bool))

        # transform image_hidden_states from image feature space to text feature space
        image_hidden_states = self.feature_space_transformation_nn(image_hidden_states)  # shape [batch_size x word_hidden_dim], with word_hidden_dim = 1024

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]

        # pass the token ids through the word embedding layer to get the word embeddings
        inputs_embeds = self.wte(input_ids)  # shape [batch_size x seq_len x hidden_dim]

        # position_ids is a tensor that specifies the position of each token in the input (necessary to create positional embeddings)
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.gpt2_blocks))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=self.device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])  # shape [1 x seq_len]

        # pass the position ids through the positional embedding layer to get the positional embeddings
        position_embeds = self.wte(position_ids)  # shape [1 x seq_len x hidden_dim]

        # addition is broadcasted around batch_size dimension
        word_hidden_states = inputs_embeds + position_embeds  # shape [batch_size x seq_len x hidden_dim]

        word_hidden_states = self.drop(word_hidden_states)

        output_shape = input_shape + (word_hidden_states.size(-1),)

        # we change the attention_mask shape to [batch_size, 1, 1, seq_len], since the attention_mask is later applied to the last dimension of
        # the attention weights that are of shape [batch_size x num_heads x seq_len x 1+seq_len]
        attention_mask = attention_mask.view(batch_size, -1)
        attention_mask = attention_mask[:, None, None, :]

        # since we have 1 additional column in the attention weights (i.e. 1+seq_len in the last dimension) due to the additional concatenated key matrix
        # of the image hidden states (see forward method of GPT2PseudoAttention), we have to shift the attention mask "one to the right" and add a column of ones
        # to the left such that the attention weights corresponding to the image are not masked out
        attention_mask_size = attention_mask.size()
        ones_column = torch.ones(attention_mask_size[:-1] + (1,), dtype=torch.int64, device=self.device)  # shape [batch_size, 1, 1, 1]
        attention_mask = torch.cat((ones_column, attention_mask), dim=-1)  # shape [batch_size, 1, 1, 1+seq_len]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely
        attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # dtype should be either torch.float32 or torch.float16
        attention_mask = (1.0 - attention_mask) * -10000.0

        presents = () if use_cache else None

        for gpt2_block, layer_past in zip(self.gpt2_blocks, past_key_values):
            layer_norm_1 = gpt2_block[0]
            pseudo_self_attention = gpt2_block[1]
            layer_norm_2 = gpt2_block[2]
            mlp = gpt2_block[3]

            residual = word_hidden_states
            word_hidden_states = layer_norm_1(word_hidden_states)

            word_hidden_states, present = pseudo_self_attention(word_hidden_states, image_hidden_states, attention_mask, layer_past, use_cache)

            # residual connection
            word_hidden_states = word_hidden_states + residual

            residual = word_hidden_states
            word_hidden_states = layer_norm_2(word_hidden_states)
            word_hidden_states = mlp(word_hidden_states)

            # residual connection
            word_hidden_states = word_hidden_states + residual

            if use_cache:
                presents += (present,)

        word_hidden_states = self.final_layernorm(word_hidden_states)

        word_hidden_states = word_hidden_states.view(output_shape)

        lm_logits = self.lm_head(word_hidden_states)  # shape [batch_size x seq_len x vocab_size], with vocab_size = 50257

        if return_loss:
            # use input_ids as ground_truth labels
            labels = input_ids

            # set padding tokens to -100, such that they are ignored and don't count towards the loss
            labels[mask_to_ignore_padding_tokens_for_loss_computation] = -100

            # shift the tokens, i.e. discard the last token in the sequence for the logits,
            # and discard the first token in the sequence for the labels

            # this way, the logits of the first token are "aligned" with the second token label,
            # the logits of the second token are "aligned" with the third token label, and so on...
            # since the previous token should predict the next token

            # discard the last lm_logit corresponding to the last token
            shift_logits = lm_logits[:, :-1, :].contiguous()  # shape [batch_size x seq_len-1 x vocab_size]

            # discard the first token in the sequence
            shift_labels = labels[:, 1:].contiguous()  # shape [batch_size x seq_len-1]

            # flatten the tokens
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))  # shape [batch_size*seq_len-1 x vocab_size]
            shift_labels = shift_labels.view(-1)  # shape [batch_size * seq_len-1]

            # padding tokens are ignored for loss computation, and loss is averaged over non-ignored targets
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits, shift_labels)

            return loss

        if use_cache:
            return lm_logits, presents

    @torch.no_grad()
    def generate(self,
                 image_hidden_states: torch.FloatTensor,  # shape [batch_size x image_hidden_dim]
                 max_length: int = None,
                 num_beams: int = 1,
                 num_beam_groups: int = 1,
                 do_sample: bool = False,
                 num_return_sequences: int = 1,
                 early_stopping: bool = False
                 ) -> torch.LongTensor:  # shape [batch_size x longest_generated_sequence_length]
        """
        Generates output ids for a batch of image features.
        These output ids can then be decoded by the tokenizer to get the generated sentences.
        """
        batch_size = image_hidden_states.size(0)

        # start with the bos_token_id for all image features in the batch.
        input_ids = torch.full(size=(batch_size, 1), fill_value=self.bos_token_id, dtype=torch.int64, device=self.device)
        model_kwargs = {"attention_mask": torch.ones(size=(batch_size, 1), dtype=torch.int64, device=self.device),
                        "use_cache": True}

        is_greedy_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is False
        is_sample_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is True
        is_beam_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is False
        is_beam_sample_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is True
        is_group_beam_gen_mode = (num_beams > 1) and (num_beam_groups > 1)

        if num_beam_groups > num_beams:
            raise ValueError("'num_beam_groups' has to be smaller or equal to 'num_beams'")
        if is_group_beam_gen_mode and do_sample is True:
            raise ValueError(
                "Diverse beam search cannot be used in sampling mode. Make sure that 'do_sample' is set to 'False'."
            )

        # go into different generation modes
        if is_greedy_gen_mode:
            if num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {num_return_sequences} when doing greedy search."
                )

            return self.greedy_search(
                input_ids,
                image_hidden_states,
                max_length,
                **model_kwargs
            )
        elif is_sample_gen_mode:
            raise NotImplementedError("Multinomial sampling is not implemented.")
        elif is_beam_gen_mode:
            if num_return_sequences > num_beams:
                raise ValueError("'num_return_sequences' has to be smaller or equal to 'num_beams'.")

            if max_length is None:
                raise ValueError("max_length has to be set for beam generation.")

            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                device=self.device,
                length_penalty=1.0,  # length_penalty > 0.0 encourages the model to generate shorter sequences
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )

            # interleave input_ids with 'num_beams' additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(input_ids, expand_size=num_beams, **model_kwargs)

            return self.beam_search(
                input_ids,
                image_hidden_states,
                max_length,
                beam_scorer,
                **model_kwargs,
            )
        elif is_beam_sample_gen_mode:
            raise NotImplementedError("Beam-search multinomial sampling is not implemented.")
        elif is_group_beam_gen_mode:
            raise NotImplementedError("Diverse beam-search decoding is not implemented.")

    def _expand_inputs_for_generation(self, input_ids, expand_size, attention_mask, **model_kwargs):
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        return input_ids, model_kwargs

    def _reorder_cache(self, past, beam_idx):
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        # only use last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

    def _update_model_kwargs_for_generation(self, presents, model_kwargs):
        model_kwargs["past"] = presents
        attention_mask = model_kwargs["attention_mask"]
        model_kwargs["attention_mask"] = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)

        return model_kwargs

    def beam_search(self,
                    input_ids,
                    image_hidden_states,
                    max_length,
                    beam_scorer,
                    **model_kwargs):
        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of 'input_ids' should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        while True:
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            lm_logits, presents = self.forward(**model_inputs, image_hidden_states=image_hidden_states, return_loss=False)

            next_token_logits = lm_logits[:, -1, :]

            next_token_scores = nn.functional.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)

            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
                # beam_indices=None,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(presents, model_kwargs)

            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)

            # increase cur_len
            cur_len += 1

            if beam_scorer.is_done or (max_length and cur_len >= max_length):
                break

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            max_length=max_length,
        )

        return sequence_outputs["sequences"]

    def greedy_search(self,
                      input_ids,  # shape [batch_size x seq_len]
                      image_hidden_states,  # shape [batch_size x image_hidden_dim]
                      max_length,
                      **model_kwargs
                      ) -> torch.LongTensor:  # shape [batch_size x longest_generated_sequence_length]
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)

        # keep track of which sequences are already finished
        # a 1 denotes that a sentence in a batch is unfinished, a 0 denotes that a sentence has finished
        # finished sentences are padded until all sentences in the batch are finished
        unfinished_sequences = torch.ones(size=(batch_size,), dtype=torch.int64, device=self.device)
        cur_len = seq_len

        while True:
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            lm_logits, presents = self.forward(**model_inputs, image_hidden_states=image_hidden_states, return_loss=False)

            next_token_logits = lm_logits[:, -1, :]  # of shape [batch_size x vocab_size]

            # no need to convert logits into probabilities first (via softmax), argmax can be directly applied to logits
            next_tokens = torch.argmax(next_token_logits, dim=-1)  # of shape [batch_size]

            # convert next token to padding token if given sentence has already finished (denoted by a 0 in unfinished_sequences)
            # padding tokens are ignored when decoding, if skip_special_tokens=True is set
            next_tokens = next_tokens * unfinished_sequences + self.pad_token_id * (1 - unfinished_sequences)

            # update variables for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(presents, model_kwargs)
            cur_len += 1

            # if eos_token was found in one sentence, set sentence to finished (by converting 1 to 0 for that sentence)
            binary_mask = (next_tokens != self.eos_token_id).long()
            unfinished_sequences = unfinished_sequences.mul(binary_mask)

            # stop when all sentences are finished (i.e. all sentences have value 0 in unfinished_sequences),
            # or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or (max_length and cur_len >= max_length):
                break

        return input_ids


def print_model_summary(batch_size, seq_len, verbose):
    """
    Choose between:
        verbose = 0 (only model params)
        verbose = 1 (model params and output shape of batch)
        verbose = 2 (model params and output shape of batch, more detailed)
    """
    inputs = {}
    inputs["input_ids"] = torch.randint(low=0, high=50257, size=(batch_size, seq_len))
    inputs["attention_mask"] = torch.randint(low=0, high=2, size=(batch_size, seq_len))
    inputs["image_hidden_states"] = torch.rand(batch_size, (2048 * 8 * 8))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LanguageModel()
    model.to(device, non_blocking=True)

    inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

    if verbose == 0:
        summary(model)
    else:
        summary(model, input_data=dict(inputs), verbose=verbose)
