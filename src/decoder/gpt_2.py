from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
import torch.nn as nn
from torchinfo import summary

# # model_path = "/u/home/tanida/gpt-2-pubmed-medium"
# # tokenizer = AutoTokenizer.from_pretrained(model_path)

# checkpoint = "stanford-crfm/pubmed_gpt"
checkpoint = "healx/gpt-2-pubmed-medium"
tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)

# setting `pad_token_id` to `eos_token_id`:50256 for open-end generation
tokenizer.pad_token = tokenizer.eos_token

# raw_inputs = [
#     "I've been waiting for a HuggingFace course my whole life.",
#     "You hate this so much!",
#     ""
# ]

# sequence = "I've been waiting for a HuggingFace course my whole life mediastinum"
sequence = "This is a cat"
input_ids = tokenizer(sequence, padding="longest", truncation=True, max_length=1024, return_tensors="pt")['input_ids']
print(input_ids)

sequence = "this is a Cat"
input_ids = tokenizer(sequence, padding="longest", truncation=True, max_length=1024, return_tensors="pt")['input_ids']
print(input_ids)

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

