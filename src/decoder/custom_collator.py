import torch


class CustomCollatorWithPadding:
    def __init__(self, tokenizer, padding):
        self.tokenizer = tokenizer
        self.padding = padding

    def __call__(self, batch: list[dict[str]]):
        # discard samples from batch where __getitem__ from custom_image_word_dataset failed (i.e. returned None)
        # otherwise, whole training loop would stop
        batch = list(filter(lambda x: x is not None, batch))  # filter out samples that are None

        image_hidden_dim = batch[0]["image_hidden_states"].size(0)

        # initiate a image_hidden_states_batch tensor that will store all image_hidden_states of the batch
        image_hidden_states_batch = torch.empty(size=(len(batch), image_hidden_dim))

        for i, sample in enumerate(batch):
            # remove image_hidden_states vectors from batch and store them in dedicated image_hidden_states_batch tensor
            image_hidden_states_batch[i] = sample.pop("image_hidden_states")

        # batch only contains samples with input_ids and attention_mask keys
        # the tokenizer will turn the batch variable into a single dict with input_ids and attention_mask keys,
        # that map to tensors of shape [batch_size x (longest) seq_len (in batch)] respectively
        batch = self.tokenizer.pad(batch, padding=self.padding, return_tensors="pt")

        # add the image_hidden_states_batch tensor to the dict
        batch["image_hidden_states"] = image_hidden_states_batch

        return batch
