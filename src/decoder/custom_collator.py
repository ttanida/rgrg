import torch


class CustomCollatorWithPadding:
    def __init__(self, tokenizer, padding, is_val, has_finding_exists_column=False):
        self.tokenizer = tokenizer
        self.padding = padding
        self.is_val = is_val
        self.has_finding_exists_column = has_finding_exists_column

    def __call__(self, batch: list[dict[str]]):
        # discard samples from batch where __getitem__ from custom_image_word_dataset failed (i.e. returned None)
        # otherwise, whole training loop would stop
        batch = list(filter(lambda x: x is not None, batch))  # filter out samples that are None

        image_hidden_dim = batch[0]["image_hidden_states"].size(0)

        # initiate a image_hidden_states_batch tensor that will store all image_hidden_states of the batch
        image_hidden_states_batch = torch.empty(size=(len(batch), image_hidden_dim))

        if self.is_val:
            # for a validation batch, create a list of list of str that hold the reference phrases to compute BLEU/BERTscores
            reference_phrases_batch = []

        for i, sample in enumerate(batch):
            # remove image_hidden_states vectors from batch and store them in dedicated image_hidden_states_batch tensor
            image_hidden_states_batch[i] = sample.pop("image_hidden_states")

            if self.is_val:
                reference_phrases_batch.append([sample.pop("reference_phrase")])

        # batch now only contains samples with input_ids and attention_mask keys
        # the tokenizer will turn the batch variable into a single dict with input_ids and attention_mask keys,
        # that map to tensors of shape [batch_size x (longest) seq_len (in batch)] respectively
        batch = self.tokenizer.pad(batch, padding=self.padding, return_tensors="pt")

        # add the image_hidden_states_batch tensor to the dict
        batch["image_hidden_states"] = image_hidden_states_batch

        # add the reference phrases to the dict for a validation batch
        if self.is_val:
            batch["reference_phrases"] = reference_phrases_batch

        return batch
