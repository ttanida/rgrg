import torch


class CustomCollatorWithPadding:
    def __init__(self, tokenizer, padding, is_val, has_is_abnormal_column=False):
        self.tokenizer = tokenizer
        self.padding = padding
        self.is_val = is_val
        self.has_is_abnormal_column = has_is_abnormal_column

    def __call__(self, batch: list[dict[str]]):
        # discard samples from batch where __getitem__ from custom_image_word_dataset failed (i.e. returned None)
        # otherwise, whole training loop would stop
        batch = list(filter(lambda x: x is not None, batch))  # filter out samples that are None

        image_shape = batch[0]["image"].size()

        # allocate an empty images_batch tensor that will store all images of the batch
        images_batch = torch.empty(size=(len(batch), image_shape))

        if self.is_val:
            # for a validation batch, create a list of list of str that hold the reference phrases to compute BLEU/BERTscores
            reference_phrases_batch = []

        # it's possible that the validation set has an additional column called "is_abnormal", that contains boolean variables
        # that indicate if a region is described as abnormal or not
        if self.has_is_abnormal_column:
            is_abnormal_list = []

        for i, sample in enumerate(batch):
            # remove image tensors from batch and store them in dedicated images_batch tensor
            images_batch[i] = sample.pop("image")

            if self.is_val:
                reference_phrases_batch.append([sample.pop("reference_phrase")])
            if self.has_is_abnormal_column:
                is_abnormal_list.append(sample.pop("is_abnormal"))

        # batch now only contains samples with input_ids and attention_mask keys
        # the tokenizer will turn the batch variable into a single dict with input_ids and attention_mask keys,
        # that map to tensors of shape [batch_size x (longest) seq_len (in batch)] respectively
        batch = self.tokenizer.pad(batch, padding=self.padding, return_tensors="pt")

        # add the images_batch to the dict
        batch["images"] = images_batch

        # add the reference phrases to the dict for a validation batch
        if self.is_val:
            batch["reference_phrases"] = reference_phrases_batch

        # add the list with the boolean variables to the validation batch
        if self.has_is_abnormal_column:
            batch["is_abnormal_list"] = is_abnormal_list

        return batch
