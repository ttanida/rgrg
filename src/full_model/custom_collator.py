import torch


class CustomCollator:
    def __init__(self, tokenizer, padding, is_val):
        self.tokenizer = tokenizer
        self.padding = padding
        self.is_val = is_val

    def __call__(self, batch: list[dict[str]]):
        """
        batch is a list of dicts where each dict corresponds to a single image and has the keys:
          - image
          - bbox_coordinates
          - bbox_labels
          - input_ids
          - attention_mask
          - bbox_phrase_exists
          - bbox_is_abnormal

        For the val dataset, we have the additional key:
          - bbox_phrases
        """
        # discard samples from batch where __getitem__ from custom_dataset failed (i.e. returned None)
        # otherwise, whole training loop would stop
        batch = list(filter(lambda x: x is not None, batch))  # filter out samples that are None

        # allocate an empty tensor images_batch that will store all images of the batch
        image_size = batch[0]["image"].size()
        images_batch = torch.empty(size=(len(batch), *image_size))

        # create an empty list image_targets that will store dicts containing the bbox_coordinates and bbox_labels
        image_targets = []

        # allocate an empty tensor region_has_sentence that will store all bbox_phrase_exists tensors of the batch
        bbox_phrase_exists_size = batch[0]["bbox_phrase_exists"].size()  # should be torch.Size([36])
        region_has_sentence = torch.empty(size=(len(batch), *bbox_phrase_exists_size))

        # allocate an empty tensor region_is_abnormal that will store all bbox_is_abnormal tensors of the batch
        bbox_is_abnormal_size = batch[0]["bbox_is_abnormal"].size()  # should be torch.Size([36])
        region_is_abnormal = torch.empty(size=(len(batch), *bbox_is_abnormal_size))

        if self.is_val:
            # for a validation batch, create a list of list of str that hold the reference phrases (i.e. bbox_phrases) to compute BLEU/BERTscores
            # the inner list will hold all reference phrases for a single image
            bbox_phrases_batch = []

        for i, sample_dict in enumerate(batch):
            # remove image tensors from batch and store them in dedicated images_batch tensor
            images_batch[i] = sample_dict.pop("image")

            # remove bbox_coordinates and bbox_labels and store them in list image_targets
            boxes = sample_dict.pop("bbox_coordinates")
            labels = sample_dict.pop("bbox_labels")
            image_targets.append({"boxes": boxes, "labels": labels})

            # remove bbox_phrase_exists tensors from batch and store them in dedicated region_has_sentence tensor
            region_has_sentence[i] = sample_dict.pop("bbox_phrase_exists")

            # remove bbox_is_abnormal tensors from batch and store them in dedicated region_is_abnormal tensor
            region_is_abnormal[i] = sample_dict.pop("bbox_is_abnormal")

            if self.is_val:
                # remove list bbox_phrases from batch and store it in the list bbox_phrases_batch
                bbox_phrases_batch.append(sample_dict.pop("bbox_phrases"))

        # batch is now a list that only contains dicts with keys input_ids and attention_mask (both of which are List[List[int]])









        if self.is_val:
            # for a validation batch, create a list of list of str that hold the reference phrases (i.e. bbox_phrases) to compute BLEU/BERTscores
            bbox_phrases_batch = []

        # it's possible that the validation set has an additional column called "is_abnormal", that contains boolean variables
        # that indicate if a region is described as abnormal or not
        if self.has_is_abnormal_column:
            is_abnormal_list = []

        for i, sample in enumerate(batch):
            # remove image_hidden_states vectors from batch and store them in dedicated image_hidden_states_batch tensor
            image_hidden_states_batch[i] = sample.pop("image_hidden_states")

            if self.is_val:
                bbox_phrases_batch.append([sample.pop("reference_phrase")])
            if self.has_is_abnormal_column:
                is_abnormal_list.append(sample.pop("is_abnormal"))

        # batch now only contains samples with input_ids and attention_mask keys
        # the tokenizer will turn the batch variable into a single dict with input_ids and attention_mask keys,
        # that map to tensors of shape [batch_size x (longest) seq_len (in batch)] respectively
        batch = self.tokenizer.pad(batch, padding=self.padding, return_tensors="pt")

        # add the image_hidden_states_batch tensor to the dict
        batch["image_hidden_states"] = image_hidden_states_batch

        # add the reference phrases to the dict for a validation batch
        if self.is_val:
            batch["reference_phrases"] = bbox_phrases_batch

        # add the list with the boolean variables to the validation batch
        if self.has_is_abnormal_column:
            batch["is_abnormal_list"] = is_abnormal_list

        return batch
