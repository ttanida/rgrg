import torch


class CustomCollator:
    def __init__(self, tokenizer, is_val_or_test, pretrain_without_lm_model):
        self.tokenizer = tokenizer
        self.is_val_or_test = is_val_or_test
        self.pretrain_without_lm_model = pretrain_without_lm_model

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

        For the val and test datasets, we have the additional key:
          - bbox_phrases
          - reference_report
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
        bbox_phrase_exists_size = batch[0]["bbox_phrase_exists"].size()  # should be torch.Size([29])
        region_has_sentence = torch.empty(size=(len(batch), *bbox_phrase_exists_size), dtype=torch.bool)

        # allocate an empty tensor region_is_abnormal that will store all bbox_is_abnormal tensors of the batch
        bbox_is_abnormal_size = batch[0]["bbox_is_abnormal"].size()  # should be torch.Size([29])
        region_is_abnormal = torch.empty(size=(len(batch), *bbox_is_abnormal_size), dtype=torch.bool)

        if self.is_val_or_test and not self.pretrain_without_lm_model:
            # for a validation and test batch, create a List[List[str]] that hold the reference phrases (i.e. bbox_phrases) to compute e.g. BLEU scores
            # the inner list will hold all reference phrases for a single image
            bbox_phrases_batch = []

            # also create a List[str] to hold the reference reports for the images in the batch
            reference_reports = []

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

            if self.is_val_or_test and not self.pretrain_without_lm_model:
                # remove list bbox_phrases from batch and store it in the list bbox_phrases_batch
                bbox_phrases_batch.append(sample_dict.pop("bbox_phrases"))

                # same for reference_report
                reference_reports.append(sample_dict.pop("reference_report"))

        if self.pretrain_without_lm_model:
            batch = {}
        else:
            # batch is now a list that only contains dicts with keys input_ids and attention_mask (both of which are List[List[int]])
            # i.e. batch is of type List[Dict[str, List[List[int]]]]
            # each dict specifies the input_ids and attention_mask of a single image, thus the outer lists always has 29 elements (with each element being a list)
            # for sentences describing 29 regions
            # we want to pad all input_ids and attention_mask to the max sequence length in the batch
            # we can use the pad method of the tokenizer for this, however it requires the input to be of type Dict[str, List[List[int]]
            # thus we first transform the batch into a dict with keys "input_ids" and "attention_mask", both of which are List[List[int]]
            # that hold the input_ids and attention_mask of all the regions in the batch (i.e. the outer list will have (batch_size * 29) elements)
            dict_with_ii_and_am = self.transform_to_dict_with_inputs_ids_and_attention_masks(batch)

            # we can now apply the pad method, which will pad the input_ids and attention_mask to the longest sequence in the batch
            # the keys "input_ids" and "attention_mask" in dict_with_ii_and_am will each map to a tensor of shape [(batch_size * 29), (longest) seq_len (in batch)]
            dict_with_ii_and_am = self.tokenizer.pad(dict_with_ii_and_am, padding="longest", return_tensors="pt")

            # treat dict_with_ii_and_am as the batch variable now (since it is a dict, and we can use it to store all the other keys as well)
            batch = dict_with_ii_and_am

        # add the remaining keys and values to the batch dict
        batch["images"] = images_batch
        batch["image_targets"] = image_targets
        batch["region_has_sentence"] = region_has_sentence
        batch["region_is_abnormal"] = region_is_abnormal

        if self.is_val_or_test and not self.pretrain_without_lm_model:
            batch["reference_sentences"] = bbox_phrases_batch
            batch["reference_reports"] = reference_reports

        return batch

    def transform_to_dict_with_inputs_ids_and_attention_masks(self, batch):
        dict_with_ii_and_am = {"input_ids": [], "attention_mask": []}
        for single_dict in batch:
            for key, outer_list in single_dict.items():
                for inner_list in outer_list:
                    dict_with_ii_and_am[key].append(inner_list)

        return dict_with_ii_and_am
