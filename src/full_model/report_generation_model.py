from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn

from binary_classifier.binary_classifier import BinaryClassifier
from src.object_detector.object_detector import ObjectDetector
from src.decoder.gpt2 import DecoderModel


class ReportGenerationModel(nn.Module):
    """
    Full model consisting of object detector encoder, binary classifier and language model decoder.
    """
    def __init__(self):
        super().__init__()
        self.object_detector = ObjectDetector(return_feature_vectors=True)
        path_to_best_object_detector_weights = "..."
        self.object_detector.load_state_dict(torch.load(path_to_best_object_detector_weights))

        self.binary_classifier = BinaryClassifier()

        self.language_model = DecoderModel()
        path_to_best_detector_weights = "..."
        self.language_model.load_state_dict(torch.load(path_to_best_detector_weights))

    def forward(self,
                images: torch.FloatTensor,  # images is of shape [batch_size, 1, 224, 224] (whole gray-scale images of size 224 x 224)
                image_targets: List[Dict],  # contains a dict for every image with keys "boxes" and "labels"
                input_ids: torch.LongTensor,  # shape [batch_size x 36 x seq_len], 1 sentence for every region for every image (sentence can be empty, i.e. "")
                attention_mask: torch.FloatTensor,  # shape [batch_size x 36 x seq_len]
                region_targets: torch.BoolTensor,  # shape [batch_size x 36], boolean mask that indicates if a region has a sentence or not
                return_loss: bool = True,
                past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                position_ids: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = False
                ):
        # top_region_features of shape [batch_size, 36, 1024] (i.e. 1 feature vector for every region for every image in batch)
        # class_predicted is a boolean tensor of shape [batch_size, 36]. Its value is True for a class if the object detector detected the class/region in the image
        if self.training:
            obj_detector_loss_dict, top_region_features, class_predicted = self.object_detector(images, image_targets)
        else:
            obj_detector_loss_dict, detections, top_region_features, class_predicted = self.object_detector(images, image_targets)

        binary_classifier_loss = self.binary_classifier(top_region_features, class_predicted, return_loss, region_targets)

        # during training, we train the decoder only on region features whose corresponding sentences are non-empty
        # this is done under the assumption that at test time, the binary classifier will do an adequate job at
        # filtering out those regions by itself
        # we also filter out region features (and corresponding inputs_ids/attention_masks) that correspond to region/classes that were not detected by the object detector
        valid_input_ids, valid_attention_mask, valid_region_features = self.get_valid_decoder_input(class_predicted, region_targets, input_ids, attention_mask, top_region_features)

        language_model_loss = self.language_model(
            valid_input_ids,
            valid_attention_mask,
            valid_region_features,
            return_loss,
            past_key_values,
            position_ids,
            use_cache
        )

        if self.training:
            return obj_detector_loss_dict, binary_classifier_loss, language_model_loss
        else:
            # detections and class_predicted needed to compute IoU of object detector during evaluation
            return obj_detector_loss_dict, binary_classifier_loss, language_model_loss, detections, class_predicted

    def get_valid_decoder_input(self,
                                class_predicted,  # shape [batch_size x 36]
                                region_mask,  # shape [batch_size x 36]
                                input_ids,  # shape [batch_size x 36 x seq_len]
                                attention_mask,  # shape [batch_size x 36 x seq_len]
                                region_features):  # shape [batch_size x 36 x 1024]
        """
        Filters out region features (and input_ids/attention_mask) whose corresponding sentences are non-empty or that were not detected by the object detector.

        Example:
            Let's assume region_mask has shape [batch_size x 36] with batch_size = 2, so shape [2 x 36].
            This means we have boolean values for all 36 regions of the 2 images in the batch, that indicate if the
            regions have a corresponding sentence in the reference report os not.

            Now, let's assume region_mask is True for the first 3 regions of each image. This means only the first
            3 regions of each image are described with sentences in the reference report.

            input_ids has shape [batch_size x 36 x seq_len].

            If we run non_empty_input_ids = input_ids[region_mask], then we get non_empty_input_ids of shape [6 x seq_len].
            We thus get the first 3 rows of the first image, and the first 3 rows of the second image concatenated
            into 1 matrix.
        """
        valid = torch.logical_and(class_predicted, region_mask)

        valid_input_ids = input_ids[valid]  # of shape [valid_num_non_empty_sentences_in_batch x seq_len]
        valid_attention_mask = attention_mask[valid]  # of shape [valid_num_non_empty_sentences_in_batch x seq_len]
        valid_region_features = region_features[valid]  # of shape [valid_num_non_empty_sentences_in_batch x 1024]

        return valid_input_ids, valid_attention_mask, valid_region_features

    @torch.no_grad()
    def generate(self,
                 images: torch.FloatTensor,  # images is of shape [batch_size, 1, 224, 224] (whole gray-scale images of size 224 x 224)
                 max_length: int = None,
                 num_beams: int = 1,
                 num_beam_groups: int = 1,
                 do_sample: bool = False,
                 num_return_sequences: int = 1,
                 early_stopping: bool = False
        ):
        """
        In inference mode, we input 1 image (with 36 regions) at a time.

        The object detector first find the region features for all 36 regions.

        The binary classifier takes the region_features of shape [batch_size=1, 36, 1024] and returns:
            - binary_classifier_filtered_region_features: shape [num_regions_selected_in_image, 1024],
            all region_features which were selected by the classifier to get a sentence generated

            - regions_selected_for_sentence_generation: shape [36], boolean array that indicates which regions
            were selected to get a sentences generated. This is needed in case we want to find the corresponding
            reference sentences to compute scores for metrics such as BertScore or BLEU.

        The decoder then takes these selected region features and generates output ids for the batch.
        These output ids can then be decoded by the tokenizer to get the generated sentences.
        """
        # top_region_features of shape [batch_size=1, 36, 1024]
        _, detections, top_region_features, class_predicted = self.object_detector(images)

        # TODO: continue working here
        # filtered_region_features of shape [num_regions_selected_in_image, 1024]
        filtered_region_features, regions_selected_for_sentence_generation = self.binary_classifier(top_region_features, class_predicted.squeeze(), return_loss=False)

        # output_ids of shape (batch_size x longest_generated_sequence_length)
        output_ids = self.language_model.generate(
            filtered_region_features,
            max_length,
            num_beams,
            num_beam_groups,
            do_sample,
            num_return_sequences,
            early_stopping)

        return output_ids, regions_selected_for_sentence_generation, detections
