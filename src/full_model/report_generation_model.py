from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.object_detector.object_detector import ObjectDetector
from src.decoder.gpt2 import DecoderModel


class ReportGenerationModel(nn.Module):
    """
    Full model consisting of object detector encoder, multi-label binary classifier and language model decoder.
    """
    def __init__(self):
        super().__init__()
        self.object_detector = ObjectDetector(return_feature_vectors=True)
        path_to_best_object_detector_weights = "..."
        self.object_detector.load_state_dict(torch.load(path_to_best_object_detector_weights))

        # TODO: implement binary classifier
        # self.binary_classifier = BinaryClassifier()

        self.language_model = DecoderModel()
        path_to_best_detector_weights = "..."
        self.language_model.load_state_dict(torch.load(path_to_best_detector_weights))

    def forward(self,
                images: torch.FloatTensor,  # images is of shape [batch_size, 1, 224, 224] (whole gray-scale images of size 224 x 224)
                input_ids: torch.LongTensor,  # shape [batch_size x 36 x seq_len], 1 sentence for every region for every image (sentence can be empty, i.e. "")
                attention_mask: torch.FloatTensor,  # shape [batch_size x 36 x seq_len]
                region_targets: torch.BoolTensor,  # shape [batch_size x 36], boolean mask that indicates if a region has a sentence or not
                return_loss: bool = True,
                past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                position_ids: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = False
                ):
        # region_features of shape [batch_size, 36, 1024] (i.e. 1 feature vector for every region for every image in batch)
        obj_detector_loss_dict, detections, region_features = self.object_detector(images)

        binary_classifier_loss = self.binary_classifier(region_features, region_targets, return_loss)

        # during training, we train the decoder only on region features whose corresponding sentences are non-empty
        # this is done under the assumption that at test time, the binary classifier will do an adequate job at
        # filtering out those regions by itself
        non_empty_input_ids, non_empty_attention_mask, filtered_region_features = self.filter_out_empty_sentences(region_targets, input_ids, attention_mask, region_features)

        language_model_loss = self.language_model(
            non_empty_input_ids,
            non_empty_attention_mask,
            filtered_region_features,
            return_loss,
            past_key_values,
            position_ids,
            use_cache
        )

        return obj_detector_loss_dict, binary_classifier_loss, language_model_loss, detections

    def filter_out_empty_sentences(self,
                                   region_mask,  # shape [batch_size x 36]
                                   input_ids,  # shape [batch_size x 36 x seq_len]
                                   attention_mask,  # shape [batch_size x 36 x seq_len]
                                   region_features):  # shape [batch_size x 36 x 1024]
        """
        Filters out region features (and input_ids/attention_mask) whose corresponding sentences are non-empty.

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
        non_empty_input_ids = input_ids[region_mask]  # of shape [num_non_empty_sentences_in_batch x seq_len]
        non_empty_attention_mask = attention_mask[region_mask]  # of shape [num_non_empty_sentences_in_batch x seq_len]
        filtered_region_features = region_features[region_mask]  # of shape [num_non_empty_sentences_in_batch x 1024]

        return non_empty_input_ids, non_empty_attention_mask, filtered_region_features

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
        # region_features of shape [batch_size=1, 36, 1024]
        _, detections, region_features = self.object_detector(images)

        # binary_classifier_filtered_region_features of shape [num_regions_selected_in_image, 1024]
        binary_classifier_filtered_region_features, regions_selected_for_sentence_generation = self.binary_classifier(region_features, return_loss=False)

        # output_ids of shape (batch_size x longest_generated_sequence_length)
        output_ids = self.language_model.generate(
            binary_classifier_filtered_region_features,
            max_length,
            num_beams,
            num_beam_groups,
            do_sample,
            num_return_sequences,
            early_stopping)

        return output_ids, regions_selected_for_sentence_generation, detections
