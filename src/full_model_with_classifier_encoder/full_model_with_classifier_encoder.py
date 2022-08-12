from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.encoder.classification_model import ClassificationModel
from src.language_model.language_model import LanguageModel


class ReportGenerationModel(nn.Module):
    """
    Full model consisting of classifier encoder and decoder.
    """
    def __init__(self):
        super().__init__()
        self.encoder = ClassificationModel(return_feature_vectors=True)
        self.decoder = LanguageModel()

    def forward(self,
                images: torch.FloatTensor,  # images is of shape [batch_size, 1, 512, 512] (gray-scale images of size 512 x 512)
                input_ids: torch.LongTensor,  # shape (batch_size x seq_len)
                attention_mask: torch.FloatTensor,  # shape (batch_size x seq_len)
                return_loss: bool = True,
                past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                position_ids: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = False
                ):
        image_features = self.encoder(images)  # image features of shape [batch_size, 2048]

        loss = self.decoder(
            input_ids,
            attention_mask,
            image_features,
            return_loss,
            past_key_values,
            position_ids,
            use_cache
        )

        return loss

    @torch.no_grad()
    def generate(self,
                 images: torch.FloatTensor,  # images is of shape [batch_size, 1, 512, 512] (gray-scale images of size 512 x 512)
                 max_length: int = None,
                 num_beams: int = 1,
                 num_beam_groups: int = 1,
                 do_sample: bool = False,
                 num_return_sequences: int = 1,
                 early_stopping: bool = False
                 ) -> torch.LongTensor:  # shape (batch_size x longest_generated_sequence_length)
        """
        Generates output ids for a batch of images.
        These output ids can then be decoded by the tokenizer to get the generated sentences.
        """
        image_features = self.encoder(images)  # image features of shape [batch_size, 2048]
        output_ids = self.decoder.generate(
            image_features,
            max_length,
            num_beams,
            num_beam_groups,
            do_sample,
            num_return_sequences,
            early_stopping)

        return output_ids  # shape (batch_size x longest_generated_sequence_length)
