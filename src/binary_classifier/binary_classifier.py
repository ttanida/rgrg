import torch.nn as nn


class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1)
        )

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        top_region_features,  # tensor of shape [batch_size x 36 x 1024]
        class_detected,  # boolean tensor of shape [batch_size x 36], indicates if the object detector has detected the region/class or not
        return_pred,  # boolean value that is True if we are in inference mode and want to get the regions that were selected for sentence generation by the classifier
        region_has_sentence=None  # boolean tensor of shape [batch_size x 36], indicates if a region has a sentence (True) or not (False) as the ground truth
    ):
        # logits of shape [batch_size x 36]
        logits = self.classifier(top_region_features).squeeze(dim=-1)

        # train or eval mode
        if not return_pred:
            # only compute loss for logits that correspond to a class that was detected
            detected_logits = logits[class_detected]
            detected_region_has_sentence = region_has_sentence[class_detected]

            loss = self.loss_fn(detected_logits, detected_region_has_sentence)

            # in train mode, only return the train loss
            if self.training:
                return loss
            else:
                # in eval model, return val loss and predictions (i.e. selected_regions)
                # selected_regions necessary to evaluate binary classifier performance
                # by comparing predictions to ground-truth region_has_sentence
                #
                # use a threshold of 0 in logit-space (i.e. 0.5 in probability-space)
                # if a logit > 0, then it means that class/region has boolean value True and a sentence should be generated for it
                # selected_regions is of shape [batch_size x 36] and is True for regions that should get a sentence
                selected_regions = logits > 0

                # set to False all regions that were not detected by object detector
                # (since no detection -> no sentence generation possible)
                selected_regions[~class_detected] = False

                return loss, selected_regions
        else:
            # in inference mode, we need the selected regions and its features by the classifier
            selected_regions = logits > 0

            # set to False all regions that were not detected by object detector
            # (since no detection -> no sentence generation possible)
            selected_regions[~class_detected] = False

            # selected_region_features is of shape [num_regions_selected_in_batch, 1024]
            selected_region_features = top_region_features[selected_regions]

            return selected_region_features, selected_regions
