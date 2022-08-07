import torch.nn as nn


class BinaryClassifierRegionAbnormal(nn.Module):
    """
    Classifier to determine if a region is abnormal or not.
    This is done as to encode this information more explicitly in the region feature vectors that are passed into the decoder.
    This may help with generating better sentences for abnormal regions (which are the minority class).

    This classifier is only applied during training and evalution, but not during inference.
    """
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
        region_is_abnormal  # ground truth boolean tensor of shape [batch_size x 36], indicates if a region is abnormal (True) or not (False)
    ):
        # logits of shape [batch_size x 36]
        logits = self.classifier(top_region_features).squeeze(dim=-1)

        # only compute loss for logits that correspond to a class that was detected
        detected_logits = logits[class_detected]
        detected_region_has_sentence = region_is_abnormal[class_detected]
        loss = self.loss_fn(detected_logits, detected_region_has_sentence)

        if self.training:
            return loss
        else:
            # for evaluation, we also need the regions that were predicted to be abnormal/normal to compare with the ground truth (region_is_abnormal)
            # and compute recall, precision etc.

            # use a threshold of 0 in logit-space (i.e. 0.5 in probability-space)
            # if a logit > 0, then it means that class/region has boolean value True and is considered abnormal
            predicted_abnormal_regions = logits > 0

            # regions that were not detected will be filtered out later (via class_detected) when computing recall, precision etc.

            return loss, predicted_abnormal_regions
