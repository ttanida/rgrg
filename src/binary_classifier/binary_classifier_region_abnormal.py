import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

        # since we have around 6.0x more normal regions than abnormal regions (see dataset/dataset_stats.txt generated from compute_stats_dataset.py),
        # we set pos_weight=6.0 to put 6.0 more weight on the loss of abnormal regions
        pos_weight = torch.tensor([6.0], device=device)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(
        self,
        top_region_features,  # tensor of shape [batch_size x 29 x 1024]
        class_detected,  # boolean tensor of shape [batch_size x 29], indicates if the object detector has detected the region/class or not
        region_is_abnormal  # ground truth boolean tensor of shape [batch_size x 29], indicates if a region is abnormal (True) or not (False)
    ):
        # logits of shape [batch_size x 29]
        logits = self.classifier(top_region_features).squeeze(dim=-1)

        # only compute loss for logits that correspond to a class that was detected
        detected_logits = logits[class_detected]
        detected_region_is_abnormal = region_is_abnormal[class_detected]

        loss = self.loss_fn(detected_logits, detected_region_is_abnormal.type(torch.float32))

        if self.training:
            return loss
        else:
            # for evaluation, we also need the regions that were predicted to be abnormal/normal to compare with the ground truth (region_is_abnormal)
            # and compute recall, precision etc.

            # use a threshold of -1 in logit-space (i.e. 0.269 in probability-space)
            # if a logit > -1, then it means that class/region has boolean value True and is considered abnormal
            predicted_abnormal_regions = logits > -1

            # regions that were not detected will be filtered out later (via class_detected) when computing recall, precision etc.
            return loss, predicted_abnormal_regions
