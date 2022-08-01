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
        class_predicted,  # boolean tensor of shape [batch_size x 36], indicates if the object detector has predicted/detected the region/class or not
        return_loss,  # boolean value
        region_targets=None  # boolean tensor of shape [batch_size x 36], indicates if a region has a sentence (True) or not (False)
    ):
        # logits of shape [batch_size x 36]
        logits = self.classifier(top_region_features).squeeze()

        if return_loss:
            # only compute loss for logits that correspond to a predicted class
            detected_logits = logits[class_predicted]
            detected_region_targets = region_targets[class_predicted]

            loss = self.loss_fn(detected_logits, detected_region_targets)
            return loss
        else:
            # use a threshold of 0 in logit-space (i.e. 0.5 in probability-space)
            # if a logit > 0, then it means that class has boolean value True and a sentence should be generated for it
            preds = logits > 0

            # but set to False all classes that were not detected by object detector
            preds[~class_predicted] = False
            return preds
