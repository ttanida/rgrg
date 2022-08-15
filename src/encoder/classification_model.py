import torch.nn as nn
# from torchinfo import summary
import torchxrayvision as xrv


class ClassificationModel(nn.Module):
    """
    Model to classify 36 anatomical regions and determine if they are normal/abnormal.

    Note that all parameters are trainable (even those of feature_extractor), since requires_grad was not set to False explicitly.
    """
    def __init__(self, return_feature_vectors=False):
        super().__init__()
        resnet = xrv.models.ResNet(weights="resnet50-res512-all")

        self.feature_extractor = nn.Sequential(*list(resnet.model.children())[:-2])

        # AdaptiveAvgPool2d to get from [batch_size, 2048, 16, 16] -> [batch_size, 2048, 1, 1]
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        # linear layers for classification
        # 36 classes for 36 anatomical regions
        self.classifier = nn.Sequential(
            nn.Linear(in_features=2048, out_features=512),
            # nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=36)
        )

        # boolean to specify if feature vectors should be returned after avg_pool
        self.return_feature_vectors = return_feature_vectors

    def forward(self, images):  # images is of shape [batch_size, 1, 512, 512] (gray-scale images of size 512 x 512)
        image_features = self.feature_extractor(images)  # image features of shape [batch_size, 2048, 16, 16]
        image_features = self.avg_pool(image_features)  # image features of shape [batch_size, 2048, 1, 1]

        if self.return_feature_vectors:
            return image_features.squeeze()  # remove all dimensions of size 1

        # flatten for linear layers
        image_features = image_features.view(image_features.size(0), -1)
        logits = self.classifier(image_features)

        return logits


# model = ClassificationModel()
# summary(model, input_size=(64, 1, 512, 512))
