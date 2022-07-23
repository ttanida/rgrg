from typing import Optional

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchinfo import summary
import torchxrayvision as xrv


class ObjectDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # 36 classes for 36 anatomical regions + background (defined as class 0)
        self.num_classes = 37

        # use only the feature extractor of the pre-trained classifcation model
        self.backbone = xrv.models.DenseNet(weights="densenet121-res224-all").features

        # FasterRCNN needs to know the number of output channels of the backbone
        # for densenet121, it's 1024 (with feature maps of size 7x7)
        self.backbone.out_channels = 1024

        # since we have 36 anatomical regions of varying sizes and aspect ratios,
        # we have to define a custom anchor generator that generates anchors that suit
        # e.g. the spine (aspect ratio ~= 8.0) or the abdomen (aspect ratio ~= 0.6)

        # TODO: run anchor optimization to find suitable hyperparameters for anchor generator
        # https://www.mathworks.com/help/vision/ug/estimate-anchor-boxes-from-training-data.html
        # https://github.com/martinzlocha/anchor-optimization
        # https://towardsdatascience.com/anchor-boxes-the-key-to-quality-object-detection-ddf9d612d4f9

        # since the input image size is 224 x 224, we choose the sizes accordingly
        self.anchor_generator = AnchorGenerator(
            sizes=((10, 20, 30, 40, 50, 60, 70, 80, 90, 150),),
            aspect_ratios=((0.2, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.3, 1.5, 2.1, 2.6, 3.0, 8.0),)
        )

        # let's define the roi pooling layer
        # if the backbone returns a Tensor, featmap_names is expected to be [0]
        # (uniform) size of feature maps after roi pooling layer is defined in output_size
        self.roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0"],
            output_size=7,
            sampling_ratio=2
        )

        # put the pieces together inside a FasterRCNN model
        self.model = FasterRCNN(
            backbone=self.backbone,
            num_classes=self.num_classes,
            rpn_anchor_generator=self.anchor_generator,
            box_roi_pool=self.roi_pooler
        )

    def forward(self,
                images: list[torch.Tensor],  # single image is of shape (1 x 224 x 224) (gray-scale images of size 224 x 224)
                targets: Optional[list[dict[str, torch.Tensor]]] = None):  # single target is a dict containing "boxes" and "labels" keys
        return self.model.forward(images, targets)


model = ObjectDetector()
print(model)
# summary(model, input_size=(64, 1, 224, 224))
