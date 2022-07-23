from numpy import size
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

        # let's make the RPN generate 5 x 3 anchors per spatial location
        # (5 different sizes and 3 different aspect ratios)
        self.anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
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


backbone = xrv.models.DenseNet(weights="densenet121-res224-all").features
backbone.out_channels = 1024
print(backbone.out_channels)
# summary(model.features, input_size=(64, 1, 224, 224))
