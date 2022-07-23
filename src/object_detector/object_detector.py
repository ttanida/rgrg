from typing import Optional, List, Dict

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchinfo import summary
import torchxrayvision as xrv


class ObjectDetector(nn.Module):
    """
    Implements Faster R-CNN with a classifier pre-trained on chest x-rays as the backbone.
    The implementation differs slightly from the PyTorch one.

    During training, the model expects both the input image tensor as well as the targets.

    The input image tensor is expected to be a tensor of shape [batch_size, 1, H, W], with H = W (which will most likely be 224 or 512).
    This differs form the PyTorch implementation, where the input images are expected to be a list of tensors (of different shapes).
    We apply transformations before inputting the images into the model, whereas the PyTorch implementation applies transformations
    after the images were inputted into the model.

    The targets is expected to be a list of dictionaries, with each dict containing:
        - boxes (FloatTensor[N, 4]): the gt boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification and regression losses for both the RPN and the R-CNN.

    During inference, the model requires only the input tensors, and returns the post-processed predictions as a List[Dict[Tensor]],
    one for each input image. The fields of the Dict are as follows:
        - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores or each prediction
    """

    def __init__(self):
        super().__init__()
        # 36 classes for 36 anatomical regions + background (defined as class 0)
        self.num_classes = 37

        # use only the feature extractor of the pre-trained classifcation model
        self.backbone = xrv.models.DenseNet(weights="densenet121-res224-all").features

        # FasterRCNN needs to know the number of output channels of the backbone
        # for densenet121, it's 1024 (with feature maps of size 7x7)
        self.backbone.out_channels = 1024

        self.rpn = self._create_rpn()
        self.roi_heads = self._create_roi_heads()

        # put the pieces together inside a FasterRCNN model
        self.model = FasterRCNN(
            backbone=self.backbone,
            num_classes=self.num_classes,
            rpn_anchor_generator=self.anchor_generator,
            box_roi_pool=self.roi_pooler
        )

    def _create_rpn(self):
        # since we have 36 anatomical regions of varying sizes and aspect ratios,
        # we have to define a custom anchor generator that generates anchors that suit
        # e.g. the spine (aspect ratio ~= 8.0) or the abdomen (aspect ratio ~= 0.6)

        # TODO: run anchor optimization to find suitable hyperparameters for anchor generator
        # https://www.mathworks.com/help/vision/ug/estimate-anchor-boxes-from-training-data.html
        # https://github.com/martinzlocha/anchor-optimization
        # https://towardsdatascience.com/anchor-boxes-the-key-to-quality-object-detection-ddf9d612d4f9

        # since the input image size is 224 x 224, we choose the sizes accordingly
        anchor_generator = AnchorGenerator(
            sizes=((10, 20, 30, 40, 50, 60, 70, 80, 90, 150),),
            aspect_ratios=((0.2, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.3, 1.5, 2.1, 2.6, 3.0, 8.0),)
        )

        rpn_head = RPNHead(self.backbone.out_channels, anchor_generator.num_anchors_per_location())

        # use default values for the RPN
        rpn = RegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=rpn_head,
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n={"training": 2000, "testing": 1000},
            post_nms_top_n={"training": 2000, "testing": 1000},
            nms_thresh=0.7,
            score_thresh=0.0
        )

        return rpn

    def _create_roi_heads(self):
        # define the roi pooling layer
        # if the backbone returns a Tensor, featmap_names is expected to be [0]
        # (uniform) size of feature maps after roi pooling layer is defined in output_size
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0"],
            output_size=7,
            sampling_ratio=2
        )

        resolution = roi_pooler.output_size[0]
        representation_size = 1024

        box_head = TwoMLPHead(self.backbone.out_channels * resolution ** 2, representation_size)
        box_predictor = FastRCNNPredictor(representation_size, self.num_classes)

        # use default values for RoI heads
        roi_heads = RoIHeads(
            box_roi_pool=roi_pooler,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.25,
            bbox_reg_weights=None,
            score_thresh=0.05,
            nms_thresh=0.5,
            detections_per_img=100
        )

        return roi_heads

    def forward(self,
                images: List[torch.Tensor],  # single image is of shape (1 x 224 x 224) (gray-scale images of size 224 x 224)
                targets: Optional[List[Dict[str, torch.Tensor]]] = None):  # single target is a dict containing "boxes" and "labels" keys
        return self.model.forward(images, targets)


device = torch.device("cpu")
model = ObjectDetector()
model.to(device)
# print(model)
input_data = [torch.rand(1, 300, 400).to(device), torch.rand(1, 500, 400).to(device)]
summary(model, input_data=[input_data], device=device)
