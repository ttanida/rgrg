from collections import OrderedDict
from typing import Optional, List, Dict

import torch
from torch import Tensor
import torch.nn as nn
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
# from torchinfo import summary

from src.object_detector.custom_roi_heads import CustomRoIHeads
from src.object_detector.custom_rpn import CustomRegionProposalNetwork
from src.object_detector.image_list import ImageList


class ObjectDetector(nn.Module):
    """
    Implements Faster R-CNN with a classifier pre-trained on chest x-rays as the backbone.
    The implementation differs slightly from the PyTorch one.

    During training, the model expects both the input image tensor as well as the targets.

    The input image tensor is expected to be a tensor of shape [batch_size, 1, H, W], with H = W (which will most likely be 512).
    This differs form the PyTorch implementation, where the input images are expected to be a list of tensors (of different shapes).
    We apply transformations before inputting the images into the model, whereas the PyTorch implementation applies transformations
    after the images were inputted into the model.

    The targets is expected to be a list of dictionaries, with each dict containing:
        - boxes (FloatTensor[N, 4]): the gt boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The PyTorch implementation returns a Dict[Tensor] containing the 4 losses in train mode, and a List[Dict[Tensor]] containing
    the detections for each image in eval mode.

    My implementation returns different things depending on if the object detector is trained/evaluated in isolation,
    or if it's trained/evaluated as part of the full model.

    Please check the doc string of the forward method for more details.
    """

    def __init__(self, return_feature_vectors=False):
        super().__init__()
        # boolean to specify if feature vectors should be returned after roi pooling inside RoIHeads
        self.return_feature_vectors = return_feature_vectors

        # 29 classes for 29 anatomical regions + background (defined as class 0)
        self.num_classes = 30

        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        # since we have grayscale images, we need to change the first conv layer to accept 1 in_channel (instead of 3)
        resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # use only the feature extractor of the pre-trained classification model
        # (i.e. use all children but the last 2, which are AdaptiveAvgPool2d and Linear)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # FasterRCNN needs to know the number of output channels of the backbone
        # for ResNet-50, it's 2048 (with feature maps of size 16x16)
        self.backbone.out_channels = 2048

        self.rpn = self._create_rpn()
        self.roi_heads = self._create_roi_heads()

    def _create_rpn(self):
        # since we have 29 anatomical regions of varying sizes and aspect ratios,
        # we have to define a custom anchor generator that generates anchors that suit
        # e.g. the spine (aspect ratio ~= 8.0) or the abdomen (aspect ratio ~= 0.6)

        # TODO: run anchor optimization to find suitable hyperparameters for anchor generator
        # https://www.mathworks.com/help/vision/ug/estimate-anchor-boxes-from-training-data.html
        # https://github.com/martinzlocha/anchor-optimization
        # https://towardsdatascience.com/anchor-boxes-the-key-to-quality-object-detection-ddf9d612d4f9

        # since the input image size is 512 x 512, we choose the sizes accordingly
        anchor_generator = AnchorGenerator(
            sizes=((20, 40, 60, 80, 100, 120, 140, 160, 180, 300),),
            aspect_ratios=((0.2, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.3, 1.5, 2.1, 2.6, 3.0, 5.0, 8.0),),
        )

        rpn_head = RPNHead(self.backbone.out_channels, anchor_generator.num_anchors_per_location()[0])

        # use default values for the RPN
        rpn = CustomRegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=rpn_head,
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n={"training": 2000, "testing": 1000},
            post_nms_top_n={"training": 2000, "testing": 1000},
            nms_thresh=0.7,
            score_thresh=0.0,
        )

        return rpn

    def _create_roi_heads(self):
        # define the roi pooling layer
        # if the backbone returns a Tensor, featmap_names is expected to be [0]
        # (uniform) size of feature maps after roi pooling layer is defined in feature_map_output_size
        feature_map_output_size = 8
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0"], output_size=feature_map_output_size, sampling_ratio=2)

        resolution = roi_pooler.output_size[0]
        representation_size = 1024

        box_head = TwoMLPHead(self.backbone.out_channels * resolution**2, representation_size)
        box_predictor = FastRCNNPredictor(representation_size, self.num_classes)

        # use default values for RoI heads
        roi_heads = CustomRoIHeads(
            return_feature_vectors=self.return_feature_vectors,
            feature_map_output_size=feature_map_output_size,
            box_roi_pool=roi_pooler,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.25,
            bbox_reg_weights=None,
            score_thresh=0.01,
            nms_thresh=0.0,
            detections_per_img=100,
        )

        return roi_heads

    def _check_targets(self, targets):
        """
        Check if
            - there are targets for training
            - all bboxes are of correct type and shape
            - there are no degenerate bboxes (where e.g. x1 > x2)
        """
        if targets is None:
            torch._assert(False, "targets should not be none when in training mode")

        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            if not isinstance(boxes, torch.Tensor):
                torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

            torch._assert(
                len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
            )

            # x1 should always be < x2 and y1 should always be < y2
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                # print the first degenerate box
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                torch._assert(
                    False,
                    "All bounding boxes should have positive height and width." f" Found invalid box {degen_bb} for target at index {target_idx}.",
                )

    def _transform_inputs_for_rpn_and_roi(self, images, features):
        """
        Tranform images and features from tensors to types that the rpn and roi_heads expect in the current PyTorch implementation.

        Concretely, images have to be of class ImageList, which is a custom PyTorch class.
        Features have to be a dict where the str "0" maps to the features.

        Args:
            images (Tensor)
            features (Tensor): of shape [batch_size, 2048, 16, 16]

        Returns:
            images (ImageList)
            features (Dict[str, Tensor])
        """
        images = ImageList(images)
        features = OrderedDict([("0", features)])

        return images, features

    def forward(self, images: Tensor, targets: Optional[List[Dict[str, Tensor]]] = None):
        """
        Args:
            images (Tensor): images to be processed of shape [batch_size, 1, 512, 512] (gray-scale images of size 512 x 512)
            targets (List[Dict[str, Tensor]]): list of batch_size dicts, where a single dict contains the fields:
                - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format
                - labels (Int64Tensor[N]): the class label for each ground-truth box

        Returns:
            (1) If object detector is trained/evaluated in isolation, then self.return_feature_vectors should be False and it returns
                (I) in train mode:
                    - losses (Dict[Tensor]), which contains the 4 object detector losses
                (II) in eval mode:
                    - losses (Dict[Tensor]). If targets == None (i.e. during inference), then (val) losses will be an empty dict
                    - detections (List[Dict[str, Tensor]]), which are the predictions for each input image.

            (2) If object detector is trained/evaluated as part of the full model, then self.return_feature_vectors should be True and it returns
                (I) in train mode:
                    - losses
                    - top_region_features (FloatTensor(batch_size, 29, 1024)):
                        - the region visual features with the highest scores for each region and for each image in the batch
                        - these are needed to train the binary classifiers and language model
                    - class_detected (BoolTensor(batch_size, 29)):
                        - boolean is True if a region/class had the highest score (i.e. was top-1) for at least 1 RoI box
                        - if the value is False for any class, then this means the object detector effectively did not detect the region,
                        and it is thus filtered out from the next modules in the full model
                (II) in eval mode:
                    - losses. If targets == None (i.e. during inference), then (val) losses will be an empty dict
                    - detections
                    - top_region_features
                    - class_detected
        """
        if targets is not None:
            self._check_targets(targets)

        features = self.backbone(images)

        images, features = self._transform_inputs_for_rpn_and_roi(images, features)

        proposals, proposal_losses = self.rpn(images, features, targets)
        roi_heads_output = self.roi_heads(features, proposals, images.image_sizes, targets)

        # the roi_heads_output always includes the detector_losses
        detector_losses = roi_heads_output["detector_losses"]

        # they include the detections and class_detected when we are evaluating
        if not self.training:
            detections = roi_heads_output["detections"]
            class_detected = roi_heads_output["class_detected"]

        # they include the top_region_features and class_detected if we train/evaluate the full model
        if self.return_feature_vectors:
            top_region_features = roi_heads_output["top_region_features"]
            class_detected = roi_heads_output["class_detected"]

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        # if we don't return the region features, then we train/evaluate the object detector in isolation (i.e. not as part of the full model)
        if not self.return_feature_vectors:
            if self.training:
                # we only need the losses to train the object detector
                return losses
            else:
                # we need both losses, detections and class_detected to evaluate the object detector
                # losses with be an empty dict if targets == None (i.e. during inference)
                return losses, detections, class_detected

        # if we return region features, then we train/evaluate the full model (with object detector as one part of it)
        if self.return_feature_vectors:
            if self.training:
                # we need the losses to train the object detector, and the top_region_features/class_detected to train the binary classifier and decoder
                return losses, top_region_features, class_detected
            else:
                # we additionally need the detections to evaluate the object detector
                # losses will be an empty dict if targets == None (i.e. during inference)
                return losses, detections, top_region_features, class_detected
