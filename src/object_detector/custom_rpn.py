from typing import Optional, List, Dict, Tuple

from torch import Tensor
import torch.nn as nn
from torchvision.models.detection.rpn import (
    AnchorGenerator,
    RegionProposalNetwork,
    concat_box_prediction_layers,
)

from .image_list import ImageList


class CustomRegionProposalNetwork(RegionProposalNetwork):
    """
    Custom RPN class is almost exact copy of PyTorch implementation:
    https://github.com/pytorch/vision/blob/9b84859e5809c68acc65f94f31b39c265867302d/torchvision/models/detection/rpn.py

    The only difference is that PyTorch's RPN only computes the objectness loss and regression loss in train mode.
    However, to compute the validation, we also need to compute those losses in eval mode.

    Which is why the "if self.training:" condition of line 375 of the PyTorch implementation is taken out in my custom
    implementation.
    """
    def __init__(
        self,
        anchor_generator: AnchorGenerator,
        head: nn.Module,
        # Faster-RCNN Training
        fg_iou_thresh: float,
        bg_iou_thresh: float,
        batch_size_per_image: int,
        positive_fraction: float,
        # Faster-RCNN Inference
        pre_nms_top_n: Dict[str, int],
        post_nms_top_n: Dict[str, int],
        nms_thresh: float,
        score_thresh: float = 0.0,
    ):
        super().__init__(
            anchor_generator,
            head,
            fg_iou_thresh,
            bg_iou_thresh,
            batch_size_per_image,
            positive_fraction,
            pre_nms_top_n,
            post_nms_top_n,
            nms_thresh,
            score_thresh,
        )

    def forward(
        self,
        images: ImageList,
        features: Dict[str, Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[List[Tensor], Dict[str, Tensor]]:

        features = list(features.values())
        objectness, pred_bbox_deltas = self.head(features)
        anchors = self.anchor_generator(images, features)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)

        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        losses = {}
        if targets is not None:
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }

        return boxes, losses
