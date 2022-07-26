from typing import Optional, List, Dict, Tuple

import torch
from torch import Tensor
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss


class CustomRoIHeads(RoIHeads):
    def __init__(
        self,
        return_feature_vectors,
        box_roi_pool,
        box_head,
        box_predictor,
        # Faster R-CNN training
        fg_iou_thresh,
        bg_iou_thresh,
        batch_size_per_image,
        positive_fraction,
        bbox_reg_weights,
        # Faster R-CNN inference
        score_thresh,
        nms_thresh,
        detections_per_img,
        # Mask
        mask_roi_pool=None,
        mask_head=None,
        mask_predictor=None,
        keypoint_roi_pool=None,
        keypoint_head=None,
        keypoint_predictor=None,
    ):
        super().__init__(
            box_roi_pool,
            box_head,
            box_predictor,
            fg_iou_thresh,
            bg_iou_thresh,
            batch_size_per_image,
            positive_fraction,
            bbox_reg_weights,
            score_thresh,
            nms_thresh,
            detections_per_img,
            mask_roi_pool,
            mask_head,
            mask_predictor,
            keypoint_roi_pool,
            keypoint_head,
            keypoint_predictor,
        )
        self.return_feature_vectors = return_feature_vectors

    def forward(
        self,
        features: Dict[str, Tensor],
        proposals: List[Tensor],
        image_shapes: List[Tuple[int, int]],
        targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]:
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                if not t["boxes"].dtype in floating_point_types:
                    raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
                if not t["labels"].dtype == torch.int64:
                    raise TypeError("target labels must of int64 type, instead got {t['labels'].dtype}")

        if targets is not None:
            proposals, _, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        detections: List[Dict[str, torch.Tensor]] = []
        detector_losses = {}

        if labels and regression_targets:
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            detector_losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}

        boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
        num_images = len(boxes)
        for i in range(num_images):
            detections.append(
                {
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                }
            )

        roi_heads_output = {}
        roi_heads_output["detections"] = detections
        roi_heads_output["detector_losses"] = detector_losses

        if self.return_feature_vectors:
            roi_heads_output["box_features"] = box_features

        return roi_heads_output
