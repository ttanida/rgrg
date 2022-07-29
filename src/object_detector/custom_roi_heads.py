from typing import Optional, List, Dict, Tuple
from matplotlib.cbook import index_of

import torch
from torch import Tensor
import torch.nn.functional as F
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss
from torchvision.ops import boxes as box_ops


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

    def get_top_box_features(
        self,
        box_features,
        class_logits,
        proposals
    ):
        # apply softmax on background class as well
        # (such that if the background class has a high score, all other classes will have a low score)
        pred_scores = F.softmax(class_logits, -1)

        # remove score of the background class
        pred_scores = pred_scores[:, 1:]

        # split pred_scores (which is a tensor with scores for all RoIs of all images in the batch)
        # into the tuple pred_scores_per_image (where 1 pred_score tensor has scores for all RoIs of 1 image)
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_scores_per_image = torch.split(pred_scores, boxes_per_image, dim=0)

        # also split box_features the same way
        box_features_per_image = torch.split(box_features, boxes_per_image, dim=0)

        # collect the tensors of shape [36 x 1024] of the top box features for each image in a list
        top_box_features_per_image = []

        # also collect an boolean array of shape [36] that specifies if a class was not predicted (True) for each image in a list
        class_not_predicted_per_image = []

        for pred_scores_image, box_features_image in zip(pred_scores_per_image, box_features_per_image):
            # get the predicted class for every box
            pred_classes = torch.argmax(pred_scores_image, dim=1)

            # create a mask that is True at the predicted class index for every box
            mask_pred_classes = torch.zeros_like(pred_scores_image, dtype=torch.bool, device=pred_scores_image.device)
            mask_pred_classes[torch.arange(mask_pred_classes.size(0)), pred_classes] = True

            # by multiplying the pred_scores with the mask, we set to 0.0 all scores except for the top score in each row
            pred_top_scores_image = pred_scores_image * mask_pred_classes

            # get the row indices of the box features with the top score for each class
            inds_boxes_with_top_scores = torch.argmax(pred_top_scores_image, dim=0)

            # check if all classes have at least 1 box where they are the predicted class (i.e. have the highest score)
            # this is done because we want to collect 36 box features (each with the highest score for the class) for 36 regions
            num_predictions_per_class = torch.sum(mask_pred_classes, dim=0)

            # get a boolean array that is True for the classes that were not predicted
            class_not_predicted = num_predictions_per_class == 0

            # extract the box features for the top score for each class
            # note that if a class was not predicted (as the class with the highest score for at least 1 box),
            # then the argmax will have returned index 0 for that class (since all scores of the class will have been 0.0)
            # but since we have the boolean array class_not_predicted, we can filter out this class (and its box feature) later
            top_box_features = box_features_image[inds_boxes_with_top_scores]
            top_box_features_per_image.append(top_box_features)

            class_not_predicted_per_image.append(class_not_predicted)

        top_box_features_per_image_tensor = torch.stack(top_box_features_per_image, dim=0)
        class_not_predicted_per_image_tensor = torch.stack(class_not_predicted_per_image, dim=0)

        return top_box_features_per_image_tensor, class_not_predicted_per_image_tensor

    def postprocess_detections(
        self,
        class_logits: Tensor,
        box_regression: Tensor,
        proposals: List[Tensor],
        image_shapes: List[Tuple[int, int]]
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

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

        if self.training:
            if self.return_feature_vectors:
                # get the top-1 bbox features for every class (i.e. a tensor of shape [batch_size, 36, 1024])
                # the box_features are sorted by class (i.e. the 2nd dim is sorted)
                top_box_features = self.get_top_box_features(box_features, class_logits, proposals)
        else:
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
