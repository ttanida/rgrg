from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss
from torchvision.ops import boxes as box_ops


class CustomRoIHeads(RoIHeads):
    def __init__(
        self,
        return_feature_vectors,
        feature_map_output_size,
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
        # return_feature_vectors == True if we train/evaluate the object detector as part of the full model
        self.return_feature_vectors = return_feature_vectors

        # set kernel_size = feature_map_output_size, such that we average over the whole feature maps
        self.avg_pool = nn.AvgPool2d(kernel_size=feature_map_output_size)
        self.dim_reduction = nn.Linear(2048, 1024)

    def get_top_region_features_detections_class_detected(
        self,
        box_features,
        box_regression,
        class_logits,
        proposals,
        image_shapes
    ):
        """
        Method returns an output dict containing different values depending on if:
            - the object detector is used in isolation (i.e. self.return_feature_vectors == False) or as part of the full model (i.e. self.return_feature_vectors == True)
            - we are in train or eval mode

        The possibilities are:

        (1) object detector is used in isolation + eval mode:
            -> output dict contains the keys "detections" and "class_detected":

            - "detections" maps to another dict with the keys "top_region_boxes" and "top_scores":
                - "top_region_boxes" maps to a tensor of shape [batch_size, 29, 4] of the detected boxes with the highest score (i.e. top-1 score) per class
                - "top_scores" maps to a tensor of shape [batch_size, 29] of the corresponding highest scores for the boxes

            - "class_detected" maps to a boolean tensor of shape [batch_size, 29] that has a True value for a class if that class had the highest score (out of all classes)
            for at least 1 proposed box. If a class has a False value, this means that for all hundreds of proposed boxes coming from the RPN for a single image,
            this class did not have the highest score (and thus was not predicted/detected as the class) for one of them. We use the boolean tensor of "class_detected"
            to mask out the boxes for these False/not-detected classes in "detections"

        (2) object detector is used with full model + train mode:
            -> output dict contains the keys "top_region_features" and "class_detected":

            - "top_region_features" maps to a tensor of shape [batch_size, 29, 2048] of the region features with the highest score (i.e. top-1 score) per class
            - "class_detected" same as above. Needed to mask out the region features for classes that were not detected later on in the full model

        (3) object detector is used with full model + eval mode:
            -> output dict contains the keys "detections", "top_region_features", "class_detected":
            -> all keys same as above
        """
        # apply softmax on background class as well
        # (such that if the background class has a high score, all other classes will have a low score)
        pred_scores = F.softmax(class_logits, -1)

        # remove score of the background class
        pred_scores = pred_scores[:, 1:]

        # get number of proposals/boxes per image
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]

        num_images = len(boxes_per_image)

        # split pred_scores (which is a tensor with scores for all RoIs of all images in the batch)
        # into the tuple pred_scores_per_img (where 1 pred_score tensor has scores for all RoIs of 1 image)
        pred_scores_per_img = torch.split(pred_scores, boxes_per_image, dim=0)

        # if we train/evaluate the full model, we need the top region/box features
        if self.return_feature_vectors:
            # split region_features the same way as pred_scores
            region_features_per_img = torch.split(box_features, boxes_per_image, dim=0)
        else:
            region_features_per_img = [None] * num_images  # dummy list such that we can still zip everything up

        # if we evaluate the object detector, we need the detections
        if not self.training:
            pred_region_boxes = self.box_coder.decode(box_regression, proposals)
            pred_region_boxes_per_img = torch.split(pred_region_boxes, boxes_per_image, dim=0)
        else:
            pred_region_boxes_per_img = [None] * num_images  # dummy list such that we can still zip everything up

        output = {}
        output["class_detected"] = []  # list collects the bool arrays of shape [29] that specify if a class was detected (True) for each image
        output["top_region_features"] = []  # list collects the tensors of shape [29 x 2048] of the top region features for each image

        # list top_region_boxes collects the tensors of shape [29 x 4] of the top region boxes for each image
        # list top_scores collects the tensors of shape [29] of the corresponding top scores for each image
        output["detections"] = {
            "top_region_boxes": [],
            "top_scores": []
        }

        for pred_scores_img, pred_region_boxes_img, region_features_img, img_shape in zip(pred_scores_per_img, pred_region_boxes_per_img, region_features_per_img, image_shapes):
            # get the predicted class for each box (dim=1 goes by box)
            pred_classes = torch.argmax(pred_scores_img, dim=1)

            # create a mask that is 1 at the predicted class index for every box and 0 otherwise
            mask_pred_classes = torch.nn.functional.one_hot(pred_classes, num_classes=29).to(pred_scores_img.device)

            # by multiplying the pred_scores with the mask, we set to 0.0 all scores except for the top score in each row
            pred_top_scores_img = pred_scores_img * mask_pred_classes

            # get the scores and row indices of the box/region features with the top-1 score for each class (dim=0 goes by class)
            top_scores, indices_with_top_scores = torch.max(pred_top_scores_img, dim=0)

            # check if all regions/classes have at least 1 box where they are the predicted class (i.e. have the highest score)
            # this is done because we want to collect 29 region features (each with the highest score for the class) for 29 regions
            num_predictions_per_class = torch.sum(mask_pred_classes, dim=0)

            # get a boolean array that is True for the classes that were detected
            class_detected = (num_predictions_per_class > 0)

            output["class_detected"].append(class_detected)

            if self.return_feature_vectors:
                # extract the region features with the top scores for each class
                # note that if a class was not predicted/detected (as the class with the highest score for at least 1 box),
                # then the argmax will have returned index 0 for that class (since all scores of the class will have been 0.0)
                # and thus its region features will be the 1st one in the tensor region_features_img
                # but since we have the boolean array class_detected, we can filter out this class (and its erroneous region feature) later on
                top_region_features = region_features_img[indices_with_top_scores]
                output["top_region_features"].append(top_region_features)

            if not self.training:
                # pred_region_boxes_img is of shape [num_boxes_in_image x 30 x 4]

                # clip boxes so that they lie inside an image of size "img_shape"
                pred_region_boxes_img = box_ops.clip_boxes_to_image(pred_region_boxes_img, img_shape)

                # remove predictions with the background label
                # pred_region_boxes_img is now of shape [num_boxes_in_image x 29 x 4]
                pred_region_boxes_img = pred_region_boxes_img[:, 1:]

                # extract the region boxes with the top scores for each class
                # note that if a class was not predicted/detected (as the class with the highest score for at least 1 box),
                # then the argmax will have returned index 0 for that class (since all scores of the class will have been 0.0)
                # and thus the region box will be the 1st one in the tensor pred_region_boxes_img
                # but since we have the boolean array class_detected, we can filter out this class (and its erroneous region box) later on

                # since indices_with_top_scores is sorted from class 0 to class 28, we first use the indices to select the correct box_array (of shape [29 x 4]),
                # and then the number in torch.arange (starting from 0 and ending at 28) will select the correct box for this class from the box_array
                top_region_boxes = pred_region_boxes_img[indices_with_top_scores, torch.arange(start=0, end=29, dtype=torch.int64, device=indices_with_top_scores.device)]

                # note: top_region_boxes and top_scores are both ordered by class
                # (i.e. class 0 will be the first row in top_region_boxes and the first value in top_scores,
                # class 28 will be the last row in top_region_boxes and the last value in top_scores)
                output["detections"]["top_region_boxes"].append(top_region_boxes)
                output["detections"]["top_scores"].append(top_scores)

        # convert lists into batched tensors
        output["class_detected"] = torch.stack(output["class_detected"], dim=0)  # of shape [batch_size x 29]

        if self.return_feature_vectors:
            output["top_region_features"] = torch.stack(output["top_region_features"], dim=0)  # of shape [batch_size x 29 x 2048]

        if not self.training:
            output["detections"]["top_region_boxes"] = torch.stack(output["detections"]["top_region_boxes"], dim=0)  # of shape [batch_size x 29 x 4]
            output["detections"]["top_scores"] = torch.stack(output["detections"]["top_scores"], dim=0)  # of shape [batch_size x 29]

        return output

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

        # box_roi_pool_feature_maps has shape [overall_num_proposals_for_all_images x 2048 x 8 x 8]
        box_roi_pool_feature_maps = self.box_roi_pool(features, proposals, image_shapes)

        # box_feature_vectors has shape [overall_num_proposals_for_all_images x 1024]
        box_feature_vectors = self.box_head(box_roi_pool_feature_maps)
        class_logits, box_regression = self.box_predictor(box_feature_vectors)

        detector_losses = {}

        if labels and regression_targets:
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            detector_losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}

        # we always return the detector_losses (even if it's an empty dict, which is the case for targets==None (i.e. during inference))
        roi_heads_output = {}
        roi_heads_output["detector_losses"] = detector_losses

        # if we train the full model (i.e. self.return_feature_vectors == True), we need the "top_region_features"
        # if we evaluate the object detector (in isolation or as part of the full model), we need the "detections"
        # if we do either of them, we always need "class_detected" (see doc_string of method for details)
        if self.return_feature_vectors or not self.training:
            # average over the spatial dimensions, i.e. transform roi pooling features maps from [num_proposals, 2048, 8, 8] to [num_proposals, 2048, 1, 1]
            box_features = self.avg_pool(box_roi_pool_feature_maps)

            # remove all dims of size 1
            box_features = torch.squeeze(box_features)

            output = self.get_top_region_features_detections_class_detected(box_features, box_regression, class_logits, proposals, image_shapes)

            roi_heads_output["class_detected"] = output["class_detected"]

            if self.return_feature_vectors:
                # transform top_region_features from [batch_size x 29 x 2048] to [batch_size x 29 x 1024]
                roi_heads_output["top_region_features"] = self.dim_reduction(output["top_region_features"])

            if not self.training:
                roi_heads_output["detections"] = output["detections"]

        return roi_heads_output
