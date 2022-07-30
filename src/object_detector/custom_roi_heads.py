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

    def get_top_region_features_detections_class_not_predicted(
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
            -> output dict contains the keys "detections" and "class_not_predicted":

            - "detections" maps to another dict with the keys "top_region_boxes" and "top_scores":
                - "top_region_boxes" maps to a tensor of shape [batch_size, 36, 4] of the detected boxes with the highest score (i.e. top-1 score) per class
                - "top_scores" maps to a tensor of shape [batch_size, 36] of the corresponding highest scores for the boxes

            - "class_not_predicted" maps to a boolean tensor of shape [batch_size, 36] that has a True value for a class if that class did not have
            the highest score (out of all classes) for at least 1 proposed box. Meaning for all hundreds of proposed boxes coming from the RPN for a single image,
            this class did not have the highest score (and thus was not predicted as the class) for one of them. We use the boolean tensor of "class_not_predicted"
            to mask out the detected boxes for these classes in "detections"

        (2) object detector is used with full model + train mode:
            -> output dict contains the keys "top_region_features" and "class_not_predicted":

            - "top_region_features" maps to a tensor of shape [batch_size, 36, 1024] of the region features with the highest score (i.e. top-1 score) per class
            - "class_not_predicted" same as above. Needed to mask out the region features for classes that were not predicted later on in the full model

        (3) object detector is used with full model + eval mode:
            -> output dict contains the keys "detections", "top_region_features", "class_not_predicted":
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
        output["class_not_predicted"] = []  # list collects the bool arrays of shape [36] that specify if a class was not predicted (True) for each image
        output["top_region_features"] = []  # list collects the tensors of shape [36 x 1024] of the top region features for each image

        # list top_region_boxes collects the tensors of shape [36 x 4] of the top region boxes for each image
        # list top_scores collects the tensors of shape [36] of the corresponding top scores for each image
        output["detections"] = {
            "top_region_boxes": [],
            "top_scores": []
        }

        for pred_scores_img, pred_region_boxes_img, region_features_img, img_shape in zip(pred_scores_per_img, pred_region_boxes_per_img, region_features_per_img, image_shapes):
            # get the predicted class for each box (dim=1 goes by box)
            pred_classes = torch.argmax(pred_scores_img, dim=1)

            # create a mask that is 1 at the predicted class index for every box and 0 otherwise
            mask_pred_classes = torch.nn.functional.one_hot(pred_classes, num_classes=36).to(pred_scores_img.device)

            # by multiplying the pred_scores with the mask, we set to 0.0 all scores except for the top score in each row
            pred_top_scores_img = pred_scores_img * mask_pred_classes

            # get the scores and row indices of the box/region features with the top-1 score for each class (dim=0 goes by class)
            top_scores, indices_with_top_scores = torch.max(pred_top_scores_img, dim=0)

            # check if all regions/classes have at least 1 box where they are the predicted class (i.e. have the highest score)
            # this is done because we want to collect 36 region features (each with the highest score for the class) for 36 regions
            num_predictions_per_class = torch.sum(mask_pred_classes, dim=0)

            # get a boolean array that is True for the classes that were not predicted
            class_not_predicted = (num_predictions_per_class == 0)

            output["class_not_predicted"].append(class_not_predicted)

            if self.return_feature_vectors:
                # extract the region features with the top scores for each class
                # note that if a class was not predicted (as the class with the highest score for at least 1 box),
                # then the argmax will have returned index 0 for that class (since all scores of the class will have been 0.0)
                # and thus its region features will be the 1st one in the tensor region_features_img
                # but since we have the boolean array class_not_predicted, we can filter out this class (and its erroneous region feature) later on
                top_region_features = region_features_img[indices_with_top_scores]
                output["top_region_features"].append(top_region_features)

            if not self.training:
                pred_region_boxes_img = box_ops.clip_boxes_to_image(pred_region_boxes_img, img_shape)

                # remove predictions with the background label
                pred_region_boxes_img = pred_region_boxes_img[:, 1:]

                # TODO: check if the line below and the selecting of the top_region_boxes is correct
                # batch everything, by making every class prediction be a separate instance
                pred_region_boxes_img = pred_region_boxes_img.reshape(-1, 4)

                # extract the region boxes with the top scores for each class
                # note that if a class was not predicted (as the class with the highest score for at least 1 box),
                # then the argmax will have returned index 0 for that class (since all scores of the class will have been 0.0)
                # and thus the region box will be the 1st one in the tensor pred_region_boxes_img
                # but since we have the boolean array class_not_predicted, we can filter out this class (and its erroneous region box) later on
                top_region_boxes = pred_region_boxes_img[indices_with_top_scores]

                output["detections"]["top_region_boxes"].append(top_region_boxes)
                output["detections"]["top_scores"].append(top_scores)

        # convert lists into batched tensors
        output["class_not_predicted"] = torch.stack(output["class_not_predicted"], dim=0)

        if self.return_feature_vectors:
            output["top_region_features"] = torch.stack(output["top_region_features"], dim=0)

        if not self.training:
            output["detections"]["top_region_boxes"] = torch.stack(output["detections"]["top_region_boxes"], dim=0)
            output["detections"]["top_scores"] = torch.stack(output["detections"]["top_scores"], dim=0)

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

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        detections: List[Dict[str, torch.Tensor]] = []
        detector_losses = {}

        if labels and regression_targets:
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            detector_losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}

        # we always return the detector_losses (even if it's an empty dict, which is the case for targets==None (i.e. during inference))
        roi_heads_output = {}
        roi_heads_output["detector_losses"] = detector_losses

        # if we train the full model (i.e. self.return_feature_vectors == True), then we need top_region_features and class_not_predicted
        # if we evaluate the object detector (in isolation or as part of the full model), then we need the detections
        if self.return_feature_vectors or not self.training:
            output = self.get_top_region_features_detections_class_not_predicted(box_features, box_regression, class_logits, proposals, image_shapes)

            if self.return_feature_vectors:
                roi_heads_output["top_region_features"] = output["top_region_features"]
                roi_heads_output["class_not_predicted"] = output["class_not_predicted"]
            
            if not self.training:
                roi_heads_output["detections"] = output["detections"]

        return roi_heads_output


        # # if we don't return the region features, then we train/evaluate the object detector in isolation (i.e. not as part of the full model)
        # if not self.return_feature_vectors:
        #     if self.training:
        #         # we only need the losses to train the object detector
        #         return losses
        #     else:
        #         # we need both losses and detections to evaluate the object detector
        #         return losses, detections

        # # if we return region features, then we train/evaluate the full model (with object detector as one part of it)
        # if self.return_feature_vectors:
        #     if self.training:
        #         # we need the losses to train the object detector, and the top_region_features/class_not_predicted to train the binary classifier and decoder
        #         return losses, top_region_features, class_not_predicted
        #     else:
        #         # we additionally need the detections to evaluate the object detector
        #         return losses, detections, top_region_features, class_not_predicted


        if self.training:
            if self.return_feature_vectors:
                # get the top-1 bbox features for every class (i.e. a tensor of shape [batch_size, 36, 1024])
                # the box_features are sorted by class (i.e. the 2nd dim is sorted)
                # also get class_not_predicted, a boolean tensor of shape [batch_size, 36], that specifies if
                # a class was predicted by the object detector for at least 1 proposal
                top_region_features, class_not_predicted = self.get_top_region_features(box_features, class_logits, proposals, return_detections=False)
        else:
            # in eval mode, also 



            # boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            # num_images = len(boxes)
            # for i in range(num_images):
            #     detections.append(
            #         {
            #             "boxes": boxes[i],
            #             "labels": labels[i],
            #             "scores": scores[i],
            #         }
            #     )

        if self.return_feature_vectors:
            roi_heads_output["top_region_features"] = top_region_features
            roi_heads_output["class_not_predicted"] = class_not_predicted

        return roi_heads_output
