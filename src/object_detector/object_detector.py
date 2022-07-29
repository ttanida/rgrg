from collections import OrderedDict
from typing import Optional, List, Dict

import torch
from torch import Tensor
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchinfo import summary
import torchxrayvision as xrv

from custom_roi_heads import CustomRoIHeads
from custom_rpn import CustomRegionProposalNetwork
from image_list import ImageList


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

    The PyTorch implementation returns a Dict[Tensor] containing the 4 losses in train mode, and a List[Dict[Tensor]] containing
    the detections for each image in eval mode.

    My implementation returns also only returns the loss_dict in train mode, but in eval mode returns both the loss_dict (with the val_losses)
    and the detections (note that if targets == None in eval mode, then losses will be an empty dict).

    A single detection dict for a single image contains the following fields:
        - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores or each prediction
    """

    def __init__(self, return_feature_vectors=False):
        super().__init__()
        # boolean to specify if feature vectors should be returned after TwoMLPHead (i.e. linear layers) inside RoIHeads
        self.return_feature_vectors = return_feature_vectors

        # 36 classes for 36 anatomical regions + background (defined as class 0)
        self.num_classes = 37

        # use only the feature extractor of the pre-trained classification model
        self.backbone = xrv.models.DenseNet(weights="densenet121-res224-all").features

        # FasterRCNN needs to know the number of output channels of the backbone
        # for densenet121, it's 1024 (with feature maps of size 7x7)
        self.backbone.out_channels = 1024

        self.rpn = self._create_rpn()
        self.roi_heads = self._create_roi_heads()

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
            aspect_ratios=((0.2, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.3, 1.5, 2.1, 2.6, 3.0, 8.0),),
            # sizes=((32, 64, 128, 256, 512),),
            # aspect_ratios=((0.5, 1.0, 2.0),),
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
        # (uniform) size of feature maps after roi pooling layer is defined in output_size
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2)

        resolution = roi_pooler.output_size[0]
        representation_size = 1024

        box_head = TwoMLPHead(self.backbone.out_channels * resolution**2, representation_size)
        box_predictor = FastRCNNPredictor(representation_size, self.num_classes)

        # use default values for RoI heads
        roi_heads = CustomRoIHeads(
            return_feature_vectors=self.return_feature_vectors,
            box_roi_pool=roi_pooler,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.25,
            bbox_reg_weights=None,
            score_thresh=0.01,  # TODO: try out different values
            nms_thresh=0.0,  # TODO: try out different values
            detections_per_img=100,  # TODO: set detections_per_img to 36 for 36 anatomical regions?
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
            features (Tensor): of shape [batch_size, 1024, 7, 7]

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
            images (Tensor): images to be processed of shape [batch_size, 1, 224, 224] (gray-scale images of size 224 x 224)
            targets (List[Dict[str, Tensor]]): list of batch_size dicts, where a single dict contains the fields:
                - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format
                - labels (Int64Tensor[N]): the class label for each ground-truth box

        Returns:
            in train mode:
                - losses (Dict[Tensor]), which contains the 4 losses

            in eval mode:
                - losses (Dict[Tensor]). If targets == None, then losses will be an empty dict.
                - detections (List[Dict[str, Tensor]]), which are the predictions for each input image.

                The fields of a single dict (for a single image) are:
                    - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format
                    - labels (Int64Tensor[N]): the predicted labels for each image
                    - scores (Tensor[N]): the scores or each prediction
        """
        if targets is not None:
            self._check_targets(targets)

        features = self.backbone(images)

        images, features = self._transform_inputs_for_rpn_and_roi(images, features)

        proposals, proposal_losses = self.rpn(images, features, targets)
        roi_heads_output = self.roi_heads(features, proposals, images.image_sizes, targets)

        # the roi_heads_output always includes the detector_losses
        detector_losses = roi_heads_output["detector_losses"]

        # they include the detections when we are evaluating
        if not self.training:
            detections = roi_heads_output["detections"]

        # they include the top_region_features and class_not_predicted if we train/evaluate the full model
        if self.return_feature_vectors:
            top_region_features = roi_heads_output["top_region_features"]
            class_not_predicted = roi_heads_output["class_not_predicted"]

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        # if we don't return the region features, then we train/evaluate the object detector in isolation (i.e. not as part of the full model)
        if not self.return_feature_vectors:
            if self.training:
                # we only need the losses to train the object detector
                return losses
            else:
                # we need both losses and detections to evaluate the object detector
                return losses, detections

        # if we return region features, then we train/evaluate the full model (with object detector as one part of it)
        if self.return_feature_vectors:
            if self.training:
                # we need the losses to train the object detector, and the top_region_features/class_not_predicted to train the binary classifier and decoder
                return losses, top_region_features, class_not_predicted
            else:
                # we additionally need the detections to evaluate the object detector
                return losses, detections, top_region_features, class_not_predicted


# model = ObjectDetector()

# device = torch.device("cpu")
# model = ObjectDetector(return_feature_vectors=False)
# model.eval()
# model.to(device)

# images = torch.rand(3, 1, 224, 224)
# targets = [
#     {
#         "boxes": torch.FloatTensor([[3, 5, 7, 8], [3, 5, 7, 8], [3, 5, 7, 8], [3, 5, 7, 8]]),
#         "labels": torch.tensor([2, 4, 3, 6], dtype=torch.int64),
#     },
#     {
#         "boxes": torch.FloatTensor([[3, 5, 7, 8], [3, 5, 7, 8], [3, 5, 7, 8], [3, 5, 7, 8]]),
#         "labels": torch.tensor([2, 4, 3, 6], dtype=torch.int64),
#     },
#     {
#         "boxes": torch.FloatTensor([[3, 5, 7, 8], [3, 5, 7, 8], [3, 5, 7, 8], [3, 5, 7, 8]]),
#         "labels": torch.tensor([2, 4, 3, 6], dtype=torch.int64),
#     },
# ]

# summary(model, input_data=(images, targets))
# summary(model)

# loss, detections = model(images)
# print(loss)
# print(detections)
# print(box_features.shape)
