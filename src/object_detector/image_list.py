import torch
from torch import Tensor


class ImageList:
    """
    rpn and roi_heads of object detector require images to be of custom type ImageList.

    This class is a slightly modified implementation of the PyTorch implementation.
    (https://github.com/pytorch/vision/blob/main/torchvision/models/detection/image_list.py)
    """

    def __init__(self, images_tensor: Tensor) -> None:
        self.tensors = images_tensor

        # all tensors have the same shape (most likely [batch_size, 1, 512, 512])
        batch_size = images_tensor.shape[0]
        image_sizes = images_tensor.shape[-2:]

        self.image_sizes = [tuple(image_sizes) for _ in range(batch_size)]

    def to(self, device: torch.device) -> "ImageList":
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)
