import torch
from torchvision.transforms.functional import InterpolationMode


class CustomResize(torch.nn.Module):
    """Resize the input image while maintaining the aspect ratio.

    Each cropped image has a different size, however the images need to be of the same size (e.g. 224x224 for DenseNet-121) in order to batch
    and input them into a model.

    Instead of resizing both the longer edge and shorter edge of the input image to the same size, and thus losing the original aspect ratio,
    we resize the longer edge of the input image to the necessary size (e.g. 224) while maintaining the aspect ratio.
    The shorter edge is then padded both sides to 224 in the next transform (Pad) to get images of uniform size (224x224).

    This is done such that e.g. a cropped image of a lung maintains its aspect ratio and is not distorted too much.

    How the resize works:

    If an image has size (width, height) with height > width, then the image will be resized to size (output_size_longer_edge * (width/height), output_size_longer_edge).

    Example: image has size (100, 150), and desired output_size_longer_edge = 90, then the image will be resized to (90 * (100/150), 90) = (60, 90).

    There is a Resize transform in torch transforms, however the transform resizes the shorter edge of the image to the input parameter 'size'
    while maintaining the aspect ratio, and not the longer edge.

    There is an option to specify the parameter 'max_size' such that if the longer edge of the image is greater than 'max_size' after the image was resized
    according to 'size', then the image is resized again so that the longer edge is equal to 'max_size'.

    However 'size' and 'max_size' are not allowed to be the same value, such that one would have to specify e.g. 223 for 'size' and 224 for 'max_size'
    to get a resize of the longer edge to the desired length.

    This would work in most cases, if after the first resize the longer edge is larger than 'max_size', such that a second resize is triggered.
    However it fails if width == height, because in that case resized_width == resized_height == 223 after the first resize, and a 2nd resize would not be triggered.

    This made the creation of a custom Resize class necessary.

    Args:
        output_size_longer_edge (int): Desired output size of the longer edge after resizing.
        interpolation (InterpolationMode): Desired interpolation mode.
    """

    def __init__(self, output_size_longer_edge: int, interpolation=InterpolationMode.BILINEAR):
        super().__init__()
        self.output_size_longer_edge = output_size_longer_edge
        self.interpolation = interpolation

    def forward(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        w, h = img.size

        short, long = (w, h) if w <= h else (h, w)
        new_short, new_long = int(self.output_size_longer_edge * (short / long)), self.output_size_longer_edge

        new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)

        if (w, h) == (new_w, new_h):
            return img
        else:
            return img.resize((new_w, new_h), self.interpolation)

    def __repr__(self) -> str:
        detail = f"(output_size_longer_edge={self.output_size_longer_edge}, interpolation={self.interpolation.value})"
        return f"{self.__class__.__name__}{detail}"
