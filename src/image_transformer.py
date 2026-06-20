"""Image preprocessing helpers: geometry transforms, tensor conversion, and display."""

import logging

from torchvision import transforms

logger = logging.getLogger(__name__)


def _geometry_transforms(image, normalization_list, resolution=512, crop="center"):
    """Crop, resize, tensorize and normalize a PIL image.

    Args:
        image: A PIL ``Image``.
        normalization_list: Per-channel mean/std used for both the mean and
            std of ``transforms.Normalize`` (e.g. ``[0.5, 0.5, 0.5]`` for RGB).
        resolution: Target square resolution in pixels.
        crop: ``"resize_crop"`` to resize first then center-crop to a square;
            any other value crops to a square (using the native shorter side)
            before resizing.

    Returns:
        A normalized ``torch.Tensor`` of shape ``(C, resolution, resolution)``.
    """
    if crop == "resize_crop":
        logger.info("Executing _geometry_transforms with resize crop")
        result = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(normalization_list, normalization_list),
        ])(image)
        logger.info("Finished _geometry_transforms, final image size: %s", tuple(result.shape))
        return result

    # Center crop
    logger.info("Executing _geometry_transforms with center crop")
    min_side = min(image.size)
    result = transforms.Compose([
        transforms.CenterCrop(min_side),
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(normalization_list, normalization_list),
    ])(image)
    logger.info("Finished _geometry_transforms, final image size: %s", tuple(result.shape))
    return result


def print_image(tensor_image, mean, std):
    """Denormalize a tensor image and display it.

    Args:
        tensor_image: A normalized ``torch.Tensor`` image, e.g. produced by
            ``to_rgb_image_tensors`` or ``to_gray_image_tensors``.
        mean: Per-channel mean used during normalization.
        std: Per-channel std used during normalization.
    """
    denormalize = transforms.Normalize(
        [-m / s for m, s in zip(mean, std)],
        [1 / s for s in std],
    )
    pil_image = transforms.ToPILImage()(denormalize(tensor_image))
    pil_image.show()


def to_rgb_image_tensors(sample_image, resolution=1600, crop="center"):
    """Convert a dataset sample's image to a normalized RGB tensor.

    Args:
        sample_image: A dataset row with an ``"image"`` key holding a PIL image.
        resolution: Target square resolution in pixels.
        crop: See ``_geometry_transforms``.

    Returns:
        A normalized ``torch.Tensor`` of shape ``(3, resolution, resolution)``.
    """
    logger.info("Executing to_rgb_image_tensors")
    image = sample_image["image"].convert("RGB")
    result = _geometry_transforms(image, [0.5, 0.5, 0.5], resolution, crop)
    logger.info("Finished to_rgb_image_tensors")
    return result


def tensor_image_to_pil(tensor_image):
    """Convert a tensor image directly to a PIL image, without denormalizing.

    Args:
        tensor_image: A ``torch.Tensor`` already in displayable ``[0, 1]`` range.

    Returns:
        A PIL ``Image``.
    """
    return transforms.ToPILImage()(tensor_image)


def to_gray_image_tensors(sample_image, resolution=1600, crop="center"):
    """Convert a dataset sample's image to a normalized grayscale tensor.

    Args:
        sample_image: A dataset row with an ``"image"`` key holding a PIL image.
        resolution: Target square resolution in pixels.
        crop: See ``_geometry_transforms``.

    Returns:
        A normalized ``torch.Tensor`` of shape ``(1, resolution, resolution)``.
    """
    logger.info("Executing to_gray_image_tensors")
    image = sample_image["image"].convert("L")
    result = _geometry_transforms(image, [0.5, 0.5], resolution, crop)
    logger.info("Finished to_gray_image_tensors")
    return result
