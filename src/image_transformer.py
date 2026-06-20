import logging

from torchvision import transforms

logger = logging.getLogger(__name__)

def _geometry_transforms(image, normalization_list, resolution=512, crop="center"):
    if crop == "resize_crop":
        logger.info("Executing _geometry_transforms with resize crop")
        result = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(normalization_list, normalization_list)
        ])(image)
        logger.info("Finished _geometry_transforms")
        return result

    # Center crop
    logger.info("Executing _geometry_transforms with center crop")
    min_side = min(image.size)
    result = transforms.Compose([
        transforms.CenterCrop(min_side),
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(normalization_list, normalization_list)
    ])(image)
    logger.info("Finished _geometry_transforms")
    return result

def to_rgb_image_tensors(sample_image, resolution=1600, crop="center"):
    logger.info("Executing to_rgb_image_tensors")
    image = sample_image["image"].convert("RGB")
    result = _geometry_transforms(image, [0.5, 0.5, 0.5], resolution, crop)
    logger.info("Finished to_rgb_image_tensors")
    return result

def to_gray_image_tensors(sample_image, resolution=1600, crop="center"):
    logger.info("Executing to_gray_image_tensors")
    image = sample_image["image"].convert("L")
    result = _geometry_transforms(image, [0.5, 0.5], resolution, crop)
    logger.info("Finished to_gray_image_tensors")
    return result
