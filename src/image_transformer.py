from torchvision import transforms

def _geometry_transforms(image, normalization_list, resolution=512, crop="center"):
    if crop == "resize_crop":
        return transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(normalization_list, normalization_list)
        ])(image)
    
    # Center crop
    min_side = min(image.size)
    return transforms.Compose([
        transforms.CenterCrop(min_side),
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(normalization_list, normalization_list)
    ])(image)

def to_rgb_image_tensors(sample_image, resolution=1600, crop="center"):
    image = sample_image["image"].convert("RGB")
    return _geometry_transforms(image, [0.5, 0.5, 0.5], resolution, crop)

def to_gray_image_tensors(sample_image, resolution=1600, crop="center"):
    image = sample_image["image"].convert("L")
    return _geometry_transforms(image, [0.5, 0.5], resolution, crop)
