from torchvision import transforms

DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
DEFAULT_NORMALIZE_STD = [0.229, 0.224, 0.225]


TRAIN = transforms.Compose(
    [
        transforms.Resize(DEFAULT_IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.7),
        transforms.GaussianBlur(3),
        transforms.ToTensor(),
        transforms.Normalize(DEFAULT_NORMALIZE_MEAN, DEFAULT_NORMALIZE_STD),
    ]
)

INFER = transforms.Compose(
    [
        transforms.Resize(DEFAULT_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(DEFAULT_NORMALIZE_MEAN, DEFAULT_NORMALIZE_STD),
    ]
)
