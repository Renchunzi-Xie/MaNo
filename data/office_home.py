"""Office-Home dataset."""

import os

import torchvision.transforms as transforms
import torchvision.datasets as datasets


def get_office_home_loader(
    clean_path,
    corruption_path,
    corruption_type,
    datatype="test",
):
    """Get the Office-Home dataset.

    Parameters
    ----------
    clean_path : str
        Path to the clean data.
    corruption_path : str
        Path to the corrupted data.
    corruption_type : str
        Corruption type.
    datatype : str, default="test"
        Type of the data. If "train", the loader is used for training.
        If "test", it is used for testing.
    """

    assert datatype in ["train", "test"], "Error: datatype should be train or test."
    if datatype == "train":
        transform = apply_transform_(resize_size=256, crop_size=224)
        data_path = os.path.join(clean_path, corruption_type)
        dataset = datasets.ImageFolder(data_path, transform)
    else:
        test_transform = apply_test_transform_(resize_size=256, crop_size=224)
        data_path = os.path.join(corruption_path, corruption_type)
        dataset = datasets.ImageFolder(data_path, test_transform)

    return dataset


def apply_transform_(resize_size=256, crop_size=224):
    """Training time transformation of the dataset."""

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose(
        [
            ResizeImage(resize_size),
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )


def apply_test_transform_(resize_size=256, crop_size=224):
    """Test time transformation of the dataset."""

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    start_center = (resize_size - crop_size - 1) / 2
    data_transforms = transforms.Compose(
        [
            ResizeImage(resize_size),
            PlaceCrop(crop_size, start_center, start_center),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return data_transforms


class ResizeImage:
    """Resize the input PIL Image to the given size."""

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


class PlaceCrop(object):
    """Crops the given PIL.Image at the particular index.
    If size is an int instead of sequence (w, h),
    a square crop of dimension (size, size) is made.
    """

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))
