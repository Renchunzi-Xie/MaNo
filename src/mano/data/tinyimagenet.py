"""TinyImageNet dataset."""

import os

import torchvision.datasets as datasets
import torchvision.transforms as transforms


def get_tinyimagenet_loader(
    clean_path,
    corruption_path,
    corruption_type,
    corruption_severity=0,
    datatype="test",
):
    """Get the TinyImageNet dataset.

    Parameters
    ----------
    clean_path : str
        Path to the clean data.
    corruption_path : str
        Path to the corrupted data.
    corruption_type : str
        Corruption type.
    corruption_severity : int, default=0
        Severity of the corruption.
    datatype : str, default="test"
        Type of the data. If "train", the loader is used for training.
        If "test", it is used for testing.
    """

    assert datatype in ["train", "test"], "Error: datatype should be train or test."

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if datatype == "train":
        transform = transforms.Compose(
            [
                transforms.Resize(256),  # Resize images to 256 x 256
                transforms.CenterCrop(224),  # Center crop image
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # Converting cropped images to tensors
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        dataset = datasets.ImageFolder(root=os.path.join(clean_path, datatype), transform=transform)
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(256),  # Resize images to 256 x 256
                transforms.CenterCrop(224),  # Center crop image
                transforms.ToTensor(),  # Converting cropped images to tensors
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        dataset = datasets.ImageFolder(
            root=os.path.join(corruption_path, corruption_type, str(corruption_severity)), transform=transform
        )
    return dataset
