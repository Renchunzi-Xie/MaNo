"""ImageNet dataset."""

import os

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Subset


def get_imagenet_loader(
    clean_path,
    corruption_path,
    corruption_type,
    corruption_severity=0,
    datatype="test",
    num_classes=10,
):
    """Get the ImageNet dataset.

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
    num_classes : int, default=10
        Number of classes to use.
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
            root=os.path.join(corruption_path, corruption_type, str(corruption_severity)),
            transform=transform,
        )

    # Subsample the dataset
    if num_classes != 1000:
        idx = []
        count = 0
        while count < num_classes:
            idx_temp = [i for i in range(len(dataset)) if dataset.imgs[i][1] == count]
            idx = idx + idx_temp
            count += 1
        subset = Subset(dataset, idx)
    else:
        subset = dataset
    return subset
