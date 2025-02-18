"""PACS dataset."""

import os

import torchvision.transforms as transforms
import torchvision.datasets as datasets


def get_pacs_loader(
    clean_path,
    corruption_path,
    corruption_type,
    datatype="test",
):
    """Get the PACS dataset.

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
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        data_path = os.path.join(clean_path, corruption_type)
        dataset = datasets.ImageFolder(data_path, transform)
    else:
        test_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        data_path = os.path.join(corruption_path, corruption_type)
        dataset = datasets.ImageFolder(data_path, test_transform)

    return dataset
