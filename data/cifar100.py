""" Cifar100 dataset."""

import os

import numpy as np

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def get_cifar100_loader(
    clean_path,
    corruption_path,
    corruption_type,
    corruption_severity=0,
    datatype="test",
    num_samples=50000,
    seed=1,
):
    """Get the Cifar100 dataset.

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
    num_samples : int, default=50000
        Number of samples to use.
    seed : int, default=1
        Random seed.
    """
    assert datatype == "test" or datatype == "train"
    training_flag = True if datatype == "train" else False

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    dataset = datasets.CIFAR100(
        clean_path,
        train=training_flag,
        transform=transform,
        download=True,
    )

    if corruption_severity > 0:
        assert not training_flag
        path_images = os.path.join(corruption_path, corruption_type + ".npy")
        path_labels = os.path.join(corruption_path, "labels.npy")
        start = (corruption_severity - 1) * 10000
        end = corruption_severity * 10000
        dataset.data = np.load(path_images)[start:end]
        dataset.targets = list(np.load(path_labels)[start:end])
        dataset.targets = [int(item) for item in dataset.targets]

    # Randomly permutate data
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    number_samples = dataset.data.shape[0]
    index_permute = torch.randperm(number_samples)
    dataset.data = dataset.data[index_permute]
    dataset.targets = np.array([int(item) for item in dataset.targets])
    dataset.targets = dataset.targets[index_permute].tolist()

    # Randomly select a subset of the dataset
    if datatype == "train" and num_samples < 50000:
        num_samples = int(num_samples)
        indices = torch.randperm(50000)[:num_samples]
        dataset = torch.utils.data.Subset(dataset, indices)
        print("number of training data: ", len(dataset))
    if datatype == "test" and num_samples < 10000:
        num_samples = int(num_samples)
        indices = torch.randperm(10000)[:num_samples]
        dataset = torch.utils.data.Subset(dataset, indices)
        print("number of test data: ", len(dataset))

    return dataset
