"""RR1-WILDS dataset."""

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from wilds.datasets.rxrx1_dataset import RxRx1Dataset


def get_wilds_rr1_loader(
    clean_path,
    corruption_type,
    datatype="test",
):
    """Get the RR1-WILDS dataset.

    Parameters
    ----------
    clean_path : str
        Path to the clean data.
    corruption_type : str
        Corruption type.
    datatype : str, default="test"
        Type of the data. If "train", the loader is used for training.
        If "test", it is used for testing.
    """

    assert datatype in ["train", "test"], "Error: datatype should be train or test."
    dataset = RxRx1Dataset(download=True, root_dir=f"{clean_path}/")

    if datatype == "train":
        subset = dataset.get_subset(datatype, transform=initialize_rxrx1_transform(True))
    else:
        subset = dataset.get_subset(corruption_type, transform=initialize_rxrx1_transform(False))

    return subset


def initialize_rxrx1_transform(training_flag=False):
    """Initialize the RxRx1 transform.

    Parameters
    -----------
    training_flag : bool, default=False
        Flag to indicate if the transform is for training or testing.
    """

    def standardize(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2))
        std = x.std(dim=(1, 2))
        std[std == 0.0] = 1.0
        return TF.normalize(x, mean, std)

    t_standardize = transforms.Lambda(standardize)

    angles = [0, 90, 180, 270]

    def random_rotation(x: torch.Tensor) -> torch.Tensor:
        angle = angles[torch.randint(low=0, high=len(angles), size=(1,))]
        if angle > 0:
            x = TF.rotate(x, angle)
        return x

    t_random_rotation = transforms.Lambda(random_rotation)

    if training_flag:
        transforms_ls = [
            t_random_rotation,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            t_standardize,
        ]
    else:
        transforms_ls = [
            transforms.ToTensor(),
            t_standardize,
        ]
    transform = transforms.Compose(transforms_ls)
    return transform
