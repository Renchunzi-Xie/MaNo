from wilds import get_dataset
from wilds.datasets.rxrx1_dataset import RxRx1Dataset
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms
import torch
import numpy as np
from wilds.datasets.wilds_dataset import WILDSSubset
import torchvision.transforms.functional as TF

def load_wilds_rxrx1(corruption_type,
                   clean_cifar_path,
                   corruption_cifar_path,
                   corruption_severity=0,
                   datatype='test',
                   seed=1):
    dataset = RxRx1Dataset(download=True, root_dir=f"{clean_cifar_path}/")

    if corruption_severity == 0:
        load_set = dataset.get_subset('train', transform=initialize_rxrx1_transform(True))
    else:
        load_set = dataset.get_subset(corruption_type, transform=initialize_rxrx1_transform(False))

    return load_set

def initialize_rxrx1_transform(is_training):

    def standardize(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2))
        std = x.std(dim=(1, 2))
        std[std == 0.] = 1.
        return TF.normalize(x, mean, std)
    t_standardize = transforms.Lambda(lambda x: standardize(x))

    angles = [0, 90, 180, 270]
    def random_rotation(x: torch.Tensor) -> torch.Tensor:
        angle = angles[torch.randint(low=0, high=len(angles), size=(1,))]
        if angle > 0:
            x = TF.rotate(x, angle)
        return x
    t_random_rotation = transforms.Lambda(lambda x: random_rotation(x))

    if is_training:
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