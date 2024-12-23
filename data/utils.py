"""Functions to load the dataset based on the dataname and the corruption type"""

import torch

from torch.utils.data import DataLoader

from data.breeds import get_breeds_loader
from data.cifar10 import get_cifar10_loader
from data.cifar100 import get_cifar100_loader
from data.imagenet import get_imagenet_loader
from data.office_home import get_office_home_loader
from data.pacs import get_pacs_loader
from data.tinyimagenet import get_tinyimagenet_loader
from data.wilds_rr1 import get_wilds_rr1_loader


def build_dataloader(dataname, args):
    """Build dataloader for the given dataset and corruption type.
    If datatype is "train", the loader is used for the training.
    Otherwise, it is used for testing.

    Parameters
    ----------
    dataname: str
        Name of the dataset.
    args: dict
        Dictionary containing the arguments.

    Returns
    -------
    loader: DataLoader
        Dataloader for the given dataset and corruption type.
    """

    # Set arguments
    random_seeds = torch.randint(0, 10000, (2,))
    if args["severity"] == 0:
        seed = 1
        datatype = "train"
        corruption_type = "clean"
    else:
        corruption_type = args["corruption"]
        seed = random_seeds[1]
        datatype = "test"

    # Load dataset
    if dataname == "cifar10":
        dataset = get_cifar10_loader(
            corruption_type,
            clean_path=args["data_path"],
            corruption_path=args["corruption_path"],
            corruption_severity=args["severity"],
            datatype=datatype,
            seed=seed,
            num_samples=args["num_samples"],
        )
    elif dataname == "cifar100":
        dataset = get_cifar100_loader(
            corruption_type,
            clean_path=args["data_path"],
            corruption_path=args["corruption_path"],
            corruption_severity=args["severity"],
            datatype=datatype,
            seed=seed,
            num_samples=args["num_samples"],
        )
    elif dataname == "imagenet":
        dataset = get_imagenet_loader(
            corruption_type,
            clean_path=args["data_path"],
            corruption_path=args["corruption_path"],
            num_classes=args["num_classes"],
            corruption_severity=args["severity"],
            datatype=datatype,
        )
    elif dataname == "tinyimagenet":
        dataset = get_tinyimagenet_loader(
            clean_path=args["data_path"],
            corruption_path=args["corruption_path"],
            corruption_type=corruption_type,
            corruption_severity=args["severity"],
            datatype=datatype,
        )
    elif dataname == "pacs":
        if (dataname == "pacs") and (corruption_type == "sketch_pacs"):
            corruption_type = "sketch"
        dataset = get_pacs_loader(
            clean_path=args["data_path"],
            corruption_path=args["corruption_path"],
            corruption_type=corruption_type,
            corruption_severity=args["severity"],
            datatype=datatype,
        )
    elif dataname == "office_home":
        dataset = get_office_home_loader(
            clean_path=args["data_path"],
            corruption_path=args["corruption_path"],
            corruption_type=corruption_type,
            corruption_severity=args["severity"],
            datatype=datatype,
        )
    elif dataname == "wilds_rr1":
        dataset = get_wilds_rr1_loader(
            corruption_type,
            clean_path=args["data_path"],
            corruption_severity=args["severity"],
        )
    elif dataname in ["entity13", "entity30", "living17", "nonliving26"]:
        if (datatype == "train") and (args["alg"] != "frechet"):
            name = args["train_data_name"]
        else:
            name = args["dataname"]

        dataset = get_breeds_loader(
            name=name,
            clean_path=args["data_path"],
            corruption_path=args["corruption_path"],
            corruption_type=corruption_type,
            corruption_severity=args["severity"],
            datatype=datatype,
        )
    else:
        raise ValueError("Unknown dataset name.")

    # Create dataloader
    loader = DataLoader(dataset, batch_size=args["batch_size"], num_workers=4, shuffle=True)
    return loader
