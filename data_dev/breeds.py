""" BREEDS dataset."""

import torchvision.transforms as transforms
from .breeds_utils.helpers import get_label_mapping
from .breeds_utils import folder
from .breeds_utils.breeds_helpers import (
    make_living17,
    make_entity13,
    make_entity30,
    make_nonliving26,
)


def get_imagenet_breeds(
    corruption_type,
    clean_cifar_path,
    corruption_cifar_path,
    name,
    corruption_severity=0,
    datatype="test",
):
    if name == "living17":
        ret = make_living17("./data/imagenet_class_hierarchy/", split="good")
    elif name == "entity13":
        ret = make_entity13("./data/imagenet_class_hierarchy/", split="good")
    elif name == "entity30":
        ret = make_entity30("./data/imagenet_class_hierarchy/", split="good")
    elif name == "nonliving26":
        ret = make_nonliving26("./data/imagenet_class_hierarchy/", split="good")
    else:
        raise KeyError("Unknown dataset name.")

    source_label_mapping = get_label_mapping("custom_imagenet", ret[1][0])
    target_label_mapping = get_label_mapping("custom_imagenet", ret[1][1])

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.4717, 0.4499, 0.3837], [0.2600, 0.2516, 0.2575]),
        ]
    )
    if (corruption_type == "clean") and (datatype == "train"):
        dataset = folder.ImageFolder(
            root=clean_cifar_path + "/" + "train",
            transform=transform,
            label_mapping=source_label_mapping,
        )
    else:
        if corruption_severity == 0:
            dataset = folder.ImageFolder(
                root=clean_cifar_path + "/" + "train",
                transform=transform,
                label_mapping=target_label_mapping,
            )
        else:
            dataset = folder.ImageFolder(
                root=corruption_cifar_path
                + "/"
                + corruption_type
                + "/"
                + str(corruption_severity),
                transform=transform,
                label_mapping=target_label_mapping,
            )
    return dataset
