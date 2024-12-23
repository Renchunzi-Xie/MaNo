import torchvision.transforms as transforms
import torchvision.datasets as dset


def get_pacs_loader(corruption_type, clean_path, corruption_path, corruption_severity=0, datatype="test"):

    assert datatype == "test" or datatype == "train"
    test_transform = transforms.Compose(
        [
            # transforms.CenterCrop(224),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    transform = transforms.Compose(
        [
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    if datatype == "test":
        data_path = corruption_path + "/" + corruption_type
        datasets = dset.ImageFolder(data_path, test_transform)
    elif datatype == "train":
        data_path = clean_path + "/" + corruption_type
        datasets = dset.ImageFolder(data_path, transform)
    return datasets
