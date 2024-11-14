import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Subset


def load_Imagenet(corruption_type,
                   clean_cifar_path,
                   corruption_cifar_path,
                   num_classes,
                   corruption_severity=0,
                   datatype='test'):

    assert datatype == 'test' or datatype == 'train' or datatype == 'val'

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
    #                                 transforms.ToTensor(), transforms.Normalize(mean, std)])
    if corruption_type == 'clean':
        # transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
        #                                 transforms.ToTensor(), transforms.Normalize(mean, std)])
        transform = transforms.Compose([
            transforms.Resize(256),  # Resize images to 256 x 256
            transforms.CenterCrop(224),  # Center crop image
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # Converting cropped images to tensors
            transforms.Normalize(mean=mean, std=std)
        ])
        dataset = datasets.ImageFolder(root=clean_cifar_path + '/' + 'train', transform=transform)
    else:

        transform = transforms.Compose([
            transforms.Resize(256),  # Resize images to 256 x 256
            transforms.CenterCrop(224),  # Center crop image
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # Converting cropped images to tensors
            transforms.Normalize(mean=mean, std=std)
        ])

        dataset = datasets.ImageFolder(root=corruption_cifar_path + '/' + corruption_type + '/' + str(corruption_severity),
                                       transform=transform)
    if num_classes != 1000:
        idx = []
        count = 0
        while count < num_classes:
            idx_temp = [i for i in range(len(dataset)) if dataset.imgs[i][1] == count]
            idx = idx + idx_temp
            count += 1
        # build the appropriate subset
        subset = Subset(dataset, idx)
    else:
        subset = dataset
    return subset
