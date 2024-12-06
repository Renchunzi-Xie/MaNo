import torchvision.transforms as transforms
import torchvision.datasets as dset


def get_offce_home_loader(corruption_type,
                   clean_cifar_path,
                   corruption_cifar_path,
                   corruption_severity=0,
                   datatype='test'):

    assert datatype == 'test' or datatype == 'train'
    transform = transform1(resize_size=256, crop_size=224)
    test_transform = test_transform1(resize_size=256, crop_size=224)
    if datatype == 'test':
        data_path = corruption_cifar_path + '/' + corruption_type
        datasets = dset.ImageFolder(data_path, test_transform)
    elif datatype == 'train':
        data_path = clean_cifar_path + '/' + corruption_type
        datasets = dset.ImageFolder(data_path, transform)
    return datasets

def transform1(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        ResizeImage(resize_size),
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def test_transform1(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    start_center = (resize_size - crop_size - 1) / 2
    data_transforms = transforms.Compose([
        ResizeImage(resize_size),
        PlaceCrop(crop_size, start_center, start_center),
        transforms.ToTensor(),
        normalize
    ])
    return data_transforms


class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


class PlaceCrop(object):
    """Crops the given PIL.Image at the particular index.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
    """

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))