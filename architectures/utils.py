from models.resnet import *
from models.wrn import *

import torchvision.models as models


def get_model(arch, num_classes, seed):
    if arch == 'resnet18':
        model = ResNet18(num_classes=num_classes, seed=seed)
    elif arch == 'resnet50':
        model = ResNet50(num_classes=num_classes, seed=seed)
    elif arch == 'wrn_50_2':
        model = wrn_50_2(num_classes=num_classes, seed=seed)
    else:
        raise Exception("Not Implemented Error")
    return model

def get_imagenet_model(arch, num_classes, seed):
    if arch == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif arch == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    elif arch == 'wrn_50_2':
        model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
    else:
        raise Exception("Not Implemented Error")
    return model