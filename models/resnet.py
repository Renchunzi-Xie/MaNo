import torch
import torch.nn as nn
import torchvision.models as models


def ResNet18(num_classes=10, seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    resnet18.fc = nn.Linear(512, num_classes)
    return resnet18

def ResNet50(num_classes=10, seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet50.fc = nn.Linear(2048, num_classes)
    return resnet50







