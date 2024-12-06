import torch
import torch.nn as nn
import torchvision.models as models

def wrn_50_2(num_classes=10, seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    wrn = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
    wrn.fc = nn.Linear(2048, num_classes)
    return wrn

