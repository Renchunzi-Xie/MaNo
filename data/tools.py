from torch.utils.data import Dataset
import torch.utils.data
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import random

class myDataset(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.target = y
    def __len__(self):
        return len(self.target)
    def __getitem__(self, item):
        data = (self.data[item], self.target[item])
        return data

# Assumes that tensor is (nchannels, height, width)
def tensor_rot_90(x):
    return x.flip(2).transpose(1, 2)


def tensor_rot_180(x):
    return x.flip(2).flip(1)


def tensor_rot_270(x):
    return x.transpose(1, 2).flip(2)


def rotate_batch_with_labels(batch, labels):
    images = []
    for img, label in zip(batch, labels):
        if label == 1:
            img = tensor_rot_90(img)
        elif label == 2:
            img = tensor_rot_180(img)
        elif label == 3:
            img = tensor_rot_270(img)
        images.append(img.unsqueeze(0))
    return torch.cat(images)


def rotate_batch(batch, label):
    device = batch.device
    if label == 'rand':
        labels = torch.randint(4, (len(batch),), dtype=torch.long).to(device)
    elif label == 'expand':
        labels = torch.cat([torch.zeros(len(batch), dtype=torch.long).to(device),
                            torch.zeros(len(batch), dtype=torch.long).to(device) + 1,
                            torch.zeros(len(batch), dtype=torch.long).to(device) + 2,
                            torch.zeros(len(batch), dtype=torch.long).to(device) + 3])
        batch = batch.repeat((4, 1, 1, 1))
    else:
        assert isinstance(label, int)
        labels = torch.zeros((len(batch),), dtype=torch.long).to(device) + label
    return rotate_batch_with_labels(batch, labels), labels

def projected_gradient_descent(model, x, y, loss_fn, num_steps, step_size):
    """Performs the projected gradient descent attack on a batch of images."""
    x_adv = x.clone().detach().requires_grad_(True).to(x.device)

    for i in range(num_steps):
        _x_adv = x_adv.clone().detach().requires_grad_(True)

        prediction = model(_x_adv)
        loss = loss_fn(prediction, y)
        loss.backward()

        with torch.no_grad():
            # Force the gradient step to be a fixed size in a certain norm
            gradients = _x_adv.grad.sign() * step_size
            x_adv += gradients
            x_adv = torch.max(torch.min(x_adv, x + 8/255), x - 8/255)
            # x_adv = torch.clip(x_adv, 0, 1)

    return x_adv.detach()