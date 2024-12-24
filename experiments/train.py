"""Pre-training of the base models to be used to estimate their accuracy on OOD test data."""

import argparse
import os

import torch
import torch.nn as nn

from data.utils import build_dataloader
from models.utils import get_model

# Arguments
parser = argparse.ArgumentParser(description="Pre-training of base models.")
parser.add_argument("--arch", default="resnet18", type=str)
parser.add_argument("--gpu", type=str, default=None)
parser.add_argument("--train_data_name", default="cifar10", type=str)
parser.add_argument("--data_path", default="../datasets/Cifar10", type=str)
parser.add_argument("--corruption_path", default="../datasets/Cifar10/CIFAR-10-C", type=str)
parser.add_argument("--num_classes", default=10, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--train_epoch", default=2, type=int)
parser.add_argument("--num_samples", default=50000, type=int)
parser.add_argument("--seed", default=1, type=int)
parser.add_argument("--severity", default=0, type=int)
parser.add_argument("--init", default="matching", type=str)
parser.add_argument("--alg", default="standard", type=str)
parser.add_argument("--corruption", default="all", type=str)
args = vars(parser.parse_args())

num_class_dict = {
    "cifar10": 10,
    "cifar100": 100,
    "tinyimagenet": 200,
    "pacs": 7,
    "imagenet": 1000,
    "office_home": 65,
    "wilds_rr1": 1139,
    "entity30": 30,
    "entity13": 13,
    "living17": 17,
    "nonliving26": 26,
    "domainnet": 345,
}
args["num_classes"] = num_class_dict[args["train_data_name"]]

# Set device
DEVICE = torch.device(f"cuda:{args['gpu']}" if torch.cuda.is_available() else "cpu")


def train(model, loader, device=torch.device("cpu")):
    """Train the base model.

    Parameters
    ----------
    model: nn.Module
        Base model to be trained.
    loader: Dataloader
        Dataloader for training data.
    device: torch.device, default=torch.device("cpu")
        Determine which device calculations are performed.
    """

    # Set training mode
    model.train()

    # Optimizer
    if "rr1" in args["train_data_name"]:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args["lr"],
            momentum=0.9,
            weight_decay=0.0,
        )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args["train_epoch"] * len(loader),
    )

    # Training loss
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args["train_epoch"]):
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, batch_data in enumerate(loader):
            inputs, targets = batch_data[0], batch_data[1]
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx % 200 == 0:
                for param_group in optimizer.param_groups:
                    current_lr = param_group["lr"]
                print(
                    f"Epoch: {epoch}",
                    f"({batch_idx}/{len(loader)})",
                    f"Loss: {train_loss / (batch_idx + 1):0.3f}",
                    f"| Acc: {100.0 * correct / total:0.3f} ({int(correct)}/{int(total)})",
                    f"| Lr: {current_lr:0.5f}",
                )
            scheduler.step()

    # Set evaluation mode
    model.eval()
    return model


if __name__ == "__main__":

    # Saving path
    if args["train_data_name"] == "imagenet":
        save_dir_path = print(f"./checkpoints/{args['train_data_name']}_{args['arch']}_{str(args['num_classes'])}")
    elif args["train_data_name"] in ["pacs", "office_home", "domainnet"]:
        save_dir_path = print(f"./checkpoints/{args['corruption']}_{args['arch']}")
    else:
        save_dir_path = print(f"./checkpoints/{args['train_data_name']}_{args['arch']}")
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    # Setup train/val_iid loaders
    trainloader = build_dataloader(args["train_data_name"], args)

    # Train base model
    base_model = get_model(args["arch"], args["num_classes"], args["seed"]).to(DEVICE)
    base_model = train(model=base_model, loader=trainloader, device=DEVICE)

    # Save base model
    torch.save(base_model.state_dict(), f"{save_dir_path}/base_model.pt")
    print("base model saved to", f"{save_dir_path}/base_model.pt")
