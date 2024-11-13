import argparse
from models.utils import get_model
import torch.nn as nn
from data.utils import build_dataloader
from data.tools import rotate_batch
import torch
import os

"""# Configuration"""
parser = argparse.ArgumentParser(description="Train base models for different matrics.")
parser.add_argument("--arch", default="resnet18", type=str)
parser.add_argument("--gpu", type=str, default=None)
parser.add_argument("--train_data_name", default="cifar10", type=str)
parser.add_argument("--cifar_data_path", default="../datasets/Cifar10", type=str)
parser.add_argument(
    "--cifar_corruption_path", default="../datasets/Cifar10/CIFAR-10-C", type=str
)
parser.add_argument("--num_classes", default=10, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--train_epoch", default=2, type=int)
parser.add_argument("--num_samples", default=50000, type=int)
parser.add_argument("--seed", default=1, type=int)
parser.add_argument("--severity", default=0, type=int)
parser.add_argument("--init", default="matching", type=str)
parser.add_argument("--alg", default="standard", type=str)
# pacs
parser.add_argument("--corruption", default="all", type=str)
args = vars(parser.parse_args())

if args["gpu"] is not None:
    device = torch.device(f"cuda:{args['gpu']}")
else:
    device = torch.device("cpu")

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


def train(net, trainloader, device):
    net.train()
    if "rr1" in args["train_data_name"]:
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)
    else:
        optimizer = torch.optim.SGD(
            net.parameters(), lr=args["lr"], momentum=0.9, weight_decay=0.0
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args["train_epoch"] * len(trainloader)
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args["train_epoch"]):
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, batch_data in enumerate(trainloader):
            inputs, targets = batch_data[0], batch_data[1]
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
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
                    "Epoch: ",
                    epoch,
                    "(",
                    batch_idx,
                    "/",
                    len(trainloader),
                    ")",
                    "Loss: %.3f | Acc: %.3f%% (%d/%d)| Lr: %.5f"
                    % (
                        train_loss / (batch_idx + 1),
                        100.0 * correct / total,
                        correct,
                        total,
                        current_lr,
                    ),
                )
            scheduler.step()
    net.eval()
    return net


if __name__ == "__main__":
    # save path
    if args["train_data_name"] == "imagenet":
        save_dir_path = "./checkpoints/{}".format(
            args["train_data_name"]
            + "_"
            + args["arch"]
            + "_"
            + str(args["num_classes"])
        )
    elif args["train_data_name"] in ["pacs", "office_home", "domainnet"]:
        save_dir_path = "./checkpoints/{}".format(
            args["corruption"] + "_" + args["arch"]
        )
    else:
        save_dir_path = "./checkpoints/{}".format(
            args["train_data_name"] + "_" + args["arch"]
        )
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    # setup train/val_iid loaders
    trainloader = build_dataloader(args["train_data_name"], args)

    # init and train base model
    base_model = get_model(args["arch"], args["num_classes"], args["seed"]).to(device)
    base_model = train(base_model, trainloader, device)

    torch.save(base_model.state_dict(), "{}/base_model.pt".format(save_dir_path))
    print("base model saved to", "{}/base_model.pt".format(save_dir_path))
