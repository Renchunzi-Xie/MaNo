import argparse
from algs.utils import create_alg
from data.utils import build_dataloader
import numpy as np
import time

"""# Configuration"""
parser = argparse.ArgumentParser(description='ProjNorm.')
parser.add_argument('--arch', default='resnet18', type=str)
parser.add_argument('--alg', default='standard', type=str)

parser.add_argument('--gpu', type=str, default=None)
parser.add_argument('--cifar_data_path',
                    default='../datasets/Cifar10', type=str)
parser.add_argument('--cifar_corruption_path',
                    default='../datasets/Cifar10/CIFAR-10-C', type=str)
parser.add_argument('--corruption', default='all', type=str)
parser.add_argument('--severity', default=0, type=int)
parser.add_argument('--dataname', default='cifar10', type=str)
parser.add_argument('--num_classes', default=10, type=int)
parser.add_argument('--num_samples', default=50000, type=float)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--threshold', default=0.5, type=float)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--norm_type', default=4, type=int)
# pacs
parser.add_argument('--source', default='None', type=str)

args = vars(parser.parse_args())

import torch
if args["gpu"] is not None:
    device = torch.device(f"cuda:{args['gpu']}")
else:
    device = torch.device('cpu')


def correlation(var1, var2):
    return np.corrcoef(var1, var2)[0, 1]
def correlation2(var1, var2):
    return (np.corrcoef(var1, var2)[0, 1]) ** 2
# spearman
def spearman(var1, var2):
    from scipy import stats
    return stats.spearmanr(var1, var2)

num_class_dict = {
    "cifar10":10,
    "cifar100":100,
    "tinyimagenet":200,
    "pacs":7,
    'imagenet': 1000,
    "office_home": 65,
    "wilds_rr1": 1139,
    "entity30": 30,
    "entity13": 13,
    "living17": 17,
    "nonliving26": 26,
    "domainnet":345
}

args["num_classes"] = num_class_dict[args["dataname"]]

if __name__ == "__main__":
    # device
    if (args['dataname']=="cifar10") or (args['dataname']=="cifar100"):
        corruption_list = ["brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost", "gaussian_blur", "gaussian_noise", "glass_blur",
                           "impulse_noise", "jpeg_compression", "motion_blur", "pixelate", "saturate", "shot_noise", "snow", "spatter", "speckle_noise", "zoom_blur"]
        max_severity = 5
    elif args["dataname"] == 'tinyimagenet':
        corruption_list = ["brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost", "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression",
                           "motion_blur", "pixelate", "shot_noise", "snow", "zoom_blur"]
        max_severity = 5
    # setup featurized val_ood/val_ood loaders
    elif args["dataname"] == 'pacs':
        corruption_list = ['art_painting', 'cartoon', 'photo', 'sketch_pacs']
        max_severity = 1
    elif args["dataname"] == 'office_home':
        corruption_list = ['Art', 'Clipart', 'Product', 'Real_World']
        max_severity = 1
    elif args["dataname"] == "imagenet":
        corruption_list = ['frost', 'impulse_noise', 'snow', 'zoom_blur', 'brightness', 'elastic_transform', 'gaussian_blur', 'jpeg_compression', 'pixelate', 'spatter', 'contrast',
                           'gaussian_noise', 'motion_blur', 'saturate', 'speckle_noise', 'defocus_blur', 'fog', 'glass_blur', 'shot_noise']
        max_severity = 5
    elif "rr1" in args["dataname"]:
        corruption_list = ['id_test', 'val', 'test']
        max_severity = 1
    elif ("entity30" in args['dataname']) or ("entity13" in args['dataname']) or ("living17" in args['dataname']) or ("nonliving26" in args['dataname']):
        corruption_list = ['frost', 'impulse_noise', 'snow', 'zoom_blur', 'brightness', 'elastic_transform', 'gaussian_blur', 'jpeg_compression', 'pixelate', 'spatter', 'contrast',
                           'gaussian_noise', 'motion_blur', 'saturate', 'speckle_noise', 'defocus_blur', 'fog', 'glass_blur', 'shot_noise']
        max_severity = 5
    elif args['dataname'] == 'domainnet':
        corruption_list = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        max_severity = 1
    else:
        raise TypeError('No relevant corruption list!')

    if args['dataname'] not in ['pacs', 'office_home', 'domainnet']:
        scores_list = []
        test_acc_list = []
        time_list = []
        print('alg:{}, dataname:{}, model:{}'.format(args['alg'], args['dataname'], args['arch']))
        for corruption in corruption_list:
            for severity in range(1, max_severity+1):
                args["corruption"] = corruption
                args["severity"] = severity
                # (original x, true labels)
                val_loader = build_dataloader(args['dataname'], args)
                # Define model
                alg_obj = create_alg(args['alg'], val_loader, device, args)
                start_time = time.time()
                scores = alg_obj.evaluate()
                end_time = time.time()
                if args['use_f1_score']:
                    test_acc = alg_obj.f1_score_test()
                else:
                    test_acc = alg_obj.test()
                scores_list.append(float(scores))
                time_list.append(float(end_time - start_time))
                test_acc_list.append(float(test_acc))
                print('corruption:{}, severity:{}, score:{}, test acc:{}'.format(args['corruption'], args['severity'], scores, test_acc))
        mean_score = np.mean(scores_list)
        mean_time = np.mean(time_list)
        print('Mean scores:{}, time:{}'.format(mean_score, mean_time))
        print("Correlation:{}".format(correlation2(scores_list, test_acc_list)))
        print("Spearman:{}".format(spearman(scores_list, test_acc_list).correlation))
    else:
        scores_list = []
        test_acc_list = []
        time_list = []
        print('alg:{}, dataname:{}, model:{}'.format(args['alg'],
                                                     args['dataname'], args['arch']))
        for corruption in corruption_list:
            for source in corruption_list:
                if corruption != source:
                    args["corruption"] = corruption
                    args["source"] = source
                    args["severity"] = 1
                    val_loader = build_dataloader(args['dataname'], args)
                    # Define model
                    alg_obj = create_alg(args['alg'], val_loader, device, args)

                    start_time = time.time()
                    scores = alg_obj.evaluate()
                    end_time = time.time()

                    test_acc = alg_obj.test()
                    scores_list.append(float(scores))
                    time_list.append(float(end_time - start_time))
                    test_acc_list.append(float(test_acc))
                    print('corruption:{}, severity:{}, score:{}, test acc:{}'.format(args['corruption'], args['severity'],
                                                                                     scores, test_acc))
        mean_score = np.mean(scores_list)
        mean_time = np.mean(time_list)
        print('Mean scores:{}, time:{}'.format(mean_score, mean_time))
        print("Correlation:{}".format(correlation2(scores_list, test_acc_list)))
        print("Spearman:{}".format(spearman(scores_list, test_acc_list).correlation))

