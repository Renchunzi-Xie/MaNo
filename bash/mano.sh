#!/usr/bin/env bash

gpu=$1


for ARCH in resnet18 resnet50 wrn_50_2
do
    python main.py --alg mano --arch ${ARCH} --severity -1 --dataname cifar10 --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path /mnt/sharedata/ssd/common/datasets --cifar_corruption_path /mnt/sharedata/ssd/common/datasets/CIFAR-10-C --norm_type 4 --score errors --beta 8
    python main.py --alg mano --arch ${ARCH} --severity -1 --dataname cifar100 --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path /mnt/sharedata/ssd/common/datasets --cifar_corruption_path /mnt/sharedata/ssd/common/datasets/CIFAR-100-C --norm_type 4 --score errors --beta 8
    python main.py --alg mano --arch ${ARCH} --severity -1 --dataname tinyimagenet --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path /mnt/sharedata/ssd/users/weihx/tiny-imagenet-200 --cifar_corruption_path /mnt/sharedata/ssd/users/weihx/Tiny-ImageNet-C --norm_type 4 --score errors --beta 8
    python main.py --alg mano --arch ${ARCH} --severity -1 --dataname imagenet --lr 0.001 --num_classes 1000 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path /mnt/sharedata/ssd/common/datasets/imagenet --cifar_corruption_path /mnt/sharedata/ssd/common/datasets/imagenet-corruption --norm_type 4 --score errors --beta 8
    python main.py --alg mano --arch ${ARCH} --dataname pacs --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path /mnt/sharedata/ssd/users/weihx/archive/pacs_data/pacs_data --cifar_corruption_path /mnt/sharedata/ssd/users/weihx/archive/pacs_data/pacs_data  --norm_type 4 --score errors --beta 3
    python main.py --alg mano --arch ${ARCH} --dataname office_home --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path /mnt/sharedata/ssd/users/weihx/OfficeHomeDataset_10072016 --cifar_corruption_path /mnt/sharedata/ssd/users/weihx/OfficeHomeDataset_10072016 --norm_type 4 --score errors --beta 8
    python main.py --alg mano --arch ${ARCH} --severity -1 --dataname domainnet --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path /mnt/sharedata/ssd/users/weihx/DomainNet --cifar_corruption_path /mnt/sharedata/ssd/users/weihx/DomainNet --norm_type 4 --score errors --beta 3
    python main.py --alg mano --arch ${ARCH} --severity -1 --dataname wilds_rr1 --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path /mnt/sharedata/ssd/users/weihx --cifar_corruption_path /mnt/sharedata/ssd/users/weihx --norm_type 4 --score errors --beta 8
    python main.py --alg mano --arch ${ARCH} --severity -1 --dataname entity13 --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path /mnt/sharedata/ssd/common/datasets/imagenet/images --cifar_corruption_path /mnt/sharedata/ssd/common/datasets/imagenet-corruption --norm_type 4 --score errors --beta 8
    python main.py --alg mano --arch ${ARCH} --severity -1 --dataname entity30 --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path /mnt/sharedata/ssd/common/datasets/imagenet/images --cifar_corruption_path /mnt/sharedata/ssd/common/datasets/imagenet-corruption --norm_type 4 --score errors --beta 8
    python main.py --alg mano --arch ${ARCH} --severity -1 --dataname living17 --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path /mnt/sharedata/ssd/common/datasets/imagenet/images --cifar_corruption_path /mnt/sharedata/ssd/common/datasets/imagenet-corruption --norm_type 4 --score errors --beta 8
    python main.py --alg mano --arch ${ARCH} --severity -1 --dataname nonliving26 --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path /mnt/sharedata/ssd/common/datasets/imagenet/images --cifar_corruption_path /mnt/sharedata/ssd/common/datasets/imagenet-corruption --norm_type 4 --score errors --beta 8
done
    python init_base_model.py --arch resnet18 --train_epoch 20 --train_data_name cifar10 --lr 0.001 --batch_size 128 --seed 123 --gpu 7 --cifar_data_path /mnt/sharedata/ssd/common/datasets --cifar_corruption_path /mnt/sharedata/ssd/common/datasets/CIFAR-10-C

    python main.py --alg mano --arch resnet18 --severity -1 --dataname cifar10 --lr 0.001 --batch_size 128 --seed 1 --gpu 7 --cifar_data_path /mnt/sharedata/ssd/common/datasets --cifar_corruption_path /mnt/sharedata/ssd/common/datasets/CIFAR-10-C --norm_type 4 --score errors --beta 8
