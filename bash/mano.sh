#!/usr/bin/env bash

gpu=$1


for ARCH in resnet18 resnet50 wrn_50_2
do
    python main.py --alg mano --arch ${ARCH} --severity -1 --dataname cifar10 --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path path_of_train_data --cifar_corruption_path path_of_corrupted_data --norm_type 4
    python main.py --alg mano --arch ${ARCH} --severity -1 --dataname cifar100 --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path path_of_train_data --cifar_corruption_path path_of_corrupted_data --norm_type 4
    python main.py --alg mano --arch ${ARCH} --severity -1 --dataname tinyimagenet --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path path_of_train_data --cifar_corruption_path path_of_corrupted_data --norm_type 4
    python main.py --alg mano --arch ${ARCH} --severity -1 --dataname imagenet --lr 0.001 --num_classes 1000 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path path_of_train_data --cifar_corruption_path path_of_corrupted_data --norm_type 4
    python main.py --alg mano --arch ${ARCH} --dataname pacs --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path path_of_train_data --cifar_corruption_path path_of_corrupted_data  --norm_type 4
    python main.py --alg mano --arch ${ARCH} --dataname office_home --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path path_of_train_data --cifar_corruption_path path_of_corrupted_data --norm_type 4
    python main.py --alg mano --arch ${ARCH} --severity -1 --dataname domainnet --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path path_of_train_data --cifar_corruption_path path_of_corrupted_data --norm_type 4
    python main.py --alg mano --arch ${ARCH} --severity -1 --dataname wilds_rr1 --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path path_of_train_data --cifar_corruption_path path_of_corrupted_data --norm_type 4
    python main.py --alg mano --arch ${ARCH} --severity -1 --dataname entity13 --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path path_of_train_data --cifar_corruption_path path_of_corrupted_data --norm_type 4
    python main.py --alg mano --arch ${ARCH} --severity -1 --dataname entity30 --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path path_of_train_data --cifar_corruption_path path_of_corrupted_data --norm_type 4
    python main.py --alg mano --arch ${ARCH} --severity -1 --dataname living17 --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path path_of_train_data --cifar_corruption_path path_of_corrupted_data --norm_type 4
    python main.py --alg mano --arch ${ARCH} --severity -1 --dataname nonliving26 --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path path_of_train_data --cifar_corruption_path path_of_corrupted_data --norm_type 4
done

