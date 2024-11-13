#!/usr/bin/env bash

gpu=$1

for ARCH in resnet18 resnet50 wrn_50_2
do
    python init_base_model.py --arch ${ARCH} --train_epoch 20 --train_data_name cifar10 --lr 0.001 --batch_size 128 --seed 123 --gpu ${gpu} --cifar_data_path path_of_train_data --cifar_corruption_path path_of_corrupted_data
    python init_base_model.py --arch ${ARCH} --train_epoch 50 --train_data_name cifar100 --lr 0.001 --batch_size 128 --seed 123 --gpu ${gpu}  --cifar_data_path path_of_train_data --cifar_corruption_path path_of_corrupted_data
    python init_base_model.py --arch ${ARCH} --train_epoch 50 --train_data_name tinyimagenet --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path path_of_train_data --cifar_corruption_path path_of_corrupted_data
    python init_base_model.py --arch ${ARCH} --train_epoch 50 --train_data_name office_home --corruption Art --severity 1 --lr 0.001 --batch_size 128 --seed 123 --gpu ${gpu} --cifar_data_path path_of_train_data --cifar_corruption_path path_of_corrupted_data
    python init_base_model.py --arch ${ARCH} --train_epoch 50 --train_data_name office_home --corruption Clipart --severity 1 --lr 0.001 --batch_size 128 --seed 123 --gpu ${gpu}  --cifar_data_path path_of_train_data --cifar_corruption_path path_of_corrupted_data
    python init_base_model.py --arch ${ARCH} --train_epoch 50 --train_data_name office_home --corruption Product --severity 1 --lr 0.001 --batch_size 128 --seed 123 --gpu ${gpu} --cifar_data_path path_of_train_data --cifar_corruption_path path_of_corrupted_data
    python init_base_model.py --arch ${ARCH} --train_epoch 50 --train_data_name office_home --corruption Real_World --severity 1 --lr 0.001 --batch_size 128 --seed 123 --gpu ${gpu}  --cifar_data_path path_of_train_data --cifar_corruption_path path_of_corrupted_data
    python init_base_model.py --arch ${ARCH} --train_epoch 50 --train_data_name entity13 --lr 0.001 --batch_size 128 --seed 123 --gpu ${gpu} --cifar_data_path path_of_train_data --cifar_corruption_path path_of_corrupted_data
    python init_base_model.py --arch ${ARCH} --train_epoch 50 --train_data_name entity30 --lr 0.001 --batch_size 128 --seed 123 --gpu ${gpu} --cifar_data_path path_of_train_data --cifar_corruption_path path_of_corrupted_data
    python init_base_model.py --arch ${ARCH} --train_epoch 50 --train_data_name living17 --lr 0.001 --batch_size 128 --seed 123 --gpu ${gpu}  --cifar_data_path path_of_train_data --cifar_corruption_path path_of_corrupted_data
    python init_base_model.py --arch ${ARCH} --train_epoch 50 --train_data_name nonliving26 --lr 0.001 --batch_size 128 --seed 123 --gpu ${gpu}  --cifar_data_path path_of_train_data --cifar_corruption_path path_of_corrupted_data
done