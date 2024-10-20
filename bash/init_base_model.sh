#!/usr/bin/env bash

gpu=$1

for ARCH in resnet18 resnet50 wrn_50_2
do
    python init_base_model.py --arch ${ARCH} --train_epoch 20 --train_data_name cifar10 --lr 0.001 --batch_size 128 --seed 123 --gpu ${gpu} --cifar_data_path ../datasets/Cifar10 --cifar_corruption_path ../datasets/CIFAR-10-C
    python init_base_model.py --arch ${ARCH} --train_epoch 50 --train_data_name cifar100 --lr 0.001 --batch_size 128 --seed 123 --gpu ${gpu}  --cifar_data_path ..datasets/Cifar100 --cifar_corruption_path ../datasets/CIFAR-100-C
    python init_base_model.py --arch ${ARCH} --train_epoch 50 --train_data_name tinyimagenet --lr 0.001 --batch_size 128 --seed 1 --gpu ${gpu} --cifar_data_path ../datasets/tiny-imagenet-200 --cifar_corruption_path ../datasets/Tiny-ImageNet-C
    python init_base_model.py --arch ${ARCH} --train_epoch 50 --train_data_name office_home --corruption Art --severity 1 --lr 0.001 --batch_size 128 --seed 123 --gpu ${gpu} --cifar_data_path ../datasets/OfficeHomeDataset_10072016 --cifar_corruption_path ../datasets/OfficeHomeDataset_10072016
    python init_base_model.py --arch ${ARCH} --train_epoch 50 --train_data_name office_home --corruption Clipart --severity 1 --lr 0.001 --batch_size 128 --seed 123 --gpu ${gpu}  --cifar_data_path ../datasets/OfficeHomeDataset_10072016 --cifar_corruption_path ../datasets/OfficeHomeDataset_10072016
    python init_base_model.py --arch ${ARCH} --train_epoch 50 --train_data_name office_home --corruption Product --severity 1 --lr 0.001 --batch_size 128 --seed 123 --gpu ${gpu}  --cifar_data_path ../datasets/OfficeHomeDataset_10072016 --cifar_corruption_path ../datasets/OfficeHomeDataset_10072016
    python init_base_model.py --arch ${ARCH} --train_epoch 50 --train_data_name office_home --corruption Real_World --severity 1 --lr 0.001 --batch_size 128 --seed 123 --gpu ${gpu}  --cifar_data_path ../datasets/OfficeHomeDataset_10072016 --cifar_corruption_path ../datasets/OfficeHomeDataset_10072016
    python init_base_model.py --arch ${ARCH} --train_epoch 50 --train_data_name entity13 --lr 0.001 --batch_size 128 --seed 123 --gpu ${gpu} --cifar_data_path /data/DS1618p/Image__ILSVRC2012 --cifar_corruption_path ../datasets/imagenet-c
    python init_base_model.py --arch ${ARCH} --train_epoch 50 --train_data_name entity30 --lr 0.001 --batch_size 128 --seed 123 --gpu ${gpu}  --cifar_data_path /data/DS1618p/Image__ILSVRC2012 --cifar_corruption_path ../datasets/imagenet-c
    python init_base_model.py --arch ${ARCH} --train_epoch 50 --train_data_name living17 --lr 0.001 --batch_size 128 --seed 123 --gpu ${gpu}  --cifar_data_path /data/DS1618p/Image__ILSVRC2012 --cifar_corruption_path ../datasets/imagenet-c
    python init_base_model.py --arch ${ARCH} --train_epoch 50 --train_data_name nonliving26 --lr 0.001 --batch_size 128 --seed 123 --gpu ${gpu}  --cifar_data_path /data/DS1618p/Image__ILSVRC2012 --cifar_corruption_path ../datasets/imagenet-c
done