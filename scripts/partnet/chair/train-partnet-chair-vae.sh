#!/usr/bin/env bash
python train.py --proj_dir proj_log \
                --exp_name mpc-partnet-chair \
                --module vae \
                --dataset_name partnet \
                --category Chair \
                --data_root /dev/data/partnet_data/partnet_original/PartNet \
                --batch_size 200 \
                --lr 5e-4 \
                --lr_decay 0.999 \
                --save_frequency 500 \
                --nr_epochs 2000 \
                --num_workers 16 \
                -g 0