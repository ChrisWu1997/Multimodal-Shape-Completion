#!/usr/bin/env bash
python train.py --proj_dir proj_log \
                --exp_name mpc-partnet_scan-table \
                --module vae \
                --dataset_name partnet_scan \
                --category Table \
                --data_root /mnt/data/partnet_data/shape_mesh_scan_pc \
                --batch_size 200 \
                --lr 5e-4 \
                --lr_decay 0.999 \
                --save_frequency 500 \
                --nr_epochs 2000 \
                --num_workers 16 \
                -g 0