#!/usr/bin/env bash
python train.py --proj_dir proj_log \
                --exp_name mpc-3depn-table \
                --module ae \
                --dataset_name 3depn \
                --category table \
                --data_root data/ShapeNetPointCloud \
                --batch_size 200 \
                --lr 5e-4 \
                --save_frequency 500 \
                --nr_epochs 3000 \
                -g 0