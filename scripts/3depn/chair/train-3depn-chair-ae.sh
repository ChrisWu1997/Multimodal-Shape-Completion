#!/usr/bin/env bash
python train.py --proj_dir proj_log \
                --exp_name mpc-3depn-chair \
                --module ae \
                --dataset_name 3depn \
                --category chair \
                --data_root data/ShapeNetPointCloud \
                --batch_size 200 \
                --lr 5e-4 \
                --save_frequency 500 \
                --nr_epochs 2000 \
                -g 0