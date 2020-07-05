#!/usr/bin/env bash
python train.py --proj_dir proj_log \
                --exp_name mpc-3depn-airplane \
                --module gan \
                --dataset_name 3depn \
                --category airplane \
                --data_root data/ShapeNetPointCloud \
                --data_raw_root data/shapenet_dim32_sdf_pc \
                --pretrain_ae_path proj_log/mpc-3depn-airplane/ae/model/ckpt_epoch3000.pth \
                --pretrain_vae_path proj_log/mpc-3depn-airplane/vae/model/ckpt_epoch3000.pth \
                --batch_size 50 \
                --lr 5e-4 \
                --save_frequency 100 \
                --nr_epochs 500 \
                -g 0