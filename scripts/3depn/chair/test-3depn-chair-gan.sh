#!/usr/bin/env bash
python test.py --proj_dir proj_log \
                --exp_name mpc-3depn-chair \
                --module gan \
                --dataset_name 3depn \
                --category chair \
                --data_root data/ShapeNetPointCloud \
                --data_raw_root data/shapenet_dim32_sdf_pc \
                --pretrain_ae_path proj_log/mpc-3depn-chair/ae/model/ckpt_epoch2000.pth \
                --pretrain_vae_path proj_log/mpc-3depn-chair/vae/model/ckpt_epoch2000.pth \
                --num_sample -1 \
                --num_z 10 \
                --ckpt 500 \
                -g 0