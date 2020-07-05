#!/usr/bin/env bash
python test.py --proj_dir proj_log \
                --exp_name mpc-3depn-table \
                --module gan \
                --dataset_name 3depn \
                --category table \
                --data_root data/ShapeNetPointCloud \
                --data_raw_root data/shapenet_dim32_sdf_pc \
                --pretrain_ae_path proj_log/mpc-3depn-table/ae/model/ckpt_epoch3000.pth \
                --pretrain_vae_path proj_log/mpc-3depn-table/vae/model/ckpt_epoch3000.pth \
                --num_sample -1 \
                --num_z 10 \
                --ckpt 500 \
                -g 0