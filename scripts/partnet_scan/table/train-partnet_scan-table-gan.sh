#!/usr/bin/env bash
python train.py --proj_dir proj_log \
                --exp_name mpc-partnet_scan-table \
                --module gan \
                --dataset_name partnet_scan \
                --category Table \
                --data_root /mnt/data/partnet_data/shape_mesh_scan_pc \
                --data_raw_root /mnt/data/partnet_data/partial_mesh_scan_pc \
                --pretrain_ae_path proj_log/mpc-partnet_scan-table/ae/model/ckpt_epoch2000.pth \
                --pretrain_vae_path proj_log/mpc-partnet_scan-table/vae/model/ckpt_epoch2000.pth \
                --batch_size 50 \
                --lr 5e-4 \
                --save_frequency 100 \
                --nr_epochs 400 \
                -g 0