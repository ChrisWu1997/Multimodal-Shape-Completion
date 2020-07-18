#!/usr/bin/env bash
python test.py --proj_dir proj_log \
                --exp_name mpc-partnet_scan-lamp \
                --module gan \
                --dataset_name partnet_scan \
                --category Lamp \
                --data_root /mnt/data/partnet_data/shape_mesh_scan_pc \
                --data_raw_root /mnt/data/partnet_data/partial_mesh_scan_pc \
                --pretrain_ae_path proj_log/mpc-partnet_scan-lamp/ae/model/ckpt_epoch2000.pth \
                --pretrain_vae_path proj_log/mpc-partnet_scan-lamp/vae/model/ckpt_epoch2000.pth \
                --num_sample -1 \
                --num_z 10 \
                --ckpt 500 \
                -g 0