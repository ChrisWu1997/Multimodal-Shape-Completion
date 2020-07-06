#!/usr/bin/env bash
python test.py --proj_dir proj_log \
                --exp_name mpc-partnet-table \
                --module gan \
                --dataset_name partnet \
                --category Table \
                --data_root /dev/data/partnet_data/partnet_original/PartNet \
                --pretrain_ae_path proj_log/mpc-partnet-table/ae/model/ckpt_epoch1000.pth \
                --pretrain_vae_path proj_log/mpc-partnet-table/vae/model/ckpt_epoch2000.pth \
                --num_sample -1 \
                --num_z 10 \
                --ckpt 400 \
                -g 0
