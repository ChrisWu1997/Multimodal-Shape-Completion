#!/usr/bin/env bash
python test.py --proj_dir proj_log \
                --exp_name mpc-partnet-lamp \
                --module gan \
                --dataset_name partnet \
                --category Lamp \
                --data_root /dev/data/partnet_data/partnet_original/PartNet \
                --pretrain_ae_path proj_log/mpc-partnet-lamp/ae/model/ckpt_epoch1000.pth \
                --pretrain_vae_path proj_log/mpc-partnet-lamp/vae/model/ckpt_epoch1000.pth \
                --num_sample -1 \
                --num_z 10 \
                --ckpt 500 \
                -g 0