from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
import json
import trimesh
from util.pc_utils import sample_point_cloud_by_n
import random


SPLIT_DIR = "data/partnet_train_val_test_split"


def get_dataloader_partnet_scan(phase, config):
    is_shuffle = phase == 'train'

    if config.module == "gan":
        dataset = GANdatasetPartNetScan(phase, config.data_root, config.data_raw_root, config.category, config.n_pts)
    elif config.module == "ae" or config.module == "vae":
        dataset = AEdatasetPartNetScan(phase, config.data_root, config.category, config.n_pts)
    else:
        raise ValueError

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=is_shuffle, num_workers=config.num_workers,
                            worker_init_fn=np.random.seed())
    return dataloader


def collect_data_id(split_dir, classname, phase):
    filename = os.path.join(split_dir, "{}.{}.json".format(classname, phase))
    if not os.path.exists(filename):
        raise ValueError("Invalid filepath: {}".format(filename))

    all_ids = []
    with open(filename, 'r') as fp:
        info = json.load(fp)
    for item in info:
        all_ids.append(item["anno_id"])

    return all_ids


class AEdatasetPartNetScan(Dataset):
    def __init__(self, phase, data_root, category, n_pts):
        super(AEdatasetPartNetScan, self).__init__()
        if phase == "validation":
            phase = "val"

        self.aug = phase == "train"

        self.data_root = data_root

        shape_names = collect_data_id(SPLIT_DIR, category, phase)
        self.shape_names = []
        for name in shape_names:
            path = os.path.join(self.data_root, "{}.ply".format(name))
            if os.path.exists(path):
                self.shape_names.append(name)

        self.n_pts = n_pts

    @staticmethod
    def load_point_cloud(path):
        pc = trimesh.load(path)
        pc = pc.vertices / 2.0 # scale to unit sphere
        return pc

    def __getitem__(self, index):
        ply_path = os.path.join(self.data_root, "{}.ply".format(self.shape_names[index]))

        pc = self.load_point_cloud(ply_path)

        pc = sample_point_cloud_by_n(pc, self.n_pts)

        pc = torch.tensor(pc, dtype=torch.float32).transpose(1, 0)
        
        return {"points": pc, "id": self.shape_names[index]}
        
    def __len__(self):
        return len(self.shape_names)


class GANdatasetPartNetScan(Dataset):
    def __init__(self, phase, data_root, data_raw_root, category, n_pts):
        super(GANdatasetPartNetScan, self).__init__()
        if phase == "validation":
            phase = "val"

        self.phase = phase
        self.aug = phase == "train"

        self.data_root = data_root
        self.data_raw_root = data_raw_root

        shape_names = collect_data_id(SPLIT_DIR, category, phase)

        # check file existence
        self.shape_names = []
        self.raw_ply_names = []
        for name in shape_names:
            shape_path = os.path.join(self.data_root, "{}.ply".format(name))
            raw_path = os.path.join(self.data_raw_root, name, "{}-partial-0-0.ply".format(name))
            if os.path.exists(shape_path) and os.path.exists(raw_path):
                self.shape_names.append(name)

                partial_shape_dir = os.path.join(self.data_raw_root, name)
                raw_names = sorted(os.listdir(partial_shape_dir))
                self.raw_ply_names.extend(raw_names)

        self.n_pts = n_pts
        self.raw_n_pts = self.n_pts // 2

        self.rng = random.Random(1234)

    @staticmethod
    def load_point_cloud(path):
        pc = trimesh.load(path)
        pc = pc.vertices / 2.0 # scale to unit sphere
        return pc

    def __getitem__(self, index):
        if self.phase == "train":
            raw_n = random.randint(0, len(self.raw_ply_names) - 1)
            raw_name = self.raw_ply_names[raw_n]
        else:
            shape_name = self.shape_names[index]
            partial_shape_dir = os.path.join(self.data_raw_root, shape_name)
            raw_names = sorted(os.listdir(partial_shape_dir))
            raw_name = self.rng.choice(raw_names)
        raw_ply_path = os.path.join(self.data_raw_root, raw_name.split('-')[0], raw_name)

        raw_pc = self.load_point_cloud(raw_ply_path)
        raw_pc = sample_point_cloud_by_n(raw_pc, self.raw_n_pts)
        raw_pc = torch.tensor(raw_pc, dtype=torch.float32).transpose(1, 0)

        real_shape_name = self.shape_names[index]
        real_ply_path = os.path.join(self.data_root, '{}.ply'.format(real_shape_name))
        real_pc = self.load_point_cloud(real_ply_path)
        real_pc = sample_point_cloud_by_n(real_pc, self.n_pts)
        real_pc = torch.tensor(real_pc, dtype=torch.float32).transpose(1, 0)

        return {"raw": raw_pc, "real": real_pc, "raw_id": raw_ply_path.split('/')[-1], "real_id": real_shape_name}

    def __len__(self):
        return len(self.shape_names)


def test():
    pass


if __name__ == "__main__":
    test()
