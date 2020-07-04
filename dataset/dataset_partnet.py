from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
import json
import random
import trimesh
from util.pc_utils import sample_point_cloud_by_n


SPLIT_DIR = "data/partnet_train_val_test_split"
PC_MERGED_LABEL_DIR = "data/partnet_pc_label"


def get_dataloader_partnet(phase, config):
    is_shuffle = phase == 'train'

    if config.module == "gan":
        dataset = GANdatasetPartNet(phase, config.data_root, config.category, config.n_pts)
    elif config.module == "ae" or config.module == "vae":
        dataset = AEdatasetPartNet(phase, config.data_root, config.category, config.n_pts)
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


class AEdatasetPartNet(Dataset):
    def __init__(self, phase, data_root, category, n_pts):
        super(AEdatasetPartNet, self).__init__()
        if phase == "validation":
            phase = "val"

        self.aug = phase == "train"

        self.data_root = data_root

        self.shape_names = collect_data_id(SPLIT_DIR, category, phase)

        self.n_pts = n_pts

    @staticmethod
    def load_point_cloud(path):
        pc = trimesh.load(path)
        pc = pc.vertices / 2.0 # scale to unit sphere
        return pc

    def __getitem__(self, index):
        ply_path = os.path.join(self.data_root, self.shape_names[index], 'point_sample/ply-10000.ply')

        pc = self.load_point_cloud(ply_path)

        pc = sample_point_cloud_by_n(pc, self.n_pts)

        pc = torch.tensor(pc, dtype=torch.float32).transpose(1, 0)

        return {"points": pc, "id": self.shape_names[index]}

    def __len__(self):
        return len(self.shape_names)


class GANdatasetPartNet(Dataset):
    def __init__(self, phase, data_root, category, n_pts):
        super(GANdatasetPartNet, self).__init__()
        if phase == "validation":
            phase = "val"

        self.phase = phase
        self.aug = phase == "train"

        self.data_root = data_root

        self.shape_names = collect_data_id(SPLIT_DIR, category, phase)

        self.n_pts = n_pts
        self.raw_n_pts = self.n_pts // 2

        self.rng = random.Random(1234)

    @staticmethod
    def load_point_cloud(path):
        pc = trimesh.load(path)
        pc = pc.vertices / 2.0 # scale to unit sphere
        return pc

    @staticmethod
    def read_point_cloud_part_label(path):
        with open(path, 'r') as fp:
            labels = fp.readlines()
        labels = np.array([int(x) for x in labels])
        return labels

    def random_rm_parts(self, raw_pc, part_labels):
        part_ids = sorted(np.unique(part_labels).tolist())
        if self.phase == "train":
            random.shuffle(part_ids)
            n_part_keep = random.randint(1, max(1, len(part_ids) - 1))
        else:
            self.rng.shuffle(part_ids)
            n_part_keep = self.rng.randint(1, max(1, len(part_ids) - 1))
        part_ids_keep = part_ids[:n_part_keep]
        point_idx = []
        for i in part_ids_keep:
            point_idx.extend(np.where(part_labels == i)[0].tolist())
        raw_pc = raw_pc[point_idx]
        return raw_pc, n_part_keep

    def __getitem__(self, index):
        raw_shape_name = self.shape_names[index]
        raw_ply_path = os.path.join(self.data_root, raw_shape_name, 'point_sample/ply-10000.ply')
        raw_pc = self.load_point_cloud(raw_ply_path)

        raw_label_path = os.path.join(PC_MERGED_LABEL_DIR, raw_shape_name, 'label-merge-level1-10000.txt')
        part_labels = self.read_point_cloud_part_label(raw_label_path)
        raw_pc, n_part_keep = self.random_rm_parts(raw_pc, part_labels)
        raw_pc = sample_point_cloud_by_n(raw_pc, self.raw_n_pts)
        raw_pc = torch.tensor(raw_pc, dtype=torch.float32).transpose(1, 0)

        real_shape_name = self.shape_names[random.randint(0, len(self.shape_names) - 1)]
        real_ply_path = os.path.join(self.data_root, real_shape_name, 'point_sample/ply-10000.ply')
        real_pc = self.load_point_cloud(real_ply_path)
        real_pc = sample_point_cloud_by_n(real_pc, self.n_pts)
        real_pc = torch.tensor(real_pc, dtype=torch.float32).transpose(1, 0)

        return {"raw": raw_pc, "real": real_pc, "raw_id": raw_shape_name, "real_id": real_shape_name,
                'n_part_keep': n_part_keep}

    def __len__(self):
        return len(self.shape_names)


if __name__ == "__main__":
    pass
