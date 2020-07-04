from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
import random
import trimesh
import csv
from util.pc_utils import rotate_point_cloud_by_axis_angle, sample_point_cloud_by_n


SPLIT_CSV_PATH = "data/shapenet-official-split.csv"


def get_dataloader_3depn(phase, config):
    is_shuffle = phase == 'train'

    if config.module == "gan":
        dataset = GANdataset3DEPN(phase, config.data_root, config.data_raw_root, config.category, config.n_pts)
    elif config.module == "ae" or config.module == "vae":
        dataset = AEdataset3DEPN(phase, config.data_root, config.category, config.n_pts)
    else:
        raise ValueError
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=is_shuffle, num_workers=config.num_workers,
                            worker_init_fn=np.random.seed())
    return dataloader


snc_synth_category_to_id = {
    'airplane' : '02691156' ,  'bag'       : '02773838' ,  'basket'        : '02801938' ,
    'bathtub'  : '02808440' ,  'bed'       : '02818832' ,  'bench'         : '02828884' ,
    'bicycle'  : '02834778' ,  'birdhouse' : '02843684' ,  'bookshelf'     : '02871439' ,
    'bottle'   : '02876657' ,  'bowl'      : '02880940' ,  'bus'           : '02924116' ,
    'cabinet'  : '02933112' ,  'can'       : '02747177' ,  'camera'        : '02942699' ,
    'cap'      : '02954340' ,  'car'       : '02958343' ,  'chair'         : '03001627' ,
    'clock'    : '03046257' ,  'dishwasher': '03207941' ,  'monitor'       : '03211117' ,
    'table'    : '04379243' ,  'telephone' : '04401088' ,  'tin_can'       : '02946921' ,
    'tower'    : '04460130' ,  'train'     : '04468005' ,  'keyboard'      : '03085013' ,
    'earphone' : '03261776' ,  'faucet'    : '03325088' ,  'file'          : '03337140' ,
    'guitar'   : '03467517' ,  'helmet'    : '03513137' ,  'jar'           : '03593526' ,
    'knife'    : '03624134' ,  'lamp'      : '03636649' ,  'laptop'        : '03642806' ,
    'speaker'  : '03691459' ,  'mailbox'   : '03710193' ,  'microphone'    : '03759954' ,
    'microwave': '03761084' ,  'motorcycle': '03790512' ,  'mug'           : '03797390' ,
    'piano'    : '03928116' ,  'pillow'    : '03938244' ,  'pistol'        : '03948459' ,
    'pot'      : '03991062' ,  'printer'   : '04004475' ,  'remote_control': '04074963' ,
    'rifle'    : '04090263' ,  'rocket'    : '04099429' ,  'skateboard'    : '04225987' ,
    'sofa'     : '04256520' ,  'stove'     : '04330267' ,  'vessel'        : '04530566' ,
    'washer'   : '04554684' ,  'boat'      : '02858304' ,  'cellphone'     : '02992529'
}


def collect_train_split_by_id(path, cat_id):
    split_info = {"train":[], 'validation':[], 'test':[]}
    with open(path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_cnt = 0
        for row in csv_reader:
            if line_cnt == 0 or row[1] != cat_id:
                pass
            else:
                if row[-1] == "train":
                    split_info["train"].append(row[-2])
                elif row[-1] == "val":
                    split_info["validation"].append(row[-2])
                else:
                    split_info["test"].append(row[-2])
            line_cnt += 1
    return split_info


class AEdataset3DEPN(Dataset):
    def __init__(self, phase, data_root, category, n_pts):
        super(AEdataset3DEPN, self).__init__()
        self.aug = phase == "train"

        self.cat_id = snc_synth_category_to_id[category]
        self.cat_pc_root = os.path.join(data_root, self.cat_id)

        # split_csv_path = os.path.join(os.path.dirname(data_root), "shapenet-official-split.csv")
        shape_names = collect_train_split_by_id(SPLIT_CSV_PATH, self.cat_id)[phase]

        # ensure file existence
        self.shape_names = []
        for name in shape_names:
            ply_path = os.path.join(self.cat_pc_root, name + '.ply')
            if os.path.exists(ply_path):
                self.shape_names.append(name)

        self.n_pts = n_pts

    def __getitem__(self, index):
        ply_path = os.path.join(self.cat_pc_root, self.shape_names[index] + '.ply')

        pc = trimesh.load(ply_path)
        p_idx = list(range(pc.shape[0]))
        random.shuffle(p_idx)
        p_idx = p_idx[:self.n_pts]
        # p_idx = random.choices(list(range(pc.shape[0])), k=self.n_pts)

        pc = pc[p_idx]
        pc = torch.tensor(pc, dtype=torch.float32).transpose(1, 0)
        
        return {"points": pc, "id": self.shape_names[index]}
        
    def __len__(self):
        return len(self.shape_names)


class GANdataset3DEPN(Dataset):
    def __init__(self, phase, data_root, data_raw_root, category, n_pts):
        super(GANdataset3DEPN, self).__init__()
        self.phase = phase
        self.aug = phase == "train"

        self.cat_id = snc_synth_category_to_id[category]
        self.cat_pc_root = os.path.join(data_root, self.cat_id)
        self.cat_pc_raw_root = os.path.join(data_raw_root, self.cat_id)

        # split_csv_path = os.path.join(os.path.dirname(data_root), "shapenet-official-split.csv")
        shape_names = collect_train_split_by_id(SPLIT_CSV_PATH, self.cat_id)[phase]

        self.raw_ply_names = sorted(os.listdir(self.cat_pc_raw_root))

        # ensure file existence
        self.shape_names = []
        for name in shape_names:
            ply_path = os.path.join(self.cat_pc_root, name + '.ply')
            path = os.path.join(self.cat_pc_raw_root, "{}__0__.ply".format(name))
            if os.path.exists(ply_path) and os.path.exists(path):
                self.shape_names.append(name)

        self.n_pts = n_pts
        self.raw_n_pts = self.n_pts // 2

        self.rng = random.Random(1234)

    def __getitem__(self, index):
        if self.phase == "train":
            raw_n = random.randint(0, len(self.raw_ply_names) - 1)
            raw_pc_name = self.raw_ply_names[raw_n]
        else:
            raw_n = self.rng.randint(0, 7)
            raw_pc_name = self.shape_names[index] + "__{}__.ply".format(raw_n)
        # process partial shapes
        raw_ply_path = os.path.join(self.cat_pc_raw_root, raw_pc_name)
        raw_pc = np.array(trimesh.load(raw_ply_path).vertices)
        raw_pc = rotate_point_cloud_by_axis_angle(raw_pc, [0,1,0], 90)
        raw_pc = sample_point_cloud_by_n(raw_pc, self.raw_n_pts)
        raw_pc = torch.tensor(raw_pc, dtype=torch.float32).transpose(1, 0)

        # process real complete shapes
        real_shape_name = self.shape_names[index]
        real_ply_path = os.path.join(self.cat_pc_root, real_shape_name + '.ply')
        real_pc = np.array(trimesh.load(real_ply_path).vertices)
        real_pc = sample_point_cloud_by_n(real_pc, self.n_pts)
        real_pc = torch.tensor(real_pc, dtype=torch.float32).transpose(1, 0)

        return {"raw": raw_pc, "real": real_pc, "raw_id": raw_pc_name, "real_id": real_shape_name}

    def __len__(self):
        return len(self.shape_names)


if __name__ == "__main__":
    pass
