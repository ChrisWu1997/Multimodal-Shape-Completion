import os
import json
import numpy as np
from tree import TreeNode
from joblib import Parallel, delayed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str, required=True, help="path to PartNet dataset")
args = parser.parse_args()

SRC_ROOT = args.src
DEST_ROOT = "../partnet_pc_label"

STAT_DIR = "../partnet_train_val_test_split"
CLASS_NAMES = ["Lamp", "Table", "Chair"]
MERGE_DEPTHS = [1, 2, 1]
# NON_MERGE_PART = [] #["Leg"]



def read_point_cloud_part_label(path):
    with open(path, 'r') as fp:
        labels = fp.readlines()
    labels = [int(x) for x in labels]
    return labels


def write_point_cloud_part_label(path, labels:list):
    with open(path, 'w') as fp:
        fp.writelines(labels)


def process_one(data_id, merge_depth, class_name):
    hier_path = os.path.join(SRC_ROOT, "{}/result.json".format(data_id))
    pc_label_path = os.path.join(SRC_ROOT, "{}/point_sample/label-10000.txt".format(data_id))

    root = TreeNode.from_json(hier_path)

    depth = merge_depth
    first_level_id = root.query_id_by_depth(depth=depth)
    while len(first_level_id) <= 1 and depth <=4:
        depth += 1
        first_level_id = root.query_id_by_depth(depth=depth)

    if len(first_level_id) <= 1:
        print("Less than 1 part! ID: {}".format(data_id))
        return

    pc_label_ori = read_point_cloud_part_label(pc_label_path)
    ori_label_ids = list(np.unique(pc_label_ori))

    id_map_dict = {}
    for i in ori_label_ids:
        node = root.query_node_by_id(i)
        par_id = node.query_parent_id(depth=depth)
        id_map_dict.update({str(i): par_id})

    pc_label_merge = [id_map_dict[str(i)] for i in pc_label_ori]
    pc_label_merge_str = [str(i) + '\n' for i in pc_label_merge]

    save_dir = os.path.join(DEST_ROOT, "{}".format(data_id))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "label-merge-level1-10000.txt")
    write_point_cloud_part_label(save_path, pc_label_merge_str)



for i, class_name in enumerate(CLASS_NAMES):
    merge_depth = MERGE_DEPTHS[i]

    all_ids = []
    for phase in ['train', 'val', 'test']:
        filename = os.path.join(STAT_DIR, "{}.{}.json".format(class_name, phase))

        with open(filename, 'r') as fp:
            info = json.load(fp)

        all_ids.extend([item["anno_id"] for item in info])

    Parallel(n_jobs=20, verbose=2)(delayed(process_one)(iid, merge_depth, class_name) for iid in all_ids)
