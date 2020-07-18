import os
import json
import trimesh
import numpy as np
from tree import TreeNode
from joblib import Parallel, delayed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str, required=True, help="path to PartNet dataset")
parser.add_argument("--dest", type=str, required=True, help="folder to save partial and complete mesh")
args = parser.parse_args()

SRC_ROOT = args.src
DEST_ROOT = args.dest + "-partial_mesh"
WHOLE_SHAPE_ROOT = args.dest_c + "-shape_mesh"

RNG = np.random.RandomState(1234)
N_PARTIAL = 4 # number of partial shapes generated for each complete shape

STAT_DIR = "../partnet_train_val_test_split"
CLASS_NAMES = ["Lamp", "Table", "Chair"]
SEG_DEPTHS = [1, 2, 1]


def ensure_dir(path):
    """
    create path by first checking its existence,
    :param paths: path
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


def process_one(data_id, seg_depth):
    data_dir = os.path.join(SRC_ROOT, str(data_id))
    hier_path = os.path.join(data_dir, "result.json")

    root = TreeNode.from_json(hier_path)

    depth = seg_depth
    first_level_id = root.query_id_by_depth(depth=depth)
    while len(first_level_id) <= 1 and depth <=4:
        depth += 1
        first_level_id = root.query_id_by_depth(depth=depth)
    n_parts = len(first_level_id)

    if len(first_level_id) <= 0:
        print("no part found! ID: {}".format(data_id))
        return

    # read part meshes
    parts_mesh = []
    for node_id in first_level_id:
        node = root.query_node_by_id(node_id)
        node_objs = node.collect_objs()
        node_mesh = []
        for name in node_objs:
            obj_path = os.path.join(data_dir, "objs/{}.obj".format(name))
            mesh = trimesh.load(obj_path)
            node_mesh.append(mesh)
        node_mesh = trimesh.util.concatenate(node_mesh)
        parts_mesh.append(node_mesh)

    # save shape mesh
    shape_mesh = trimesh.util.concatenate(parts_mesh)
    save_path = os.path.join(WHOLE_SHAPE_ROOT, "{}.obj".format(data_id))
    shape_mesh.export(save_path)
    del shape_mesh

    # random delete parts and save partial mesh
    lst = list(range(1, n_parts))
    if len(lst) == 0:
        print("only one part! ID: {}.".format(data_id))
        return
    lst_weight = (np.array(lst) / np.sum(lst)).tolist()

    save_dir = os.path.join(DEST_ROOT, str(data_id))
    ensure_dir(save_dir)

    save_info = []
    for i in range(N_PARTIAL):
        n_keep = RNG.choice(lst, p=lst_weight)

        parts_keep = RNG.choice(list(range(n_parts)), n_keep, replace=False)
        save_info.append(parts_keep)

        partial_mesh = [parts_mesh[i] for i in parts_keep]
        partial_mesh = trimesh.util.concatenate(partial_mesh)

        save_path = os.path.join(save_dir, "partial-{}.obj".format(i))
        partial_mesh.export(save_path)
    save_path = os.path.join(save_dir, "info.txt")
    with open(save_path, 'w') as fp:
        for line in save_info:
            print("{}".format(line), file=fp)


ensure_dir(DEST_ROOT)
ensure_dir(WHOLE_SHAPE_ROOT)


for i, class_name in enumerate(CLASS_NAMES):
    seg_depth = SEG_DEPTHS[i]

    all_ids = []
    for phase in ['train', 'val', 'test']:
        filename = os.path.join(STAT_DIR, "{}.{}.json".format(class_name, phase))

        with open(filename, 'r') as fp:
            info = json.load(fp)

        all_ids.extend([item["anno_id"] for item in info])

    Parallel(n_jobs=20, verbose=2)(delayed(process_one)(iid, seg_depth) for iid in all_ids)
