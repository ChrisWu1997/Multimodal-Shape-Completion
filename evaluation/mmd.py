import sys
sys.path.append("..")
import tensorflow as tf
import argparse
import os
import numpy as np
from tqdm import tqdm
import time
import trimesh
import random
from util.pc_utils import sample_point_cloud_by_n
import glob
import csv
import json

try:
    from sklearn.neighbors import NearestNeighbors
except:
    print ('Sklearn module not installed (JSD metric will not work).')
    exit()

try:
    from external.structural_losses.tf_nndistance import nn_distance
    from external.structural_losses.tf_approxmatch import approx_match, match_cost
except:
    print('External Losses (Chamfer-EMD) cannot be loaded. Please install them first.')
    exit()


ShapeNet_PC_DATA_ROOT = "../data/ShapeNetPointCloud"
EPN_DATA_ROOT = "../data/shapenet_dim32_sdf_pc"
ShapeNet_SPLIT_CSV_PATH = "../data/shapenet-official-split.csv"

PartNet_DATA_ROOT = "/dev/data/partnet_data/partnet_original/PartNet"
PartNet_SPLIT_DIR = "../data/partnet_train_val_test_split"

PartNet_Scan_DATA_ROOT = "/dev/data/partnet_data/shape_mesh_scan_pc"

NUM_PTS = 2048

random.seed(1234)

snc_synth_category_to_id = {
    'airplane': '02691156', 'bag': '02773838', 'basket': '02801938',
    'bathtub': '02808440', 'bed': '02818832', 'bench': '02828884',
    'bicycle': '02834778', 'birdhouse': '02843684', 'bookshelf': '02871439',
    'bottle': '02876657', 'bowl': '02880940', 'bus': '02924116',
    'cabinet': '02933112', 'can': '02747177', 'camera': '02942699',
    'cap': '02954340', 'car': '02958343', 'chair': '03001627',
    'clock': '03046257', 'dishwasher': '03207941', 'monitor': '03211117',
    'table': '04379243', 'telephone': '04401088', 'tin_can': '02946921',
    'tower': '04460130', 'train': '04468005', 'keyboard': '03085013',
    'earphone': '03261776', 'faucet': '03325088', 'file': '03337140',
    'guitar': '03467517', 'helmet': '03513137', 'jar': '03593526',
    'knife': '03624134', 'lamp': '03636649', 'laptop': '03642806',
    'speaker': '03691459', 'mailbox': '03710193', 'microphone': '03759954',
    'microwave': '03761084', 'motorcycle': '03790512', 'mug': '03797390',
    'piano': '03928116', 'pillow': '03938244', 'pistol': '03948459',
    'pot': '03991062', 'printer': '04004475', 'remote_control': '04074963',
    'rifle': '04090263', 'rocket': '04099429', 'skateboard': '04225987',
    'sofa': '04256520', 'stove': '04330267', 'vessel': '04530566',
    'washer': '04554684', 'boat': '02858304', 'cellphone': '02992529'
}


def collect_shapenet_split_by_id(class_name):
    class_id = snc_synth_category_to_id[class_name]
    split_info = {"train":[], 'validation':[], 'test':[]}
    with open(ShapeNet_SPLIT_CSV_PATH, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_cnt = 0
        for row in csv_reader:
            if line_cnt == 0 or row[1] != class_id:
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


def collect_partnet_split_by_id(split_dir, classname, phase):
    filename = os.path.join(split_dir, "{}.{}.json".format(classname, phase))
    if not os.path.exists(filename):
        raise ValueError("Invalid filepath: {}".format(filename))

    all_ids = []
    with open(filename, 'r') as fp:
        info = json.load(fp)
    for item in info:
        all_ids.append(item["anno_id"])

    return all_ids


def iterate_in_chunks(l, n):
    '''Yield successive 'n'-sized chunks from iterable 'l'.
    Note: last chunk will be smaller than l if n doesn't divide l perfectly.
    '''
    for i in range(0, len(l), n):
        yield l[i:i + n]


def unit_cube_grid_point_cloud(resolution, clip_sphere=False):
    '''Returns the center coordinates of each cell of a 3D grid with resolution^3 cells,
    that is placed in the unit-cube.
    If clip_sphere it True it drops the "corner" cells that lie outside the unit-sphere.
    '''
    grid = np.ndarray((resolution, resolution, resolution, 3), np.float32)
    spacing = 1.0 / float(resolution - 1)
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                grid[i, j, k, 0] = i * spacing - 0.5
                grid[i, j, k, 1] = j * spacing - 0.5
                grid[i, j, k, 2] = k * spacing - 0.5

    if clip_sphere:
        grid = grid.reshape(-1, 3)
        grid = grid[np.linalg.norm(grid, axis=1) <= 0.5]

    return grid, spacing


def scale_to_unit_sphere(points, center=None):
    """
    scale point clouds into a unit sphere
    :param points: (n, 3) numpy array
    :return:
    """
    if center is None:
        midpoints = (np.max(points, axis=0) + np.min(points, axis=0)) / 2
    else:
        midpoints = np.asarray(center)
    points = points - midpoints
    scale = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    points = points / scale
    return points


def minimum_mathing_distance_tf_graph(n_pc_points, batch_size=None, normalize=True, sess=None, verbose=False, use_sqrt=False, use_EMD=False):
    ''' Produces the graph operations necessary to compute the MMD and consequently also the Coverage due to their 'symmetric' nature.
    Assuming a "reference" and a "sample" set of point-clouds that will be matched, this function creates the operation that matches
    a _single_ "reference" point-cloud to all the "sample" point-clouds given in a batch. Thus, is the building block of the function
    ```minimum_mathing_distance`` and ```coverage``` that iterate over the "sample" batches and each "reference" point-cloud.
    Args:
        n_pc_points (int): how many points each point-cloud of those to be compared has.
        batch_size (optional, int): if the iterator code that uses this function will
            use a constant batch size for iterating the sample point-clouds you can
            specify it hear to speed up the compute. Alternatively, the code is adapted
            to read the batch size dynamically.
        normalize (boolean): if True, the matched distances are normalized by diving them with
            the number of points of the compared point-clouds (n_pc_points).
        use_sqrt (boolean): When the matching is based on Chamfer (default behavior), if True,
            the Chamfer is computed based on the (not-squared) euclidean distances of the
            matched point-wise euclidean distances.
        use_EMD (boolean): If true, the matchings are based on the EMD.
    '''
    if normalize:
        reducer = tf.reduce_mean
    else:
        reducer = tf.reduce_sum

    if sess is None:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

    # Placeholders for the point-clouds: 1 for the reference (usually Ground-truth) and one of variable size for the collection
    # which is going to be matched with the reference.
    ref_pl = tf.placeholder(tf.float32, shape=(1, n_pc_points, 3))
    sample_pl = tf.placeholder(tf.float32, shape=(batch_size, n_pc_points, 3))

    if batch_size is None:
        batch_size = tf.shape(sample_pl)[0]

    ref_repeat = tf.tile(ref_pl, [batch_size, 1, 1])
    ref_repeat = tf.reshape(ref_repeat, [batch_size, n_pc_points, 3])

    if use_EMD:
        match = approx_match(ref_repeat, sample_pl)
        all_dist_in_batch = match_cost(ref_repeat, sample_pl, match)
        if normalize:
            all_dist_in_batch /= n_pc_points
    else:
        ref_to_s, _, s_to_ref, _ = nn_distance(ref_repeat, sample_pl)
        if use_sqrt:
            ref_to_s = tf.sqrt(ref_to_s)
            s_to_ref = tf.sqrt(s_to_ref)
        all_dist_in_batch = reducer(ref_to_s, 1) + reducer(s_to_ref, 1)

    best_in_batch = tf.reduce_min(all_dist_in_batch)   # Best distance, of those that were matched to single ref pc.
    location_of_best = tf.argmin(all_dist_in_batch, axis=0)
    return ref_pl, sample_pl, best_in_batch, location_of_best, sess


def minimum_mathing_distance(sample_pcs, ref_pcs, batch_size, normalize=True, sess=None, verbose=False, use_sqrt=False, use_EMD=False):
    '''Computes the MMD between two sets of point-clouds.
    Args:
        sample_pcs (numpy array SxKx3): the S point-clouds, each of K points that will be matched and
            compared to a set of "reference" point-clouds.
        ref_pcs (numpy array RxKx3): the R point-clouds, each of K points that constitute the set of
            "reference" point-clouds.
        batch_size (int): specifies how large will the batches be that the compute will use to make
            the comparisons of the sample-vs-ref point-clouds.
        normalize (boolean): if True, the distances are normalized by diving them with
            the number of the points of the point-clouds (n_pc_points).
        use_sqrt: (boolean): When the matching is based on Chamfer (default behavior), if True, the
            Chamfer is computed based on the (not-squared) euclidean distances of the matched point-wise
             euclidean distances.
        sess (tf.Session, default None): if None, it will make a new Session for this.
        use_EMD (boolean: If true, the matchings are based on the EMD.
    Returns:
        A tuple containing the MMD and all the matched distances of which the MMD is their mean.
    '''

    n_ref, n_pc_points, pc_dim = ref_pcs.shape
    _, n_pc_points_s, pc_dim_s = sample_pcs.shape

    if n_pc_points != n_pc_points_s or pc_dim != pc_dim_s:
        raise ValueError('Incompatible size of point-clouds.')

    ref_pl, sample_pl, best_in_batch, _, sess = minimum_mathing_distance_tf_graph(n_pc_points, normalize=normalize,
                                                                                  sess=sess, use_sqrt=use_sqrt,
                                                                                  use_EMD=use_EMD)
    matched_dists = []
    pbar = tqdm(range(n_ref))
    for i in pbar:
        best_in_all_batches = []
        for sample_chunk in iterate_in_chunks(sample_pcs, batch_size):
            feed_dict = {ref_pl: np.expand_dims(ref_pcs[i], 0), sample_pl: sample_chunk}
            b = sess.run(best_in_batch, feed_dict=feed_dict)
            best_in_all_batches.append(b)
        matched_dists.append(np.min(best_in_all_batches))

        pbar.set_postfix({"mmd": np.mean(matched_dists)})

    mmd = np.mean(matched_dists)
    return mmd, matched_dists


def collect_shapenet_test_set_pcs(args):
    """collect shapenet test set: complete point clouds """
    start = time.time()
    d = collect_shapenet_split_by_id(args.class_name)
    class_id = snc_synth_category_to_id[args.class_name]
    shape_names_all = d["test"]
    shape_names = []
    # remove shapes not in 3depn
    for name in shape_names_all:
        path = os.path.join(EPN_DATA_ROOT, class_id, "{}__0__.ply".format(name))
        if os.path.exists(path):
            shape_names.append(name)

    if not args.n_used_test == -1:
        shape_names = shape_names[args.n_used_test:]

    ref_pcs = []
    for name in shape_names:
        src_pts_path = os.path.join(ShapeNet_PC_DATA_ROOT, class_id, "{}.ply".format(name))
        target_ply = trimesh.load(src_pts_path)
        target_pts = target_ply.vertices

        target_pts = sample_point_cloud_by_n(target_pts, NUM_PTS)

        ref_pcs.append(target_pts)
    ref_pcs = np.stack(ref_pcs, axis=0)
    print("reference point clouds: {}".format(ref_pcs.shape))
    print("time: {:.2f}s".format(time.time() - start))
    return ref_pcs


def collect_partnet_test_set_pcs(args):
    """collect PartNet test set: complete point clouds """
    start = time.time()
    shape_names = collect_partnet_split_by_id(PartNet_SPLIT_DIR, args.class_name, "test")

    if not args.n_used_test == -1:
        shape_names = shape_names[args.n_used_test:]

    ref_pcs = []
    for name in shape_names:
        src_pts_path = os.path.join(PartNet_DATA_ROOT, name, "point_sample/ply-10000.ply")
        target_ply = trimesh.load(src_pts_path)
        target_pts = target_ply.vertices / 2.0

        target_pts = sample_point_cloud_by_n(target_pts, NUM_PTS)

        ref_pcs.append(target_pts)
    ref_pcs = np.stack(ref_pcs, axis=0)
    print("reference point clouds: {}".format(ref_pcs.shape))
    print("time: {:.2f}s".format(time.time() - start))
    return ref_pcs


def collect_partnetscan_test_set_pcs(args):
    """collect PartNet-scan test set: complete point clouds """
    start = time.time()
    shape_names = collect_partnet_split_by_id(PartNet_SPLIT_DIR, args.class_name, "test")

    if not args.n_used_test == -1:
        shape_names = shape_names[args.n_used_test:]

    ref_pcs = []
    for name in shape_names:
        src_pts_path = os.path.join(PartNet_Scan_DATA_ROOT, name + ".ply")
        target_ply = trimesh.load(src_pts_path)
        target_pts = target_ply.vertices / 2.0

        target_pts = sample_point_cloud_by_n(target_pts, NUM_PTS)

        ref_pcs.append(target_pts)
    ref_pcs = np.stack(ref_pcs, axis=0)
    print("reference point clouds: {}".format(ref_pcs.shape))
    print("time: {:.2f}s".format(time.time() - start))
    return ref_pcs


def collect_src_pcs(args):
    """collect completion results: generated point clouds"""
    start = time.time()

    all_paths = glob.glob(os.path.join(args.src, "*/fake-z*.ply"))

    gen_pcs = []
    for path in all_paths:
        sample_pts = trimesh.load(path)
        sample_pts = sample_pts.vertices
        sample_pts = sample_point_cloud_by_n(sample_pts, NUM_PTS)
        gen_pcs.append(sample_pts)

    gen_pcs = np.stack(gen_pcs, axis=0)
    print("generated point clouds: {}".format(gen_pcs.shape))
    print("time: {:.2f}s".format(time.time() - start))
    return gen_pcs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help='path to completion results')
    parser.add_argument("--dataset", type=str, choices=['3depn', 'partnet', 'partnet_scan'], required=True)
    parser.add_argument('-g', '--gpu_ids', type=str, default=None, help="gpu to use, e.g. 0  0,1,2. CPU not supported.")
    parser.add_argument("--class_name", type=str)
    parser.add_argument("--n_used_test", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("-o", "--output", type=str)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)

    if args.output is None:
        args.output = args.src + '-eval_mmd_pts{}.txt'.format(NUM_PTS)

    if args.dataset == '3depn':
        ref_pcs = collect_shapenet_test_set_pcs(args)
    elif args.dataset == 'partnet':
        ref_pcs = collect_partnet_test_set_pcs(args)
    elif args.dataset == 'partnet_scan':
        ref_pcs = collect_partnetscan_test_set_pcs(args)
    else:
        raise NotImplementedError

    sample_pcs = collect_src_pcs(args)

    mmd, matched_dists = minimum_mathing_distance(sample_pcs, ref_pcs, args.batch_size)
    print("Minimum Mathing Distance: {}".format(mmd))

    with open(args.output, "w") as fp:
        fp.write("SRC: {}\n".format(args.src))
        fp.write("MMD-CD: {}\n".format(mmd))


if __name__ == '__main__':
    main()
