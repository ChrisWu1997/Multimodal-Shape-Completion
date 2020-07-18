# A simple script that uses blender to render views of a single object by rotation the camera around it.
# Also produces depth map at the same time.
#
# Example:
# /home/song/wurundi/blender-2.79-linux-glibc219-x86_64/blender --background --python generate_partnetpartial_point_cloud.py -- /mnt/data/partnet_data/partial_mesh/2460/partial-2.obj  --nb_scan 4 --nb_view 1 --orth_scale 2 --output_folder ./2460-partnetscan-360

# find /mnt/data/partnet_data/partial_mesh -name '*.obj' -print0 | xargs -0 -n1 -P10 -I {} /home/song/wurundi/blender-2.79-linux-glibc219-x86_64/blender --background --python generate_partnetpartial_point_cloud.py -- --nb_scan 4 --nb_view 1 --orth_scale 2 --output_folder ./tmp {}

import argparse, sys, os
import numpy as np

import bpy
from math import radians

import OpenEXR as exr
import Imath
import array

from PIL import Image
sys.path.append('.')
import util
import blender_camera_util
import blender_util

from scipy import spatial

from tqdm import tqdm
import trimesh
import random

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument('--reso', type=int, default=640,
                    help='resolution')
parser.add_argument('--nb_scan', type=int, default=4,
                    help='number of scan per model to render passes')
parser.add_argument('--nb_view', type=int, default=1,
                    help='number of views per model to render passes')
parser.add_argument('--orth_scale', type=int, default=2,
                    help='view scale of orthogonal camera')
parser.add_argument('obj', type=str,
                    help='Path to the obj file to be rendered.')
parser.add_argument('--output_folder', type=str, default='./tmp',
                    help='The path the output will be dumped to.')
parser.add_argument('--normalization_mode', type=str, default=None,
                    help='if scale the mesh to be within a unit sphere. [None | diag2sphere | unit_sphere | unit_cube]')
# usually fix below args
parser.add_argument('--remove_doubles', type=bool, default=True,
                    help='Remove double vertices to improve mesh quality.')
parser.add_argument('--remove_iso_verts', type=bool, default=True,
                    help='Remove isolated vertices.')
parser.add_argument('--edge_split', type=bool, default=True,
                    help='Adds edge split filter.')
parser.add_argument('--depth_scale', type=float, default=0.5,
                    help='Scaling that is applied to depth. Depends on size of mesh. Try out various values until you get a good result. Ignored if format is OPEN_EXR.')
parser.add_argument('--color_depth', type=str, default='16',
                    help='Number of bit per channel used for output. Either 8 or 16.')
parser.add_argument('--format', type=str, default='OPEN_EXR',
                    help='Format of files generated. Either PNG or OPEN_EXR')

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

def get_parnet_partial_name(path):
    model_id = path.split('/')[-2]
    partial_id = path.split('/')[-1].split('.')[0]
    name = model_id + '-' + partial_id
    return model_id, name

# cls_id, modelname = util.get_shapenet_clsID_modelname_from_filename(args.obj)
model_id, modelname = get_parnet_partial_name(args.obj)

blender_util.clear_scene_objects()
depth_file_output,normal_file_output,albedo_file_output,matidx_file_output = blender_util.rendering_pass_setup(args)

# shapenet v2 coordinate system: Y - up, -Z - face
# after imported to blender, the up of the object will be the Z axis in blender world...
bpy.ops.import_scene.obj(filepath=args.obj, use_smooth_groups=False, use_split_objects=False, use_split_groups=False)
blender_util.process_scene_objects(args) # including normalization

# assign each material a unique id
# disable transparency for all materials
for i, mat in enumerate(bpy.data.materials):
  if mat.name in ['Material']: continue
  mat.pass_index = i
  mat.use_transparency  = False

# setup camera resolution etc
blender_util.setup_render(args)
scene = bpy.context.scene

for j in range(args.nb_scan):
    # generate random camera rotations
    rot_angles_list = []
    for i in range(args.nb_view):
        rot_x_angle = random.randint(0, 360) # camera elevation angle
        rot_y_angle = 0 # do not rot around y, no in-plane rotation
        rot_z_angle = random.randint(0, 360) # camera azimuth angle
        rot_angles_list.append([rot_x_angle, rot_y_angle, rot_z_angle])

    # render shapenet shape to get color point cloud
    all_points_normals_colors_mindices = blender_util.scan_point_cloud(depth_file_output, normal_file_output, albedo_file_output, matidx_file_output, args, rot_angles_list)
    all_points_normals_colors_mindices = util.sample_from_point_cloud(all_points_normals_colors_mindices, 10000)

    output_shape_dir = os.path.join(args.output_folder, model_id)
    if not os.path.exists(output_shape_dir):
        os.makedirs(output_shape_dir)
    util.write_ply(all_points_normals_colors_mindices[:, :3], os.path.join(output_shape_dir, modelname+'-{}.ply'.format(j)), colors=all_points_normals_colors_mindices[:, 6:9], normals=all_points_normals_colors_mindices[:, 3:6])
    print('PartNet partial point cloud scanning done!')

# clear the objects imported previously
blender_util.clear_scene_objects()
