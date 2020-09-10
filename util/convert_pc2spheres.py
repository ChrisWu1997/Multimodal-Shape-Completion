import os
import trimesh
from plyfile import PlyData, PlyElement
import numpy as np
from tqdm import tqdm
import argparse


def read_ply_xyz(filename):
    """ read XYZ point cloud from filename PLY file """
    assert(os.path.isfile(filename))
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
    return vertices


def write_obj(points, faces, filename):
    with open(filename, 'w') as F:
        for p in points:
            F.write('v %f %f %f\n'%(p[0], p[1], p[2]))
        
        for f in faces:
            F.write('f %d %d %d\n'%(f[0], f[1], f[2]))


def convert_point_cloud_to_balls(pc_ply_filename):
    if not pc_ply_filename.endswith('.ply'):
        return

    shape_id = pc_ply_filename.split('/')[-2]
    ply_name = pc_ply_filename.split('/')[-1]
    if 'raw.ply' == ply_name:
        sphere_r = 0.012
    else:
        if 'airplane' in pc_ply_filename:
            sphere_r = 0.012
        else:
            sphere_r = 0.016

    output_path = os.path.dirname(os.path.dirname(pc_ply_filename)) + '_spheres'
    out_shape_dir = os.path.join(output_path, shape_id)
    if not os.path.exists(out_shape_dir):
        os.makedirs(out_shape_dir)
    output_filename = os.path.join(out_shape_dir, ply_name[:-3] + '_{:.4f}.obj'.format(sphere_r))
    if os.path.exists(output_filename):
        return

    pc = read_ply_xyz(pc_ply_filename)

    points = []
    faces = []

    for pts in (pc):
        sphere_m = trimesh.creation.uv_sphere(radius=sphere_r, count=[8,8])
        sphere_m.apply_translation(pts)

        faces_offset = np.array(sphere_m.faces) + len(points)
        faces.extend(faces_offset)
        points.extend(np.array(sphere_m.vertices))
    
    points = np.array(points)
    faces = np.array(faces)
    #print(points.shape, faces.shape)
    finale_mesh = trimesh.Trimesh(vertices=points, faces=faces)
    finale_mesh.export(output_filename)
