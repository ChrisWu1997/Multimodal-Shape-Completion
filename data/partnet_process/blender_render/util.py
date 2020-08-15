import argparse, sys, os
import numpy as np

from math import radians

from plyfile import PlyData, PlyElement
from PIL import Image

cat_name2id = {
                'plane': '02691156',
                'car': '02958343',
                'chair': '03001627',
                'table': '04379243',
                'lamp': '03636649',
                'sofa': '04256520',
                'boat': '04530566',
                'dresser': '02933112'
                }
transformation_ShapeNet_v1tov2 = np.array([ [ 0, 0, 1, 0],
                                            [ 0, 1, 0, 0],
                                            [-1, 0, 0, 0],
                                            [ 0, 0, 0, 1]])
transformation_ShapeNet_v2tov1 = np.array([ [ 0, 0, -1, 0],
                                            [ 0, 1,  0, 0],
                                            [ 1, 0,  0, 0],
                                            [ 0, 0,  0, 1]])

def get_shapenet_clsID_modelname_from_filename(filename):
    clsid = filename.split('/')[-4]
    mn = filename.split('/')[-3]
    return clsid, mn

def get_index_of_arr_in_list(list_of_array, arr):
    for i, a in enumerate(list_of_array):
        if (a==arr).all(): return i
    return None

def items_in_txt_file(txt_filename):
    with open(txt_filename) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content

# ----------------------------------------
# Point cloud IO
# ----------------------------------------
def read_ply(filename, return_faces=False):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([ [b for b in a] for a in pc])

    if not return_faces:
        return pc_array

    try:
        faces = plydata['face'].data
        if len(faces[0]) == 3:
            face_array = np.array([ [b for b in a] for a in faces])
            face_array = np.squeeze(face_array)
        else:
            face_array = []
            for f in faces:
                f_indices = f[0].tolist()
                color = list((f.tolist())[1:])
                face_array.append(f_indices + color)
    except Exception as e:
        print('Warning: returning None face array')
        face_array = None    

    return np.array(pc_array), np.array(face_array)

def write_ply(points, filename, colors=None, normals=None, text=False):
    """ input: Nx3, write points to filename as PLY format. """
    if colors is not None: assert(points.shape[0]==colors.shape[0])
    if normals is not None: assert(points.shape[0]==normals.shape[0])

    if colors is None and normals is None:
        points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
        vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    elif colors is not None and normals is None:
        points = [(points[i,0], points[i,1], points[i,2], 
                   int(colors[i, 0]*255), int(colors[i, 1]*255), int(colors[i, 2]*255)
                  ) for i in range(points.shape[0])]
        vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                                         ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    elif colors is None and normals is not None:
        points = [(points[i,0], points[i,1], points[i,2], 
                   normals[i, 0], normals[i, 1], normals[i, 2]
                  ) for i in range(points.shape[0])]
        vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                                         ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])
    elif colors is not None and normals is not None:
        points = [(points[i,0], points[i,1], points[i,2], 
                   normals[i, 0], normals[i, 1], normals[i, 2], 
                   int(colors[i, 0]*255), int(colors[i, 1]*255), int(colors[i, 2]*255)
                   ) for i in range(points.shape[0])]
        vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                                         ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'), 
                                         ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)

######### image IO ################
import OpenEXR as exr
import Imath
import array
def read_exr_image(exr_filename, channels=['R', 'G', 'B']):
    exrfile = exr.InputFile(exr_filename)
    dw = exrfile.header()['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    channelData_arr = dict()

    for c in channels:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)        
        channelData_arr[c] = C

    # get the whole array
    img_arr = np.concatenate([channelData_arr[c][...,np.newaxis] for c in ['R', 'G', 'B']], axis=2) # (res, res, 3)
    return img_arr

def write_exr_image(im_arr, exr_filename):
    channels = ['R', 'G', 'B', 'A']
    new_header = exr.Header(im_arr.shape[0], im_arr.shape[1])
    new_header['channel'] = { 'R' : Imath.Channel(Imath.PixelType(exr.FLOAT)),
                              'G' : Imath.Channel(Imath.PixelType(exr.FLOAT)),
                              'B' : Imath.Channel(Imath.PixelType(exr.FLOAT)),
                              'A' : Imath.Channel(Imath.PixelType(exr.FLOAT))}
    if im_arr.shape[-1] == 3: 
        channels = ['R', 'G', 'B']
        new_header['channel'] = { 'R' : Imath.Channel(Imath.PixelType(exr.FLOAT)),
                                  'G' : Imath.Channel(Imath.PixelType(exr.FLOAT)),
                                  'B' : Imath.Channel(Imath.PixelType(exr.FLOAT))}
    elif im_arr.shape[-1] == 2:
        channels = ['R', 'A']
        new_header['channel'] = { 'R' : Imath.Channel(Imath.PixelType(exr.FLOAT)),
                                  'A' : Imath.Channel(Imath.PixelType(exr.FLOAT))}
    elif im_arr.ndim == 2: # 1-channel
        channels = ['R']
        new_header['channel'] = { 'R' : Imath.Channel(Imath.PixelType(exr.FLOAT))}
        im_arr = np.expand_dims(im_arr, -1)
    
    channelData = dict()
    for i, c in enumerate(channels):
        channelData[c] = array.array('f', im_arr[:, :, i].astype(np.float32).flatten().tostring())

    exr_out = exr.OutputFile(exr_filename, new_header)
    exr_out.writePixels(channelData)
    return

############# util functions ###################
def read_verts(mesh):
    mverts_co = np.zeros((len(mesh.data.vertices)*3), dtype=np.float)
    mesh.data.vertices.foreach_get("co", mverts_co)
    return np.reshape(mverts_co, (len(mesh.data.vertices), 3)) 

def sample_from_point_cloud(point_cloud, nb_samples=1000):
    indices = list(range(point_cloud.shape[0]))
    random_choices = np.random.choice(indices, nb_samples, replace=True)
    sampled_pc = point_cloud[random_choices]
    return sampled_pc

def pc_normalize(pc, center_type='bbox', norm_type='diag2sphere', eps=0.01):
    if center_type == 'bbox':
        pts_min = np.amin(pc, axis=0)
        pts_max = np.amax(pc, axis=0)
        centroid = (pts_max + pts_min) / 2.
    elif center_type == 'mass':
        centroid = np.mean(pc, axis=0)
    else:
        raise NotImplementedError

    if norm_type == 'unit_sphere':
      # fit model into a unit sphere
      #eps = 0.025

      pc_trans = pc - centroid
      m_r = np.max(np.sqrt(np.sum(pc_trans**2, axis=1)))
      scale_f = (0.5 - eps) / m_r
      
      return -centroid, scale_f
    elif norm_type == 'unit_cube':
      # scale the model such that its max side length is 1
      #eps = 0.05

      pc_trans = pc - centroid
      
      pts_min_trans = np.amin(pc_trans, axis=0)
      pts_max_trans = np.amax(pc_trans, axis=0)
      extent_trans = pts_max_trans - pts_min_trans
      max_side_len_trans = np.max(extent_trans)
      scale_f = (1. - eps) / max_side_len_trans 

      return -centroid, scale_f
    elif norm_type == 'diag2sphere':
      # make the diagnal of bbox equal to 1 unit
      pc_trans = pc - centroid
      pts_min_trans = np.amin(pc_trans, axis=0)
      pts_max_trans = np.amax(pc_trans, axis=0)
      diag_length = np.linalg.norm(pts_min_trans - pts_max_trans)
      scale_f = (1. + eps) / diag_length
      #print(pts_min_trans, pts_max_trans, diag_length, scale_f)
      return -centroid, scale_f
    else:
      print('Error: unknow normalization type: %s. Not normalizing'%(norm_type))
      raise NotImplementedError

def mesh_normalize(tmesh, center_type='bbox', norm_type='unit_sphere', eps=0.05):
    pc = np.array(tmesh.vertices)
    if center_type == 'bbox':
        pts_min = np.amin(pc, axis=0)
        pts_max = np.amax(pc, axis=0)
        centroid = (pts_max + pts_min) / 2.
    elif center_type == 'mass':
        centroid = np.mean(pc, axis=0)
    else:
        raise NotImplementedError

    if norm_type == 'unit_sphere':
      # fit model into a unit sphere
      pc_trans = pc - centroid
      m_r = np.max(np.sqrt(np.sum(pc_trans**2, axis=1)))
      scale_f = (0.5 - eps) / m_r
      
    elif norm_type == 'unit_cube':
      # scale the model such that its max side length is 1
      #eps = 0.05
      pc_trans = pc - centroid
      pts_min_trans = np.amin(pc_trans, axis=0)
      pts_max_trans = np.amax(pc_trans, axis=0)
      extent_trans = pts_max_trans - pts_min_trans
      max_side_len_trans = np.max(extent_trans)
      scale_f = (1. - eps) / max_side_len_trans 
    elif norm_type == 'diag2sphere':
      # make the diagnal of bbox equal to 1 unit
      pc_trans = pc - centroid
      pts_min_trans = np.amin(pc_trans, axis=0)
      pts_max_trans = np.amax(pc_trans, axis=0)
      diag_length = np.linalg.norm(pts_min_trans - pts_max_trans)
      scale_f = (1.-eps) / diag_length
    else:
      print('Error: unknow normalization type: %s. Not normalizing'%(norm_type))
      raise NotImplementedError

    trans_v = -centroid
    scale_f = scale_f

    new_verts = pc + trans_v
    new_verts = new_verts * scale_f
    
    return trimesh.Trimesh(vertices=new_verts, faces=tmesh.faces)

def read_and_correct_normal(exr_filename, correct_normal=True, mask_arr=None):
    '''
    input:
      exr_filename: normal image
      correct_normal: if set True, will flip normals that are wrongly pointing
      mask_arr: 0 - bg, 1 - fg, if is not None, will set the bg normal to [0, 0, 0]
    return: corrected normal
    '''

    # get the whole array and toward all normals to camera
    img_arr = -read_exr_image(exr_filename)

    if correct_normal:
        # flip those wrong-oriented normals
        wrong_mask = np.all(np.expand_dims(img_arr[:, :, 2], axis=2) < 0, axis=-1) # (res, res)
        img_arr[wrong_mask] = -img_arr[wrong_mask]

    if mask_arr is not None:
        bg_mask = np.all((np.expand_dims(mask_arr, axis=2))==0, axis=-1)
        img_arr[bg_mask] = [0, 0, 0] # set bg normal to 0

    return img_arr
  
def read_depth_and_get_mask(depth_exr_filename, far_thre=1):
    # foreground -> 1
    # bg -> 0
    # any pixel with depth larger than far_thre will be considered as bg
    # get the whole array
    img_arr = read_exr_image(depth_exr_filename)

    depth_arr = img_arr[:, :, 0] # only the first channel, the rest is the same
    depth_arr = np.expand_dims(depth_arr, axis=2) # (res, res, 1)
    bg_mask = np.all(depth_arr > far_thre, axis=-1)
    # init a full opaque image
    im_mask = np.ones((img_arr.shape[0], img_arr.shape[1]))
    im_mask[bg_mask] = 0 # set bg to full transperancy

    depth_arr = np.squeeze(depth_arr)
    return depth_arr, im_mask

def blend_im_mask_to_exr(exr_filename, im_mask, clip_min=-1, clip_max=1):
    exrfile = exr.InputFile(exr_filename)
    #print(exrfile.header())
    dw = exrfile.header()['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    channels = ['R', 'G', 'B']
    channelData = dict()

    for c in channels:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))

        C_arr = np.fromstring(C, dtype=np.float32)
        #C_arr = np.reshape(C_arr, isize)
        C_arr = np.clip(C_arr, a_min=clip_min, a_max=clip_max)

        channelData[c] = array.array('f', C_arr.astype(np.float32).flatten().tostring())

    # alpha channel with im_mask data
    channelData['A'] = array.array('f', im_mask.astype(np.float32).flatten().tostring())

    os.remove(exr_filename)
    new_header = exr.Header(args.reso, args.reso)
    new_header['channel'] = { 'R' : Imath.Channel(Imath.PixelType(exr.FLOAT)),
                              'G' : Imath.Channel(Imath.PixelType(exr.FLOAT)),
                              'B' : Imath.Channel(Imath.PixelType(exr.FLOAT)),
                              'A' : Imath.Channel(Imath.PixelType(exr.FLOAT))}
    exr_out = exr.OutputFile(exr_filename.replace('0001', ''), new_header)
    exr_out.writePixels(channelData)
    return

def get_3D_points_from_ortho_depth(depth_arr, ortho_view_scale=1.):

    width_pix = depth_arr.shape[1]
    height_pix = depth_arr.shape[0]

    pix_len_w = ortho_view_scale / width_pix 
    pix_len_h = ortho_view_scale / height_pix 

    x_coords = np.linspace(-ortho_view_scale/2. + pix_len_w/2.,  ortho_view_scale/2. - pix_len_w/2., width_pix)
    y_coords = np.linspace( ortho_view_scale/2. - pix_len_h/2., -ortho_view_scale/2. + pix_len_h/2., height_pix)

    xv, yv = np.meshgrid(x_coords, y_coords, indexing='xy')

    if depth_arr.ndim == 2:
        depth_arr = np.expand_dims(depth_arr, axis=-1)
    xv = np.expand_dims(xv, axis=-1)
    yv = np.expand_dims(yv, axis=-1)
    xyz = np.concatenate((xv, yv, -depth_arr), axis=-1)
    return xyz

def get_rays_from_ori_ortho_view(resolution=640, view_scale=1.):
    width_pix = resolution
    height_pix = resolution

    pix_len_w = view_scale / width_pix 
    pix_len_h = view_scale / height_pix 

    x_coords = np.linspace(-view_scale/2. + pix_len_w/2.,  view_scale/2. - pix_len_w/2., width_pix)
    y_coords = np.linspace( view_scale/2. - pix_len_h/2., -view_scale/2. + pix_len_h/2., height_pix)

    xv, yv = np.meshgrid(x_coords, y_coords, indexing='xy')

    xv = np.expand_dims(xv, axis=-1)
    yv = np.expand_dims(yv, axis=-1)
    zv = np.zeros(xv.shape)
    origins = np.concatenate((xv, yv, zv), axis=-1)

    zv = -np.ones(xv.shape)
    dirs = np.concatenate((xv, yv, zv), axis=-1)

    return origins, dirs

import blender_camera_util
def get_albedo_by_ray_intersection(tmesh, blender_cam, reso, ortho_view_scale=1.):

    def get_arr_index_from_flat_index(flat_index):
        if flat_index < 0  or flat_index >= reso*reso:
            return None

        row_idx = int(flat_index / reso)
        col_idx = flat_index - row_idx*reso

        return (row_idx, col_idx)

    r_locations, r_dirs = get_rays_from_ori_ortho_view(reso, 1.0)
    r_locations = np.reshape(r_locations, (-1, 3))
    r_dirs = np.reshape(r_dirs, (-1, 3))

    RT_bcam2world = blender_camera_util.get_bcam2world_RT_matrix_from_blender(blender_cam)
    r_locations, r_dirs = transform_points(r_locations, RT_bcam2world), transform_points(r_dirs, RT_bcam2world)
    # ray testing
    print('Ray intersection testing...')
    tri_indices, r_indices = tmesh.ray.intersects_id(r_locations, r_dirs)
    print('Ray intersection testing done.')
    albedo_arr = np.ones((reso, reso, 3))
    all_mesh_tri_colors = tmesh.visual.face_colors
    for hit_idx, tri_idx in enumerate(tri_indices):
        hit_color = all_mesh_tri_colors[tri_idx]
        albedo_arr_idx = get_arr_index_from_flat_index(r_indices[hit_idx])
        albedo_arr[albedo_arr_idx] = hit_color[:3]

    return albedo_arr

def translate_points(points, trans_v):

    return

def transform_points(points, trans_mat):
    '''
        points: np.array, nx3
        trans_mat: np.array, 4x4
    '''
    ones = np.ones((points.shape[0], 1))
    points = np.concatenate([points, ones], axis=-1)

    points = np.dot(points, trans_mat.transpose())
    return points[:, :-1]

def remove_bg_points(points_normals_colors):

    new_points = []
    for i in range(points_normals_colors.shape[0]):
        normal = points_normals_colors[i][3:6]
        if np.array_equal(normal, np.array([0,0,0])):
            continue
        else:
            new_points.append(points_normals_colors[i])
    return np.array(new_points)

###### convert point cloud to sdf #####
def plane_which_side(pts_on_plane, plane_normal, query_pts):
    d = - (plane_normal[0]*pts_on_plane[0] + plane_normal[1]*pts_on_plane[1] + plane_normal[2]*pts_on_plane[2])
    res = np.dot(np.array([plane_normal[0], plane_normal[1], plane_normal[2], d]), np.array([query_pts[0], query_pts[1], query_pts[2], 1]))

    if res >= 0: return 1 # positive side
    else: return -1 # negtive side

from tqdm import tqdm
from scipy import spatial
def scan_2_sdf(points_normals_colors, sdf_resolution=128, sdf_scale=1):
    points = points_normals_colors[:, :3]

    print('Construct KD-tree from %d points.'%(points.shape[0]))
    tree = spatial.KDTree(points)

    # get centers of volumn voxels
    vox_len = sdf_scale / sdf_resolution
    x_coords = np.linspace(-sdf_scale/2. + vox_len/2.,  sdf_scale/2. - vox_len/2., sdf_resolution)
    y_coords = np.linspace(-sdf_scale/2. + vox_len/2.,  sdf_scale/2. - vox_len/2., sdf_resolution)
    z_coords = np.linspace(-sdf_scale/2. + vox_len/2.,  sdf_scale/2. - vox_len/2., sdf_resolution)
    xv, yv, zv = np.meshgrid(x_coords, y_coords, z_coords, indexing='xy')
    xv = np.expand_dims(xv, axis=-1)
    yv = np.expand_dims(yv, axis=-1)
    zv = np.expand_dims(zv, axis=-1)
    xyz_center_arr = np.concatenate((xv, yv, zv), axis=-1)

    sdf_volumn = np.ones((sdf_resolution, sdf_resolution, sdf_resolution, 4)) # channels for signed distance, and rgb

    # 
    print('Querying all points...')
    distances, pts_indices = tree.query(np.reshape(xyz_center_arr, (-1, 3)))
    distances = np.reshape(distances, (sdf_resolution, sdf_resolution, sdf_resolution))
    pts_indices = np.reshape(pts_indices, (sdf_resolution, sdf_resolution, sdf_resolution))
    print('Querying done.')

    for i in tqdm(range(sdf_resolution)):
        for j in range(sdf_resolution):
            for k in range(sdf_resolution):
                xyz_center = xyz_center_arr[i, j, k]
                dist, pts_idx = distances[i,j,k], pts_indices[i,j,k]

                target_pts = points_normals_colors[pts_idx, :3]
                target_nor = points_normals_colors[pts_idx, 3:6]
                target_clr = points_normals_colors[pts_idx, 6:9]

                # check the sign of distance
                side = plane_which_side(target_pts, target_nor, xyz_center)
                if side == 1:
                    sdf_volumn[i, j, k] = [dist, target_clr[0], target_clr[1], target_clr[2]]
                elif side == -1:
                    sdf_volumn[i, j, k] = [-dist, target_clr[0], target_clr[1], target_clr[2]]
                else:
                    raise NotImplementedError('Unknown side result: ', side)

    return sdf_volumn

import trimesh
def get_color_from_reference_pointcloud(tmesh, points_normals_colors):
    points = points_normals_colors[:, :3]

    print('Construct KD-tree from %d points.'%(points.shape[0]))
    tree = spatial.KDTree(points)

    if True:
        tri_centers = []
        for tri in tmesh.faces:
            tri_center = (tmesh.vertices[tri[0]] + tmesh.vertices[tri[1]] + tmesh.vertices[tri[2]]) / 3
            tri_centers.append(tri_center)
        
        distances, indices = tree.query(tri_centers)

        tri_colors = []
        for i in range(0, len(tri_centers)):
            target_i = indices[i]
            target_clr = points_normals_colors[target_i, 6:9]
            tri_colors.append(target_clr)
        tri_colors = np.array(tri_colors)

        color_mesh = trimesh.Trimesh(vertices=tmesh.vertices, faces=tmesh.faces, face_colors=tri_colors)

    elif False:
        distances, indices = tree.query(tmesh.vertices)
        colors = points_normals_colors[indices, 6:9]
        color_mesh = trimesh.Trimesh(vertices=tmesh.vertices, faces=tmesh.faces, vertex_colors=colors)
    else:
        tri_centers = []
        for tri in tmesh.faces:
            tri_center = (tmesh.vertices[tri[0]] + tmesh.vertices[tri[1]] + tmesh.vertices[tri[2]]) / 3
            tri_centers.append(tri_center)
        
        distances, indices = tree.query(tri_centers)

        tri_colors = []
        for i in range(0, len(tri_centers)):
            target_i = indices[i]
            target_clr = points_normals_colors[target_i, 6:9]
            tri_colors.append(target_clr)
        tri_colors = np.array(tri_colors)

        distances, indices = tree.query(tmesh.vertices)
        colors = points_normals_colors[indices, 6:9]
        color_mesh = trimesh.Trimesh(vertices=tmesh.vertices, faces=tmesh.faces, vertex_colors=colors, face_colors=tri_colors)

    return color_mesh

import sys
sys.setrecursionlimit(1000000)
def get_ref_point_idx_from_point_cloud(tmesh, point_cloud_with_feat, faster=True):
    points = point_cloud_with_feat[:, :3]

    if faster: 
        print('Construct cKD-tree from %d points.'%(points.shape[0]))
        tree = spatial.cKDTree(points)
    else:
        print('Construct KD-tree from %d points.'%(points.shape[0]))
        tree = spatial.KDTree(points)
    print('(c)KD-tree done.')
    tri_centers = []
    for tri in tmesh.faces:
        tri_center = (tmesh.vertices[tri[0]] + tmesh.vertices[tri[1]] + tmesh.vertices[tri[2]]) / 3
        tri_centers.append(tri_center)
    
    print('Querying for %d points...'%(len(tri_centers)))
    _, indices = tree.query(tri_centers)
    print('Queries done.')
    return indices

from scipy.io import loadmat
import skimage.measure
def mesh_from_voxels(voxel_mat_filename, downsample_factor=1):
    voxel_model_mat = loadmat(voxel_mat_filename)
    voxel_model_b = voxel_model_mat['b'][:].astype(np.int32)
    voxel_model_bi = voxel_model_mat['bi'][:].astype(np.int32)-1
    voxel_model_256 = np.zeros([256,256,256],np.uint8)
    for i in range(16):
        for j in range(16):
            for k in range(16):
                voxel_model_256[i*16:i*16+16,j*16:j*16+16,k*16:k*16+16] = voxel_model_b[voxel_model_bi[i,j,k]]
    # downsample
    if downsample_factor != 1:
        if downsample_factor not in [1, 2, 4]:
            print('Skip downsampling, invalid downsample factor: ', downsample_factor)
        else:
            voxel_model_256 = skimage.measure.block_reduce(voxel_model_256, (downsample_factor,downsample_factor,downsample_factor), np.max)
    
    #add flip&transpose to convert coord from shapenet_v1 to shapenet_v2
    voxel_model_256 = np.transpose(voxel_model_256, (2,1,0))
    voxel_model_256 = np.flip(voxel_model_256, 2)

    voxel_size = 1/ (256 / downsample_factor)
    verts, faces, normals_, values = skimage.measure.marching_cubes_lewiner(
        voxel_model_256, level=0.0, spacing=[voxel_size] * 3
        )

    # move to the orgine
    verts = verts - [0.5,0.5,0.5]
    # flip the index order for all faces
    faces = faces[:, [0, 2, 1]]

    # NOTE: for car, still need to flip x?
    if '02958343' in voxel_mat_filename:
        verts[:, 0] = -verts[:, 0]
        faces = faces[:, [0, 2, 1]] # do not why we need change the order again to flip the face normal

    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    return mesh

if __name__ == "__main__":
    color_arr = read_exr_image('color0001.exr')
    color_iamge = Image.fromarray((color_arr*255).astype(np.unit8))
    color_iamge.save('color.png')

'''
import skimage.measure
if __name__ == "__main__":
    sdf_resolution = 64
    sdf_scale = 1
    vox_size = float(sdf_scale) / sdf_resolution

    points_normals_colors = read_ply('test_after.ply')
    points_normals_colors = points_normals_colors[0:-1:100]
    sdf_v = scan_2_sdf(points_normals_colors, sdf_resolution=sdf_resolution)
    vertices, triangles, normals, values = skimage.measure.marching_cubes_lewiner(
        sdf_v[:,:,:, 0], level=0.0, spacing=[vox_size] * 3
    )
'''

'''
if __name__ == "__main__":
    depth_arr = np.ones((10,10))
    xyz = get_3D_points_from_ortho_depth(depth_arr)

    points = np.reshape(xyz, (-1, 3))
    
    colors = np.zeros((100,3))
    colors[:, 2] = 1
    write_ply(points, 'test.ply', colors, normals=colors)
'''