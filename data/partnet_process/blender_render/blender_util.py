import bpy
import bpy_extras
from mathutils import Matrix
import numpy as np
import sys
sys.path.append('.')
import util
import blender_camera_util
import os
from math import radians
from PIL import Image
###################################################
# convertion between blender coord system and shapenet object system
# X_b = X_s, Y_b = -Z_s, Z_b = Y_s
# subscript: b for blender, s for shapenet
###################################################
R_axis_switching_BtoS = np.array([[1, 0, 0, 0],
                                 [ 0, 0, 1, 0],
                                 [ 0, -1, 0, 0],
                                 [ 0, 0, 0, 1]])
R_axis_switching_StoB = np.array([[1, 0, 0, 0],
                                 [ 0, 0, -1, 0],
                                 [ 0, 1, 0, 0],
                                 [ 0, 0, 0, 1]])

def get_material_from_passIdx(mat_pass_idx):
    for mat in bpy.data.materials:
        if mat.pass_index == mat_pass_idx: return mat
    return None

def clear_scene_objects():
    for obj in bpy.data.objects:
        if 'Lamp' in obj.name or 'Camera' in obj.name or 'lamp' in obj.name or 'camera' in obj.name: continue
        bpy.data.objects[obj.name].select = True
        bpy.ops.object.delete()
        print('Delete object: ', obj.name)

def get_lookat_target(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_empty.rotation_mode = 'XYZ'
    #b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.objects.link(b_empty)
    scn.objects.active = b_empty
    return b_empty

def get_obj_verts(obj, read_global=False):
    vertices = obj.data.vertices
    if read_global:
        verts = [obj.matrix_world * vert.co for vert in vertices] 
    else:
        verts = [vert.co for vert in vertices] 
    verts = np.array(verts)
    return verts

from mathutils import Vector, Matrix
import mathutils
def translate_obj(obj, trans_v):
    obj.delta_location[0] = trans_v[0]
    obj.delta_location[1] = trans_v[1]
    obj.delta_location[2] = trans_v[2]

def rotate_obj(obj, euler_angles=[0, 0, 0]):
    '''
    rotate object in XYZ order, in angles
    '''
    mat_world = obj.matrix_world

    rot_x = Matrix.Rotation(radians(euler_angles[0]), 4, 'X')
    rot_y = Matrix.Rotation(radians(euler_angles[1]), 4, 'Y')
    rot_z = Matrix.Rotation(radians(euler_angles[2]), 4, 'Z')
    rot_mat = rot_z * rot_y * rot_x

    mat_edit = rot_mat * mat_world
    obj.matrix_world = mat_edit


def scale_obj(obj, scale_v):
    obj.delta_scale[0] = scale_v[0]
    obj.delta_scale[1] = scale_v[1]
    obj.delta_scale[2] = scale_v[2]

########### setup before rendering
def rendering_pass_setup(args):
    bpy.context.scene.frame_set(1)
    # Set up rendering.
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    # Add passes for additionally dumping albedo and normals.
    bpy.context.scene.render.layers["RenderLayer"].use_pass_normal = True
    bpy.context.scene.render.layers["RenderLayer"].use_pass_color = True
    #bpy.context.scene.render.layers["RenderLayer"].use_pass_diffuse = True
    bpy.context.scene.render.layers["RenderLayer"].use_pass_material_index = True
    bpy.context.scene.render.image_settings.file_format = args.format
    bpy.context.scene.render.image_settings.color_depth = args.color_depth

    # Clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    # Create input render layer node.
    render_layers = tree.nodes.new('CompositorNodeRLayers')

    # depth pass
    depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    if args.format == 'OPEN_EXR':
        links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
    else:
        # Remap as other types can not represent the full range of depth.
        map = tree.nodes.new(type="CompositorNodeMapValue")
        # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
        map.offset = [-0.7]
        map.size = [args.depth_scale]
        map.use_min = True
        map.min = [0]
        links.new(render_layers.outputs['Depth'], map.inputs[0])

        links.new(map.outputs[0], depth_file_output.inputs[0])

    # normal pass
    if args.format == 'OPEN_EXR':
        normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
        normal_file_output.label = 'Normal Output'
        links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])
    else:
        print('Unknow format.')

    # color pass
    albedo_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    albedo_file_output.label = 'Albedo Output'
    links.new(render_layers.outputs['Color'], albedo_file_output.inputs[0])

    # material index pass
    matidx_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    matidx_file_output.label = 'Matidx Output'
    links.new(render_layers.outputs['IndexMA'], matidx_file_output.inputs[0])

    for output_node in [depth_file_output, 
                    normal_file_output, 
                    albedo_file_output, 
                    matidx_file_output]:
        output_node.base_path = ''

    return depth_file_output, normal_file_output, albedo_file_output, matidx_file_output

def process_scene_objects(args):
    for object in bpy.context.scene.objects:
        if object.name in ['Camera', 'Lamp']:
            continue
    
        bpy.context.scene.objects.active = object
        object.select = True
        
        if object.name == 'sphere':
            # do not touch the sphere model, which intends to give white albedo color for the background
            continue 
        else:
          if args.remove_doubles:
              bpy.ops.object.mode_set(mode='EDIT')
              bpy.ops.mesh.remove_doubles()
              bpy.ops.object.mode_set(mode='OBJECT')
          if args.remove_iso_verts:
              bpy.ops.object.mode_set(mode='EDIT')
              bpy.ops.mesh.delete_loose(use_verts=True, use_edges=True, use_faces=False)
              bpy.ops.object.mode_set(mode='OBJECT')
          if args.edge_split:
              bpy.ops.object.modifier_add(type='EDGE_SPLIT')
              bpy.context.object.modifiers["EdgeSplit"].split_angle = 1.32645
              bpy.ops.object.modifier_apply(apply_as='DATA', modifier="EdgeSplit")
          if args.normalization_mode is not None:
              # scale to be within a unit sphere (r=0.5, d=1)
              '''
              v = object.data.vertices
              verts_np = util.read_verts(object) # NOTE: get vertices in object local space
              trans_v, scale_f = util.pc_normalize(verts_np, norm_type=args.normalization_mode)
              # the axis conversion of importing does not change the data in-place,
              # so we do it manually
              trans_v_axis_replaced = trans_v.copy()
              
              trans_v_axis_replaced[0] = trans_v[0]
              trans_v_axis_replaced[1] = -trans_v[2]
              trans_v_axis_replaced[2] = trans_v[1]
              '''

              verts_np = get_obj_verts(object, read_global=True)
              trans_v, scale_f = util.pc_normalize(verts_np, norm_type=args.normalization_mode)
              trans_v_axis_replaced = trans_v

              bpy.ops.transform.translate(value=(trans_v_axis_replaced[0], trans_v_axis_replaced[1], trans_v_axis_replaced[2]))
              bpy.ops.object.transform_apply(location=True)
              bpy.ops.transform.resize(value=(scale_f, scale_f, scale_f))
              bpy.ops.object.transform_apply(scale=True)
              bpy.ops.export_scene.obj(filepath='./test.obj', use_selection=True)
          
          object.select = False

def convert_quad_mesh_to_triangle_mesh():
    for object in bpy.context.scene.objects:
        if not object.type == 'MESH': continue
        bpy.context.scene.objects.active = object
        object.select = True
        
        if object.name == 'sphere':
            # do not touch the sphere model, which intends to give white albedo color for the background
            continue 
        else:
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.quads_convert_to_tris()
            bpy.ops.object.mode_set(mode='OBJECT')
          
        object.select = False

def setup_render(args):
    ########## camera settings ##################
    scene = bpy.context.scene
    scene.render.resolution_x = args.reso
    scene.render.resolution_y = args.reso
    scene.render.resolution_percentage = 100
    scene.render.alpha_mode = 'TRANSPARENT'
    scene.render.image_settings.file_format = 'PNG'  # set output format to .png
    return

############# heavy rendering ################

from datetime import datetime
def get_default_camera():
    cam_obj = bpy.data.objects.get("Camera")
    if cam_obj is None:
        print('Camera not found, create a new one')
        cam_data = bpy.data.cameras.new("Camera")
        cam_obj = bpy.data.objects.new("Camera", cam_data)
        bpy.context.scene.objects.link(cam_obj)
        bpy.context.scene.camera = cam_obj
        return cam_obj
    else:
        bpy.context.scene.camera = cam_obj
        return cam_obj

def scan_point_cloud(depth_file_output, normal_file_output, albedo_file_output, matidx_file_output, args, rot_angles_list):
    scene = bpy.context.scene
    ######### filename for output ##############
    if 'ShapeNetCore' not in args.obj:
        model_identifier = args.obj.split('/')[-1].split('.')[0]
        correct_normal = False
    else:
        model_identifier = args.obj.split('/')[-3]
        correct_normal = True
    fp = os.path.join(args.output_folder, model_identifier)
    scene.render.image_settings.file_format = 'PNG'  # set output format to .png
    
    # scan shapenet shape into point cloud with features
    all_points_normals_colors_mindices = None
    for xyz_angle in rot_angles_list:

        # init camera
        cam = get_default_camera()
        cam_init_location = (0, 1.0, 0) # (0, 0.5, 0)
        cam.data.type = 'ORTHO'
        cam.data.ortho_scale = args.orth_scale
        cam.data.clip_start = 0
        cam.data.clip_end = 100 # a value that is large enough
        cam_constraint = cam.constraints.new(type='TRACK_TO')
        cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
        cam_constraint.up_axis = 'UP_Y'
        b_empty = get_lookat_target(cam)
        cam_constraint.target = b_empty # track to a empty object at the origin
        
        # rotate camera
        euler_rot_mat = euler2mat(radians(xyz_angle[0]), radians(xyz_angle[1]), radians(xyz_angle[2]), 'sxyz')
        new_cam_location = np.dot(euler_rot_mat, np.array(cam_init_location))
        cam.location = new_cam_location

        scene.render.filepath = fp + '-rotx=%.2f_roty=%.2f_rotz=%.2f'%(xyz_angle[0], xyz_angle[1], xyz_angle[2])
        depth_file_output.file_slots[0].path = scene.render.filepath + "_depth"
        normal_file_output.file_slots[0].path = scene.render.filepath + "_normal"
        albedo_file_output.file_slots[0].path = scene.render.filepath + "_albedo"
        matidx_file_output.file_slots[0].path = scene.render.filepath + "_matidx"

        # render and write out
        bpy.ops.render.render(write_still=True)  # render still

        depth_arr, hard_mask_arr = util.read_depth_and_get_mask(scene.render.filepath + "_depth0001.exr", far_thre=2)
        normal_arr = util.read_and_correct_normal(scene.render.filepath + "_normal0001.exr", correct_normal=correct_normal, mask_arr=hard_mask_arr)
        albedo_arr = util.read_exr_image(scene.render.filepath + "_albedo0001.exr")
        matidx_arr = util.read_exr_image(scene.render.filepath + "_matidx0001.exr")[:,:,0]
        # and the clip value range
        depth_arr = np.clip(depth_arr, a_min=0, a_max=2.5)
        normal_arr = np.clip(normal_arr, a_min=-1, a_max=1)
        albedo_arr = np.clip(albedo_arr, a_min=0, a_max=1)

        # process renderings to get the point cloud
        xyz_arr = util.get_3D_points_from_ortho_depth(depth_arr, args.orth_scale)
        xyz_normal_rgb_midx = np.reshape(np.concatenate([xyz_arr, normal_arr, albedo_arr, np.expand_dims(matidx_arr, -1)], axis=-1), (-1, 10))
        xyz_normal_rgb_midx = util.remove_bg_points(xyz_normal_rgb_midx)
        # transform from depth to 3D world point cloud
        RT_bcam2world = blender_camera_util.get_bcam2world_RT_matrix_from_blender(cam)
        # matrix for switching back axis of the obj file when output
        xyz_normal_rgb_midx[:, :3] = util.transform_points(xyz_normal_rgb_midx[:, :3], np.dot(R_axis_switching_BtoS, RT_bcam2world))
        xyz_normal_rgb_midx[:, 3:6] = util.transform_points(xyz_normal_rgb_midx[:, 3:6], np.dot(R_axis_switching_BtoS, RT_bcam2world))
        if all_points_normals_colors_mindices is None:
            all_points_normals_colors_mindices = xyz_normal_rgb_midx
        else:
            all_points_normals_colors_mindices = np.concatenate([all_points_normals_colors_mindices, xyz_normal_rgb_midx], axis=0)
        
        # remove renderings
        os.remove(scene.render.filepath+'.png')
        os.remove(scene.render.filepath + "_normal0001.exr")
        os.remove(scene.render.filepath + "_depth0001.exr")
        os.remove(scene.render.filepath + "_albedo0001.exr")
        os.remove(scene.render.filepath + "_matidx0001.exr")
        #os.remove('Image0001.exr')

    return all_points_normals_colors_mindices

def setup_sunlamp(target_obj):
    scene = bpy.context.scene
    # clear lamp created previously
    for obj in bpy.data.objects:
        if 'Lamp' in obj.name:
            bpy.data.objects[obj.name].select = True
            bpy.ops.object.delete()
            print('Delete lamp: ', obj.name)

    # Create new lamp datablock
    lamp_data = bpy.data.lamps.new(name="Lamp", type='SUN')
    lamp_data.use_specular = True
    lamp_data.use_diffuse = True
    lamp_data.color = (1,1,1)
    lamp_data.use_shadow = True
    lamp_data.shadow_method = 'RAY_SHADOW'
    lamp_data.energy = 1.
    # Create new object with our lamp datablock
    follow_lamp = bpy.data.objects.new(name="Lamp", object_data=lamp_data)
    # Link lamp object to the scene so it'll appear in this scene
    scene.objects.link(follow_lamp)
    # Place lamp to a specified location
    follow_lamp.location = (0, 0.5, 0)
    lamp_constraint = follow_lamp.constraints.new(type='TRACK_TO')
    lamp_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    lamp_constraint.up_axis = 'UP_Y'
    lamp_constraint.target = target_obj

    # also setup one right above the object
    lamp_data_overhead = bpy.data.lamps.new(name="Lamp_overhead", type='SUN')
    lamp_data_overhead.use_specular = True
    lamp_data_overhead.use_diffuse = True
    lamp_data_overhead.color = (1,1,1)
    lamp_data_overhead.use_shadow = True
    lamp_data_overhead.shadow_method = 'RAY_SHADOW'
    lamp_data_overhead.energy = 0.5
    # Create new object with our lamp datablock
    lamp_object_overhead = bpy.data.objects.new(name="Lamp_overhead", object_data=lamp_data_overhead)
    # Link lamp object to the scene so it'll appear in this scene
    scene.objects.link(lamp_object_overhead)
    # Place lamp to a specified location
    lamp_object_overhead.location = (0, 0, 0.5)

    return follow_lamp

from transforms3d.euler import euler2mat
def render_passes(depth_file_output, normal_file_output, albedo_file_output, args, rot_angles_list, subfolder_name='gt', output_format='exr'):
    scene = bpy.context.scene
    ######### filename for output ##############
    if 'ShapeNetCore' not in args.obj:
        model_identifier = args.obj.split('/')[-1].split('.')[0]
        correct_normal = False
    else:
        model_identifier = args.obj.split('/')[-3]
        correct_normal = True
    fp = os.path.join(args.output_folder, subfolder_name, model_identifier)
    scene.render.image_settings.file_format = 'PNG'  # set output format to .png

    # setup camera and render
    cam_init_location = (0., 0.5, 0.)
    cam = get_default_camera()
    cam.location = cam_init_location
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = args.orth_scale
    cam.data.clip_start = 0
    cam.data.clip_end = 100 # a value that is large enough
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    b_empty = get_lookat_target(cam)
    cam_constraint.target = b_empty # track to a empty object at the origin

    # setup light
    sun_lamp = setup_sunlamp(b_empty)

    for xyz_angle in rot_angles_list:

        # rotate camera
        euler_rot_mat = euler2mat(radians(xyz_angle[0]), radians(xyz_angle[1]), radians(xyz_angle[2]), 'sxyz')
        new_cam_location = np.dot(euler_rot_mat, np.array(cam_init_location))
        cam.location = new_cam_location
        # the sun lamp follows
        sun_lamp.location = new_cam_location

        scene.render.filepath = fp + '-rotx=%.2f_roty=%.2f_rotz=%.2f'%(xyz_angle[0], xyz_angle[1], xyz_angle[2])
        depth_file_output.file_slots[0].path = scene.render.filepath + "_depth"
        normal_file_output.file_slots[0].path = scene.render.filepath + "_normal"
        albedo_file_output.file_slots[0].path = scene.render.filepath + "_albedo"

        # render and write out
        bpy.ops.render.render(write_still=True)  # render still

        depth_arr, hard_mask_arr = util.read_depth_and_get_mask(scene.render.filepath + "_depth0001.exr")
        normal_arr = util.read_and_correct_normal(scene.render.filepath + "_normal0001.exr", correct_normal=correct_normal, mask_arr=hard_mask_arr)
        albedo_arr = util.read_exr_image(scene.render.filepath + "_albedo0001.exr")
        # and the clip value range
        depth_arr = np.clip(depth_arr, a_min=0, a_max=1)
        normal_arr = np.clip(normal_arr, a_min=-1, a_max=1)
        albedo_arr = np.clip(albedo_arr, a_min=0, a_max=1)

        # write out passes
        if output_format == 'exr':
            util.write_exr_image(depth_arr, scene.render.filepath + "_depth.exr")
            #util.write_exr_image(xyz_sworld_arr, scene.render.filepath + "_wxyz.exr")
            util.write_exr_image(normal_arr, scene.render.filepath + "_normal.exr")
            #util.write_exr_image(normal_sworld_arr, scene.render.filepath + "_wnormal.exr")
            util.write_exr_image(albedo_arr, scene.render.filepath + "_albedo.exr")
            util.write_exr_image(hard_mask_arr, scene.render.filepath + "_mask.exr")
        elif output_format == 'png':
            depth_arr = np.array(depth_arr*255, dtype=np.uint8)
            depth_pil = Image.fromarray(depth_arr)
            depth_pil.save(scene.render.filepath + "_depth.png")

            normal_arr = np.array((normal_arr+1)/2.*255, dtype=np.uint8)
            normal_pil = Image.fromarray(normal_arr)
            normal_pil.save(scene.render.filepath + "_normal.png")
        
            albedo_arr = np.array(albedo_arr*255, dtype=np.uint8)
            albedo_pil = Image.fromarray(albedo_arr)
            albedo_pil.save(scene.render.filepath + "_albedo.png")

            hard_mask_arr = np.array(hard_mask_arr*255, dtype=np.uint8)
            mask_pil = Image.fromarray(hard_mask_arr)
            mask_pil.save(scene.render.filepath + "_mask.png")

        # remove renderings
        #os.remove(scene.render.filepath+'.png')
        os.remove(scene.render.filepath + "_normal0001.exr")
        os.remove(scene.render.filepath + "_depth0001.exr")
        os.remove(scene.render.filepath + "_albedo0001.exr")
        #os.remove('Image0001.exr')


###########################
# for CYCLES RENDER
def process_scene_objects_CYCLES(args):
    
    # only worry about data in the startup scene
    # remove lamps and cameras
    for bpy_data_iter in (
            bpy.data.lamps,
            bpy.data.cameras
    ):
        for id_data in bpy_data_iter:
            bpy_data_iter.remove(id_data)

    for ob in bpy.context.scene.objects:
        if ob.type != 'MESH':
            ob.select = True
        else:
            ob.select = False
    bpy.ops.object.delete()

    # join all objects togather
    # make the first one active object
    # select the rest ones, then join
    '''
    for ob in bpy.context.scene.objects:
        if ob.type == 'MESH':
            ob.select = True
            bpy.context.scene.objects.active = ob
        else:
            ob.select = False
    bpy.ops.object.join()
    '''

    #assert len(bpy.context.scene.objects) == 1
    '''
    bpy.ops.object.select_all(action='DESELECT')
    obj_names = []
    for i, object in enumerate(bpy.context.scene.objects):
        obj_names.append(object.name)
    
    bpy.ops.object.mode_set(mode='EDIT')
    for obj_name in obj_names:
        bpy.data.objects[obj_name].select = True
        bpy.ops.mesh.separate(type='LOOSE')
        bpy.ops.object.select_all(action='DESELECT')
    '''

    # normalization
    # apply all modifiers in all objects to get the actual meshes
    for i, object in enumerate(bpy.context.scene.objects):
        bpy.context.scene.objects.active = object
        object.select = True
        for modifier in object.modifiers:
            bpy.ops.object.modifier_apply(modifier=modifier.name)

    # get all actual points for calculating transformation
    all_verts = []
    for i, object in enumerate(bpy.context.scene.objects):
        verts_np = get_obj_verts(object, read_global=True)
        all_verts += list(verts_np)
    all_verts_np = np.array(all_verts)
    trans_v, scale_f = util.pc_normalize(all_verts_np, norm_type=args.normalization_mode)

    #bpy.ops.wm.save_as_mainfile(filepath='test0.blend')

    for i, object in enumerate(bpy.context.scene.objects):
        bpy.context.scene.objects.active = object
        object.select = True
        
        if args.normalization_mode is not None:
            # too complicated to scale the object, so we only translate the object to the original without touching the scaling
            translate_obj(object, trans_v)

            # debug
            rotate_obj(object, [0,0,180])
            
            #scale_obj(object, [scale_f, scale_f, scale_f])
            #bpy.ops.transform.resize(value=(scale_f, scale_f, scale_f))
            #bpy.ops.object.transform_apply(scale=True)
            #bpy.ops.export_scene.obj(filepath='test.obj', use_selection=True)
        
        object.select = False
    # save to debug
    bpy.ops.wm.save_as_mainfile(filepath='test1.blend')

    diag_length = 1 / scale_f
    return diag_length

def rendering_pass_setup_CYCLES(args):
    bpy.context.scene.frame_set(1)
    # Set up rendering.
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    # Add passes for additionally dumping albedo and normals.
    bpy.context.scene.render.layers["RenderLayer"].use_pass_normal = True
    bpy.context.scene.render.layers["RenderLayer"].use_pass_diffuse_color = True
    bpy.context.scene.render.layers["RenderLayer"].use_pass_glossy_direct = True
    #bpy.context.scene.render.layers["RenderLayer"].use_pass_diffuse = True
    bpy.context.scene.render.layers["RenderLayer"].use_pass_material_index = True
    bpy.context.scene.render.image_settings.file_format = args.format
    bpy.context.scene.render.image_settings.color_depth = args.color_depth

    # Clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    # Create input render layer node.
    render_layers = tree.nodes.new('CompositorNodeRLayers')

    # depth pass
    depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    if args.format == 'OPEN_EXR':
        links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
    else:
        # Remap as other types can not represent the full range of depth.
        map = tree.nodes.new(type="CompositorNodeMapValue")
        # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
        map.offset = [-0.7]
        map.size = [args.depth_scale]
        map.use_min = True
        map.min = [0]
        links.new(render_layers.outputs['Depth'], map.inputs[0])

        links.new(map.outputs[0], depth_file_output.inputs[0])

    # normal pass
    if args.format == 'OPEN_EXR':
        normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
        normal_file_output.label = 'Normal Output'
        links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])
    else:
        print('Unknow format.')

    # color pass
    albedo_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    albedo_file_output.label = 'Albedo Output'
    #print(render_layers.outputs.keys())
    links.new(render_layers.outputs['DiffCol'], albedo_file_output.inputs[0])

    # glossy direct pass
    glossydir_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    glossydir_file_output.label = 'Glossy Direct Output'
    links.new(render_layers.outputs['GlossDir'], glossydir_file_output.inputs[0])

    # material index pass
    matidx_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    matidx_file_output.label = 'Matidx Output'
    links.new(render_layers.outputs['IndexMA'], matidx_file_output.inputs[0])

    for output_node in [depth_file_output, 
                    normal_file_output, 
                    albedo_file_output, 
                    matidx_file_output,
                    glossydir_file_output]:
        output_node.base_path = ''

    return depth_file_output, normal_file_output, albedo_file_output, matidx_file_output, glossydir_file_output

def get_material_roughness(mat_idx):
    mat = get_material_from_passIdx(mat_idx)
    roughness = 1.
    if mat is None: return roughness
    
    for mat_bsdf in mat.node_tree.nodes.keys():
        if 'Glossy BSDF' in mat_bsdf or 'Glass BSDF' in mat_bsdf:
            roughness_tmp = mat.node_tree.nodes[mat_bsdf].inputs[1].default_value
            if roughness_tmp < roughness: roughness = roughness_tmp
    return roughness

def assemble_roughness_map(matidx_arr):
    matidx_arr_cp = matidx_arr.copy()
    for i in range(len(bpy.data.materials)):
        roughness_here = get_material_roughness(i)
        matidx_arr_cp[matidx_arr_cp==i] = roughness_here
    return matidx_arr_cp

def render_passes_CYCLES(depth_file_output, normal_file_output, albedo_file_output, matidx_file_output, glossdir_file_output, args, rot_angles_list, diag_length=1., subfolder_name='gt', output_format='exr'):
    scaling_factor_unit2scene = diag_length / 1.
    scaling_facotr_scene2unit = 1. / diag_length

    scene = bpy.context.scene
    ######### filename for output ##############
    if 'ShapeNetCore' not in args.obj:
        model_identifier = args.obj.split('/')[-1].split('.')[0]
    else:
        model_identifier = args.obj.split('/')[-3]
    fp = os.path.join(args.output_folder, subfolder_name, model_identifier)
    scene.render.image_settings.file_format = 'PNG'  # set output format to .png

    # setup camera and render
    cam_init_location = (0.0 * scaling_factor_unit2scene, 0.5 * scaling_factor_unit2scene, 0.0 * scaling_factor_unit2scene)
    cam = get_default_camera()
    cam.location = cam_init_location
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = args.orth_scale * scaling_factor_unit2scene
    cam.data.clip_start = 0
    cam.data.clip_end = 100 # a value that is large enough
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    b_empty = get_lookat_target(cam)
    cam_constraint.target = b_empty # track to a empty object at the origin

    # setup light
    sun_lamp = setup_sunlamp(b_empty)

    for xyz_angle in rot_angles_list:
        # rotate camera
        euler_rot_mat = euler2mat(radians(xyz_angle[0]), radians(xyz_angle[1]), radians(xyz_angle[2]), 'sxyz')
        new_cam_location = np.dot(euler_rot_mat, np.array(cam_init_location))
        cam.location = new_cam_location
        # the sun lamp follows
        sun_lamp.location = new_cam_location

        scene.render.filepath = fp + '-rotx=%.2f_roty=%.2f_rotz=%.2f'%(xyz_angle[0], xyz_angle[1], xyz_angle[2])
        depth_file_output.file_slots[0].path = scene.render.filepath + "_depth"
        normal_file_output.file_slots[0].path = scene.render.filepath + "_normal"
        albedo_file_output.file_slots[0].path = scene.render.filepath + "_albedo"
        matidx_file_output.file_slots[0].path = scene.render.filepath + "_matidx"
        glossdir_file_output.file_slots[0].path = scene.render.filepath + "_glossdir"

        # render and write out
        bpy.ops.render.render(write_still=True, animation=False)  # render still
        
        depth_arr, hard_mask_arr = util.read_depth_and_get_mask(scene.render.filepath + "_depth0001.exr", depth_scaling_factor=scaling_facotr_scene2unit)
        normal_arr = util.read_normal(scene.render.filepath + "_normal0001.exr", mask_arr=hard_mask_arr)
        # in CYCLES, the normal is in world system, rotate it to camera system
        normal_cam_arr = util.transform_points(np.reshape(normal_arr, (-1, 3)), blender_camera_util.get_world2bcam_R_matrix_from_blender(cam))
        normal_cam_arr = np.reshape(normal_cam_arr, normal_arr.shape)
        normal_arr = normal_cam_arr
        albedo_arr = util.read_exr_image(scene.render.filepath + "_albedo0001.exr")
        matidx_arr = util.read_exr_image(scene.render.filepath + "_matidx0001.exr")[:,:,0]
        glossdir_arr = util.read_exr_image(scene.render.filepath + "_glossdir0001.exr")
        # and the clip value range
        depth_arr = np.clip(depth_arr, a_min=0, a_max=1)
        normal_arr = np.clip(normal_arr, a_min=-1, a_max=1)
        albedo_arr = np.clip(albedo_arr, a_min=0, a_max=1)
        glossdir_arr = np.clip(glossdir_arr, a_min=0, a_max=1)

        # 
        roughness_arr = assemble_roughness_map(matidx_arr)

        # write out passes
        if output_format == 'exr':
            util.write_exr_image(depth_arr, scene.render.filepath + "_depth.exr")
            #util.write_exr_image(xyz_sworld_arr, scene.render.filepath + "_wxyz.exr")
            util.write_exr_image(normal_arr, scene.render.filepath + "_normal.exr")
            #util.write_exr_image(normal_sworld_arr, scene.render.filepath + "_wnormal.exr")
            util.write_exr_image(albedo_arr, scene.render.filepath + "_albedo.exr")
            util.write_exr_image(hard_mask_arr, scene.render.filepath + "_mask.exr")
            util.write_exr_image(glossdir_arr, scene.render.filepath + "_glossdir.exr")
            util.write_exr_image(roughness_arr, scene.render.filepath + "_roughness.exr")
        elif output_format == 'png':
            depth_arr = np.array(depth_arr*255, dtype=np.uint8)
            depth_pil = Image.fromarray(depth_arr)
            depth_pil.save(scene.render.filepath + "_depth.png")

            normal_arr = np.array((normal_arr+1)/2.*255, dtype=np.uint8)
            normal_pil = Image.fromarray(normal_arr)
            normal_pil.save(scene.render.filepath + "_normal.png")
        
            albedo_arr = np.array(albedo_arr*255, dtype=np.uint8)
            albedo_pil = Image.fromarray(albedo_arr)
            albedo_pil.save(scene.render.filepath + "_albedo.png")

            hard_mask_arr = np.array(hard_mask_arr*255, dtype=np.uint8)
            mask_pil = Image.fromarray(hard_mask_arr)
            mask_pil.save(scene.render.filepath + "_mask.png")

            glossdir_arr = np.array(glossdir_arr*255, dtype=np.uint8)
            glossdir_pil = Image.fromarray(glossdir_arr)
            glossdir_pil.save(scene.render.filepath + "_glossdir.png")

            roughness_arr = np.array(roughness_arr*255, dtype=np.uint8)
            roughness_pil = Image.fromarray(roughness_arr)
            roughness_pil.save(scene.render.filepath + "_roughness.png")

        # remove renderings
        #os.remove(scene.render.filepath+'.png')
        os.remove(scene.render.filepath + "_normal0001.exr")
        os.remove(scene.render.filepath + "_depth0001.exr")
        os.remove(scene.render.filepath + "_albedo0001.exr")
        os.remove(scene.render.filepath + "_matidx0001.exr")
        os.remove(scene.render.filepath + "_glossdir0001.exr")
        #os.remove('Image0001.exr')
   

def bcam2world_RT_matrixes(rot_angles_list):
    scene = bpy.context.scene
    ######### filename for output ##############

    # setup camera and render
    cam_init_location = (0., 0.5, 0.)
    cam = get_default_camera()
    cam.location = cam_init_location
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = 1
    cam.data.clip_start = 0
    cam.data.clip_end = 100 # a value that is large enough
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    b_empty = get_lookat_target(cam)
    cam_constraint.target = b_empty # track to a empty object at the origin

    RT_dict = {}

    for xyz_angle in rot_angles_list:

        # rotate camera
        euler_rot_mat = euler2mat(radians(xyz_angle[0]), radians(xyz_angle[1]), radians(xyz_angle[2]), 'sxyz')
        new_cam_location = np.dot(euler_rot_mat, np.array(cam_init_location))
        cam.location = new_cam_location

        bpy.context.scene.update() # NOTE: important! Not doing rendering (which updates the scene) but we need to update the scene after we reset the camera
        
        # transform from depth to 3D world point cloud
        RT_bcam2world = blender_camera_util.get_bcam2world_RT_matrix_from_blender(cam)
        key_tuple = tuple(xyz_angle)
        RT_dict[key_tuple] = RT_bcam2world

    return RT_dict

import pickle
if __name__ == '__main__':
    clear_scene_objects()

    rot_angles_list = []
    for x_angle in range(0, 60):
        for z_angle in range(0, 361):
            rot_x_angle = x_angle
            rot_y_angle = 0 # do not rot around y, no in-plane rotation
            rot_z_angle = z_angle
            rot_angles_list.append([rot_x_angle, rot_y_angle, rot_z_angle])

    RT_mat_dict = bcam2world_RT_matrixes(rot_angles_list)
    pickle.dump( RT_mat_dict, open( "RT_matrixes_dict.pickle", "wb" ) )

    with open('RT_matrixes_dict.pickle', 'rb') as handle:
        b = pickle.load(handle)