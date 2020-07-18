#!/usr/bin/env bash
# arg 1: path to PartNet dataset ; arg 2: folder to save partial and complete mesh/point cloud
python create_partial_mesh.py --src $1 --dest $2

fp_shape_mesh=$2"-shape_mesh"
fp_partial_mesh=$2"-partial_mesh"
fp_shape_pc=$2"-shape_mesh_scan_pc"
fp_partial_pc=$2"-partial_mesh_scan_pc"

cd blender_render

# scan complete shape
find $fp_shape_mesh -name '*.obj' -print0 | xargs -0 -n1 -P10 -I {} blender --background --python generate_partnet_point_cloud.py -- --output_folder $fp_shape_pc {}

# scan partial shape, 4 views each
find $fp_partial_mesh -name '*.obj' -print0 | xargs -0 -n1 -P10 -I {} blender --background --python generate_partnetpartial_point_cloud.py -- --nb_scan 4 --nb_view 1 --orth_scale 2 --output_folder $fp_partial_pc {}
