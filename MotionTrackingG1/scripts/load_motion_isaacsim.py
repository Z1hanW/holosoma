import os, sys
sys.path.insert(1, '/home/jiashunw/research/protomotions/poselib')


from poselib.skeleton.skeleton3d import (
    SkeletonMotion,
    SkeletonState,
    SkeletonTree,
)
from poselib.core.rotation3d import *

import omni
from pxr import Gf, UsdGeom, UsdShade, Vt

stage = omni.usd.get_context().get_stage()
UsdGeom.Xform.Define(stage, "/World/Xform/char0")
stage.GetPrimAtPath("/World/Xform/char0").GetReferences().AddReference("/home/jiashunw/research/parkour/humanoid.usd")

scale = 100
fps = 30
folder = "/home/jiashunw/research/parkour/output/recordings/7"
file = os.path.join(folder, f"trajectory_poselib_0.npy")
source_motion = SkeletonMotion.from_file(file)
source_tpose = SkeletonState.zero_pose(source_motion.skeleton_tree)

local_rotation = source_motion.local_rotation
local_rotation = local_rotation.numpy().copy()
local_translation = source_motion.local_translation
local_translation = local_translation.numpy().copy() * scale
global_translation = source_motion.global_translation.numpy().copy() * scale
global_rotation = source_motion.global_rotation.numpy().copy()

num_frames = len(local_rotation)
print(num_frames)
stage.SetEndTimeCode(num_frames)
stage.SetFramesPerSecond(fps)
stage.SetTimeCodesPerSecond(fps)

import xml.etree.ElementTree as ET

urdf_file_path = "/home/jiashunw/research/parkour/motion_tracking/data/assets/urdf/parkour_line10_seed135_eval_7.urdf"

# Parse the URDF XML
tree = ET.parse(urdf_file_path)
root = tree.getroot()

# Parse boxes from the URDF
parsed_boxes = []  # List to store parsed box data: [(size, xyz, rpy), ...]
for link in root.findall("link"):
    for visual in link.findall("visual"):
        origin = visual.find("origin")
        xyz = list(map(float, origin.get("xyz").split()))
        rpy = list(map(float, origin.get("rpy").split()))
        box = visual.find("geometry").find("box")
        size = list(map(float, box.get("size").split()))
        parsed_boxes.append((size, xyz, rpy))

# Add parsed boxes to the USD stage
for i, (size, xyz, rpy) in enumerate(parsed_boxes):
    # Apply scale factor to position and size
    scaled_xyz = [coord * scale for coord in xyz]
    scaled_size = [dim * scale for dim in size]

    # Create a transform for the box
    box_path = f"/World/Box{i}"
    box_prim = UsdGeom.Xform.Define(stage, box_path)
    
    # Set translation and rotation
    translate_op = box_prim.AddTranslateOp()
    translate_op.Set(value=Gf.Vec3d(*scaled_xyz))
    
    rotate_op = box_prim.AddRotateXYZOp()
    rotate_op.Set(value=Gf.Vec3f(
        rpy[0] * 180.0 / 3.14159,  # Convert radians to degrees
        rpy[1] * 180.0 / 3.14159,
        rpy[2] * 180.0 / 3.14159
    ))

    # Create a cube with the correct size
    cube_geom = UsdGeom.Cube.Define(stage, f"{box_path}/Cube")

    # In USD, cubes have a default size of 1 and are scaled uniformly.
    # We scale the cube to match the given size.
    cube_geom.GetPrim().GetAttribute("size").Set(1.0)  # Keep cube default size
    scale_op = cube_geom.AddScaleOp()
    scale_op.Set(Gf.Vec3d(scaled_size[0], scaled_size[1], scaled_size[2]))


##### humanoid #####
for f in range(num_frames):
    prim = stage.GetPrimAtPath('/World/Xform/char0')
    children_prim = prim.GetChildren()
    num_bodies = local_translation.shape[1]
    for j in range(len(children_prim)):
        ops = UsdGeom.Xform.Get(stage, children_prim[j].GetPath()).GetOrderedXformOps()
        body_position = global_translation[f, j]
        body_rotation = global_rotation[f, j]

        ops[0].Set(value=Gf.Vec3f(body_position.tolist()), time=f)
        ops[1].Set(value=Gf.Quatf(body_rotation[3].item(), Gf.Vec3f(body_rotation[0:3].tolist())), time=f)
print("1")      

# Define a parent Xform node for the camera
camera_transform_path = '/World/camera_follow'
camera_transform = UsdGeom.Xform.Define(stage, camera_transform_path)
camera_transform.ClearXformOpOrder()

# Define the camera under the Xform node
camera_path = f'{camera_transform_path}/Camera'
camera = UsdGeom.Camera.Define(stage, camera_path)
camera.ClearXformOpOrder()

# No local offset for the camera itself, all done by parent
camera_local_translate = camera.AddTranslateOp()
camera_local_translate.Set(Gf.Vec3d(0,0,0))

# Set the up vector to align with Z being up
up_vector = Gf.Vec3d(0,0,1)

# Choose a fixed offset from the pelvis: behind (-Y) and above (+Z)
camera_x_trans = 0.
camera_y_trans = -2000
camera_z_trans = 1000.
camera_offset = Gf.Vec3f(camera_x_trans, camera_y_trans, camera_z_trans)
additional_roll = Gf.Rotation(Gf.Vec3d(1, 0, 0), 90)  # 90 degrees around X-axis

for f in range(num_frames):
    # Get pelvis world position at frame f
    pelvis_transform = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Xform/char0/Pelvis")).ComputeLocalToWorldTransform(f)
    pelvis_position = pelvis_transform.ExtractTranslation()
    
    # Compute camera world position from pelvis + offset
    camera_world_pos = Gf.Vec3d(
        pelvis_position[0] + camera_offset[0],
        pelvis_position[1] + camera_offset[1],
        100 + camera_offset[2]
    )
    
    # Camera should look directly at the pelvis
    look_target = pelvis_position
    look_target[2] = 100
    
    # Create a look-at matrix using Z as up
    look_at_matrix = Gf.Matrix4d().SetLookAt(
        camera_world_pos, # from
        look_target,      # to
        up_vector         # up
    )
    
    # Extract rotation from the look-at matrix
    rotation = look_at_matrix.ExtractRotation()
    rotation_quat = rotation.GetQuaternion()
    
    # Update camera_transform each frame
    camera_transform.ClearXformOpOrder()
    translate_op = camera_transform.AddTranslateOp()
    translate_op.Set(camera_world_pos, time=f)

    orient_op = camera_transform.AddOrientOp()
    
    if camera_y_trans>=0:
        orient_op.Set(Gf.Quatf(rotation_quat.GetReal(), Gf.Vec3f(rotation_quat.GetImaginary())), time=f)
    else:
        orient_op.Set(Gf.Quatf(rotation_quat.GetReal(), -Gf.Vec3f(rotation_quat.GetImaginary())), time=f)


folder = "/home/jiashunw/research/parkour/output/recordings/7"
file = os.path.join(folder, f"trajectory_poselib_5.usd")
omni.usd.get_context().save_as_stage(file)
print("2")      