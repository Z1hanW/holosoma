import torch
from copy import deepcopy

from isaac_utils import torch_utils, rotations

import numpy as np
import os
import yaml

from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
from poselib.core.rotation3d import *

from motion_tracking.utils.device_dtype_mixin import DeviceDtypeModuleMixin

import torch
from torch import nn, Tensor
from tqdm import tqdm
from torch import Tensor
from typing import List, Tuple, Any

from dataclasses import dataclass

from motion_tracking.utils.motion_lib import LoadedMotions

from typing import List

import numpy as np
import os
from pathlib import Path
import typer
import shutil
import torch
from tqdm import tqdm

from scipy.spatial.transform import Rotation as sRot
import xml.etree.ElementTree as ET
import numpy as np

from smpllib.smpllib.utils import TMP_SMPL_DIR
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES, SMPLH_BONE_ORDER_NAMES, SMPLH_MUJOCO_NAMES
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
from smpllib.utils import flags

from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import (
    SkeletonMotion,
    SkeletonState,
    SkeletonTree
)

import numpy as np


def compute_rotation_matrix(A, B):
    # Normalize vectors A and B
    A = A / np.linalg.norm(A)
    B = B / np.linalg.norm(B)
    
    # Calculate cosine and sine of the angle using dot product and determinant
    cos_theta = np.dot(A, B)
    sin_theta = np.cross(A, B)
    
    # Since A and B are 2D in xy-plane, np.cross will give a scalar (z-component of cross product)
    sin_theta = sin_theta[2] if isinstance(sin_theta, np.ndarray) else sin_theta
    
    # Rotation matrix around the z-axis
    R_z = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta,  cos_theta, 0],
        [0,          0,         1]
    ])
    
    theta_degrees = np.arctan2(sin_theta, cos_theta) * (180 / np.pi)

    return R_z, theta_degrees

# B = R @ A


def get_smpl_robot(motion_data, outputpath, xyz_transl, xy_rotatate_deg, need_rotate=False):
   
    humanoid_type = 'smpl'

    assert humanoid_type in ["smpl", "smplx", "smplh"], "Humanoid type must be one of smpl, smplx, smplh"
    append_name = humanoid_type

    if humanoid_type == "smpl":
        mujoco_joint_names = SMPL_MUJOCO_NAMES
        joint_names = SMPL_BONE_ORDER_NAMES
    elif humanoid_type == "smplx" or humanoid_type == "smplh":
        mujoco_joint_names = SMPLH_MUJOCO_NAMES
        joint_names = SMPLH_BONE_ORDER_NAMES
    else:
        raise NotImplementedError

    upright_start = True
    robot_cfg = {
                "mesh": False,
                "rel_joint_lm": True,
                "upright_start": upright_start,
                "remove_toe": False,
                "real_weight": True,
                "real_weight_porpotion_capsules": True,
                "real_weight_porpotion_boxes": True, 
                "replace_feet": True,
                "masterfoot": False,
                "big_ankle": True,
                "freeze_hand": False, 
                "box_body": False,
                "master_range": 50,
                "body_params": {},
                "joint_params": {},
                "geom_params": {},
                "actuator_params": {},
                "model": "smpl",
            }

    smpl_local_robot = LocalRobot(robot_cfg,)
    def _maybe_get(keys, default=None):
        for key in keys:
            try:
                if isinstance(motion_data, dict) and key in motion_data:
                    return motion_data[key]
                if key in motion_data:
                    return motion_data[key]
            except Exception:
                continue
        return default

    amass_pose = _maybe_get(["poses"])
    if amass_pose is None:
        raise KeyError("Motion data does not contain 'poses'.")

    amass_trans = _maybe_get(["trans", "transl"])
    if amass_trans is None:
        raise KeyError("Motion data does not contain 'trans' or 'transl'.")

    betas_arr = _maybe_get(["betas", "beta", "shape_params", "shape"])
    if betas_arr is None:
        betas_arr = np.zeros(16, dtype=np.float32)
    betas_arr = np.asarray(betas_arr, dtype=np.float32)
    if betas_arr.ndim > 1:
        betas_vec = betas_arr[0]
    else:
        betas_vec = betas_arr

    gender = _maybe_get(["gender"])
    if isinstance(gender, np.ndarray):
        gender = gender.item()
    if gender is None:
        gender = "neutral"

    if need_rotate:
        from scipy.spatial.transform import Rotation as sRot
        transform = sRot.from_euler('xyz', np.array([-np.pi / 2, 0, 0]), degrees=False)
        new_root_rot = (transform * sRot.from_rotvec(amass_pose[:, :3])).as_rotvec()
        amass_pose[:, :3] = new_root_rot
        amass_trans = amass_trans.dot(transform.as_matrix().T)


    rad = np.deg2rad(xy_rotatate_deg)
    if xy_rotatate_deg!=0:
        from scipy.spatial.transform import Rotation as sRot
        transform = sRot.from_euler('xyz', np.array([0, 0, rad]), degrees=False) #nishizhen 90du
        new_root_rot = (transform * sRot.from_rotvec(amass_pose[:, :3])).as_rotvec()
        amass_pose[:, :3] = new_root_rot
        amass_trans = amass_trans.dot(transform.as_matrix().T)

    mocap_fr = 30
    pose_aa = torch.tensor(amass_pose)  
    amass_trans = torch.tensor(amass_trans) + torch.tensor(xyz_transl)
    betas = torch.from_numpy(betas_vec)
    batch_size = pose_aa.shape[0]

    motion_data = {
        "pose_aa": pose_aa.numpy(),
        "trans": amass_trans.numpy(),
        "beta": betas.numpy(),
        "gender": gender,
    }

    smpl_2_mujoco = [
        joint_names.index(q) for q in mujoco_joint_names if q in joint_names
    ]
    from scipy.spatial.transform import Rotation as sRot
    if humanoid_type == "smpl":
        pose_aa = np.concatenate([motion_data["pose_aa"][:, :66], np.zeros((batch_size, 6))], axis=1)  # TODO: need to extract correct handle rotations instead of zero
        pose_quat = sRot.from_rotvec(pose_aa.reshape(-1, 3)).as_quat().reshape(batch_size, 24, 4)[..., smpl_2_mujoco, :]
    else:
        pose_aa = np.concatenate([motion_data['pose_aa'][:, :66], motion_data['pose_aa'][:, 75:]], axis=-1)
        pose_quat = sRot.from_rotvec(pose_aa.reshape(-1, 3)).as_quat().reshape(batch_size, 52, 4)[..., smpl_2_mujoco, :]

    if isinstance(gender, bytes):
        gender = gender.decode("utf-8")
    if gender == "neutral":
        gender_number = [0]
    elif gender == "male":
        gender_number = [1]
    elif gender == "female":
        gender_number = [2]
    else:
        import ipdb
        ipdb.set_trace()
        raise Exception("Gender Not Supported!!")


    smpl_local_robot.load_from_skeleton(betas=betas[None, ], gender=gender_number, objs_info=None)
    smpl_local_robot.write_xml(f"{TMP_SMPL_DIR}/smpl_humanoid_1.xml")
    skeleton_tree = SkeletonTree.from_mjcf(f"{TMP_SMPL_DIR}/smpl_humanoid_1.xml")
    # import pdb;pdb.set_trace()

    root_trans_offset = torch.from_numpy(motion_data["trans"]) + skeleton_tree.local_translation[0]

    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree, # This is the wrong skeleton tree (location wise) here, but it's fine since we only use the parent relationship here.
        torch.from_numpy(pose_quat),
        root_trans_offset,
        is_local=True
    )
    sk_motion = SkeletonMotion.from_skeleton_state(
        new_sk_state,
        fps=mocap_fr
    )

    if robot_cfg["upright_start"]:
        B = pose_aa.shape[0]
        pose_quat_global = (sRot.from_quat(sk_motion.global_rotation.reshape(-1, 4).numpy()) *sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(B, -1, 4)

        global_translations = sk_motion.global_translation
        # min_height = torch.min(global_translations[:, :, 0])  # take x axis (up axis) across all frames and joints

        trans = root_trans_offset.clone()
        # trans[:, 0] -= min_height  # subtract offset from root x axis for all frames

        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            torch.from_numpy(pose_quat_global),
            trans,
            is_local=False)

        new_sk_motion = SkeletonMotion.from_skeleton_state(
            new_sk_state,
            fps=mocap_fr
        )

        print(f"Saving to {outputpath}")
        new_sk_motion.to_file(str(outputpath))

        # Attach betas metadata so downstream consumers (e.g. dynamic shape loader) can recover it
        try:
            saved_motion = np.load(outputpath, allow_pickle=True).item()
            betas_meta = betas_vec.reshape(1, -1) if betas_vec.ndim == 1 else betas_vec
            saved_motion["betas"] = betas_meta
            saved_motion["beta"] = betas_vec
            saved_motion["gender"] = gender
            np.save(outputpath, saved_motion)
        except Exception as _save_meta_err:
            print(f"[get_smpl_robot] Warning: failed to add betas metadata to {outputpath}: {_save_meta_err}")
    return motion_data["trans"][-1]

def rpy_to_xyz_transform(rpy, xyz, rad):

    transform = sRot.from_euler('xyz', np.array([0, 0, rad]), degrees=False)
    
    yaw = float(rpy[2]) + rad  # Adding 90 degrees to the yaw component
    rpy[2] = yaw % (2 * np.pi)
    
    # Apply the additional rotation to the xyz coordinates
    xyz = transform.as_matrix() @ xyz
    
    return rpy, xyz

def modify_urdf(filename, output_name, degree, transl):
    tree = ET.parse(filename)
    root = tree.getroot()
    rad = np.deg2rad(degree)

    # Find all origins in both visual and collision elements
    for origin in root.findall('.//visual/origin') + root.findall('.//collision/origin'):
        if origin is not None:
            rpy = [float(x) for x in origin.get('rpy', '0 0 0').split()]
            xyz = [float(x) for x in origin.get('xyz', '0 0 0').split()]
            
            new_rpy, new_xyz = rpy_to_xyz_transform(rpy, xyz, rad)
            
            new_xyz = np.array(new_xyz) + np.array(transl)
            origin.set('rpy', ' '.join(f"{x:.4f}" for x in new_rpy))
            origin.set('xyz', ' '.join(f"{x:.4f}" for x in new_xyz))

    # Write the modified tree back to a new file
    tree.write(output_name)
    print(output_name)

import xml.etree.ElementTree as ET
from xml.dom import minidom
def merge_geometries(urdf_paths, output_path):
    # Initialize a new robot element
    new_root = ET.Element("robot", name="merged_object")
    new_link = ET.SubElement(new_root, "link", name="object")
    
    # Iterate over each URDF path
    for urdf_path in urdf_paths:
        # Parse the URDF file
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        
        # Find the 'link' element (assuming there is only one and it's named 'object')
        link_element = root.find(".//link[@name='object']")
        
        # If a link is found, proceed to extract and append visual and collision elements
        if link_element is not None:
            # Append visual and collision elements to the new link element
            for visual in link_element.findall("visual"):
                new_link.append(visual)
            for collision in link_element.findall("collision"):
                new_link.append(collision)

    # Convert the new root element into a tree and write to the specified output file
    new_tree = ET.ElementTree(new_root)
    new_tree.write(output_path, encoding='unicode', xml_declaration=True)


def compute_box_corners(size, origin):
    x, y, z = map(float, size.split())
    ox, oy, oz = map(float, origin['xyz'].split())
    rpy = list(map(float, origin['rpy'].split())) if 'rpy' in origin else (0.0, 0.0, 0.0)
    
    # Create rotation matrix from RPY (roll, pitch, yaw)
    R = RPY_to_rotation_matrix(*rpy)
    
    # Half sizes for corner calculation
    dx, dy, dz = x / 2, y / 2, z / 2
    corners = np.array([
        [dx, dy, dz],
        [dx, dy, -dz],
        [dx, -dy, dz],
        [dx, -dy, -dz],
        [-dx, dy, dz],
        [-dx, dy, -dz],
        [-dx, -dy, dz],
        [-dx, -dy, -dz]
    ])
    
    # Rotate and translate corners
    corners = np.dot(R, corners.T).T + [ox, oy, oz]
    face_centers = calculate_face_centers(corners)

    return corners, face_centers

def calculate_face_centers(corners):
    face_indices = [
        [0, 1, 3, 2],  # Front face
        [4, 5, 7, 6],  # Back face
        [0, 1, 5, 4],  # Top face
        [2, 3, 7, 6],  # Bottom face
        [0, 2, 6, 4],  # Left face
        [1, 3, 7, 5]   # Right face
    ]
    face_centers = []
    for indices in face_indices:
        face = corners[indices]
        center = face.mean(axis=0)
        face_centers.append(center)
    return np.array(face_centers)

def RPY_to_rotation_matrix(roll, pitch, yaw):
    # Compute rotation matrix from roll, pitch, yaw
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    
    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [   -sp,          cp*sr,          cp*cr]
    ])
    return R

def read_urdf(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    boxes = []
    all_corners = []
    all_face_points = []
    for visual in root.findall(".//visual"):
        geometry = visual.find('geometry')
        box = geometry.find('box')
        if box is not None:
            size = box.get('size')
            origin = visual.find('origin').attrib
            corners,face_points = compute_box_corners(size, origin)
            boxes.append({'size': size, 'origin': origin, 'corners': corners})
            all_corners.append(corners)
            all_face_points.append(face_points)
    
    return boxes, all_corners, all_face_points

def create_urdf_spheres_from_points(points, radius=0.05):
    """ Generate URDF XML content for spheres at given points.

    Args:
    points (np.array): Array of points where each row is [x, y, z].
    radius (float): Radius of each sphere.

    Returns:
    str: A string containing URDF XML content.
    """
    urdf_content = []
    for point in points:
        x, y, z = point
        sphere_visual = f"""
        <visual>
          <origin xyz="{x:.8f} {y:.8f} {z:.8f}"/>
          <geometry>
            <sphere radius="{radius}"/>
          </geometry>
        </visual>
        """
        urdf_content.append(sphere_visual)
    
    return "\n".join(urdf_content)

# Example usage
def save_points(filename, save_name, save=True):
    boxes, all_corners, all_face_points = read_urdf(filename)
    all_corners = np.vstack(all_corners)
    all_face_points = np.vstack(all_face_points)

    points = np.concatenate([all_face_points,all_corners],axis=0).astype(np.float32) # Example: generate 100 random 3D points
    
    if save:
        np.save(save_name,points)
    
    return points


def combine_loaded_motions(motion_data: List[str], height_adjust, resample) -> LoadedMotions:
    """ Combine multiple LoadedMotions instances from given file paths. """
    combined_data = None

    for data in motion_data:
        # data = torch.load(path)
        if combined_data is None:
            combined_data = data
        else:
            combined_data = merge_two_loaded_motions(combined_data, data)
    N = len(combined_data.motion_lengths)
    combined_data.motion_weights = combined_data.motion_weights/N
    if resample:
        start_value = 0.4
        end_value = 0.1
        step = (end_value - start_value) / 4
        for i in range(0,combined_data.motion_weights.shape[0],5):
            for j in range(5):
                combined_data.motion_weights[i+j] = start_value + j * step 

    # import pdb;pdb.set_trace()
    combined_data.sub_motion_to_motion = torch.arange(N)
    motions = combined_data.motions
        
    #js check
    for ii in range(len(motions)):
        # if ii==1:
        #     motions[ii].global_translation[...,0] += 6
        # elif ii==2:
        #     motions[ii].global_translation[...,0] += 12
        motions[ii].global_translation[...,2] += height_adjust[ii]
    return combined_data

def merge_two_loaded_motions(data1: LoadedMotions, data2: LoadedMotions) -> LoadedMotions:
    """ Merge two LoadedMotions instances into one. """
    # Concatenate tensor attributes
    motion_lengths = torch.cat([data1.motion_lengths, data2.motion_lengths], dim=0)
    motion_weights = torch.cat([data1.motion_weights, data2.motion_weights], dim=0)
    motion_timings = torch.cat([data1.motion_timings, data2.motion_timings], dim=0)
    motion_fps = torch.cat([data1.motion_fps, data2.motion_fps], dim=0)
    motion_dt = torch.cat([data1.motion_dt, data2.motion_dt], dim=0)
    motion_num_frames = torch.cat([data1.motion_num_frames, data2.motion_num_frames], dim=0)
    sub_motion_to_motion = torch.cat([data1.sub_motion_to_motion, data2.sub_motion_to_motion], dim=0)
    ref_respawn_offsets = torch.cat([data1.ref_respawn_offsets, data2.ref_respawn_offsets], dim=0)
    text_embeddings = torch.cat([data1.text_embeddings, data2.text_embeddings], dim=0)
    has_text_embeddings = torch.cat([data1.has_text_embeddings, data2.has_text_embeddings], dim=0)

    # Concatenate tuple attributes
    motions = data1.motions + data2.motions
    motion_files = data1.motion_files + data2.motion_files

    # Handle list of lists
    supported_object_names = data1.supported_object_names + data2.supported_object_names

    # Create a new LoadedMotions instance
    return LoadedMotions(
        motions=motions,
        motion_lengths=motion_lengths,
        motion_weights=motion_weights,
        motion_timings=motion_timings,
        motion_fps=motion_fps,
        motion_dt=motion_dt,
        motion_num_frames=motion_num_frames,
        motion_files=motion_files,
        sub_motion_to_motion=sub_motion_to_motion,
        ref_respawn_offsets=ref_respawn_offsets,
        text_embeddings=text_embeddings,
        has_text_embeddings=has_text_embeddings,
        supported_object_names=supported_object_names
    )
