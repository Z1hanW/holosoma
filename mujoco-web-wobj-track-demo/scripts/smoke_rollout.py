#!/usr/bin/env python3
"""Offline smoke rollout for the MuJoCo web tracking demo."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import onnxruntime

try:
    import mujoco as mj
except ModuleNotFoundError as exc:
    raise SystemExit(
        "MuJoCo Python bindings are not available in the current interpreter. "
        "Use `/home/ubuntu/.holosoma_deps/miniconda3/envs/sim/bin/python` to run this script."
    ) from exc


def quat_apply_wxyz(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    vec = np.asarray(vec, dtype=np.float64)
    xyz = quat[1:]
    w = quat[0]
    t = np.cross(xyz, vec) * 2.0
    return vec + w * t + np.cross(xyz, t)


def quat_inverse_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)


def quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(a, dtype=np.float64)
    w2, x2, y2, z2 = np.asarray(b, dtype=np.float64)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def subtract_frame_transforms_wxyz(frame_quat_wxyz: np.ndarray, target_quat_wxyz: np.ndarray) -> np.ndarray:
    return quat_mul_wxyz(quat_inverse_wxyz(frame_quat_wxyz), target_quat_wxyz)


def quat_to_rot6d_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-8:
        raise ValueError("Quaternion norm too small.")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return rot[:, :2].reshape(-1)


def yaw_from_quat_wxyz(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64)
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def yaw_quat_wxyz(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5)], dtype=np.float64)


def remove_yaw_offset_wxyz(quat_wxyz: np.ndarray, yaw_offset: float) -> np.ndarray:
    if abs(yaw_offset) < 1e-8:
        return np.asarray(quat_wxyz, dtype=np.float64).copy()
    return quat_mul_wxyz(yaw_quat_wxyz(-yaw_offset), quat_wxyz)


def quat_rotate_inverse_wxyz(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    return quat_apply_wxyz(quat_inverse_wxyz(quat_wxyz), vec)


def clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def vec3_from_buffer(buffer_like, index: int) -> np.ndarray:
    base = index * 3
    return np.asarray(buffer_like[base : base + 3], dtype=np.float64)


def quat_wxyz_from_buffer(buffer_like, index: int) -> np.ndarray:
    base = index * 4
    return np.asarray(buffer_like[base : base + 4], dtype=np.float64)


def _load_manifest(asset_root: Path) -> dict:
    manifest_path = asset_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _load_clip_config(asset_root: Path, clip_id: str | None) -> tuple[str, dict]:
    manifest = _load_manifest(asset_root)
    clips = manifest.get("clips", [])
    if not clips:
        raise RuntimeError(f"No clips found under {asset_root}")
    selected_id = clip_id or manifest.get("default_clip_id") or clips[0]["id"]
    for clip in clips:
        if clip["id"] == selected_id:
            config_path = asset_root / clip["config_path"]
            return clip["id"], json.loads(config_path.read_text(encoding="utf-8"))
    raise RuntimeError(f"Unknown clip id {selected_id!r}")


class OfflineWebRollout:
    def __init__(self, asset_root: Path, config: dict):
        self.asset_root = asset_root
        self.config = config
        scene_path = (asset_root / config["scene_path"]).resolve()
        previous_cwd = Path.cwd()
        try:
            os.chdir(scene_path.parent)
            self.model = mj.MjModel.from_xml_path(str(scene_path))
        finally:
            os.chdir(previous_cwd)
        self.data = mj.MjData(self.model)
        self.session = onnxruntime.InferenceSession(str(asset_root / config["model_path"]))

        body_enum = mj.mjtObj.mjOBJ_BODY
        joint_enum = mj.mjtObj.mjOBJ_JOINT
        actuator_enum = mj.mjtObj.mjOBJ_ACTUATOR

        root_body_id = mj.mj_name2id(self.model, body_enum, "pelvis")
        self.ref_body_id = mj.mj_name2id(self.model, body_enum, config["ref_body_name"])
        self.object_body_id = mj.mj_name2id(self.model, body_enum, config["object_body_name"])
        self.root_joint_qpos_adr = int(self.model.jnt_qposadr[self.model.body_jntadr[root_body_id]])
        self.root_joint_qvel_adr = int(self.model.jnt_dofadr[self.model.body_jntadr[root_body_id]])

        self.object_joint_qpos_adr = None
        self.object_joint_qvel_adr = None
        if self.object_body_id >= 0 and self.model.body_jntnum[self.object_body_id] > 0:
            self.object_joint_qpos_adr = int(self.model.jnt_qposadr[self.model.body_jntadr[self.object_body_id]])
            self.object_joint_qvel_adr = int(self.model.jnt_dofadr[self.model.body_jntadr[self.object_body_id]])

        self.joint_bindings = []
        for index, name in enumerate(config["dof_names"]):
            joint_id = mj.mj_name2id(self.model, joint_enum, name)
            actuator_id = mj.mj_name2id(self.model, actuator_enum, name)
            self.joint_bindings.append(
                {
                    "index": index,
                    "name": name,
                    "joint_id": joint_id,
                    "actuator_id": actuator_id,
                    "qpos_adr": int(self.model.jnt_qposadr[joint_id]),
                    "qvel_adr": int(self.model.jnt_dofadr[joint_id]),
                    "range_min": float(self.model.jnt_range[joint_id, 0]),
                    "range_max": float(self.model.jnt_range[joint_id, 1]),
                    "ctrl_limit": (
                        float(
                            max(
                                abs(self.model.actuator_ctrlrange[actuator_id, 0]),
                                abs(self.model.actuator_ctrlrange[actuator_id, 1]),
                            )
                        )
                        if actuator_id >= 0
                        else float("inf")
                    ),
                }
            )

        self.last_policy_action = np.zeros(len(self.joint_bindings), dtype=np.float32)
        self.current_torques = np.zeros(len(self.joint_bindings), dtype=np.float32)
        self.motion_command_t = None
        self.ref_quat_xyzw_t = None
        self.ref_pos_xyz_t = None
        self.motion_timestep = 0
        self.current_step = 0
        self.robot_yaw_offset = 0.0
        self.motion_yaw_offset = 0.0

    def reset(self) -> None:
        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0

        motion = self.config["motion"]
        root_quat_wxyz = np.asarray(motion["initial_root_quat_wxyz"], dtype=np.float64)
        self.data.qpos[self.root_joint_qpos_adr : self.root_joint_qpos_adr + 3] = motion["initial_root_pos_w"]
        self.data.qpos[self.root_joint_qpos_adr + 3 : self.root_joint_qpos_adr + 7] = root_quat_wxyz
        self.data.qvel[self.root_joint_qvel_adr : self.root_joint_qvel_adr + 3] = motion.get(
            "initial_root_lin_vel_w", [0.0, 0.0, 0.0]
        )
        self.data.qvel[self.root_joint_qvel_adr + 3 : self.root_joint_qvel_adr + 6] = quat_rotate_inverse_wxyz(
            root_quat_wxyz,
            np.asarray(motion.get("initial_root_ang_vel_w", [0.0, 0.0, 0.0]), dtype=np.float64),
        )

        reset_joint_pos = motion.get("reset_joint_pos", motion["initial_joint_pos"])
        reset_joint_vel = motion.get("reset_joint_vel", motion["initial_joint_vel"])
        for binding in self.joint_bindings:
            idx = binding["index"]
            self.data.qpos[binding["qpos_adr"]] = reset_joint_pos[idx]
            self.data.qvel[binding["qvel_adr"]] = reset_joint_vel[idx]

        if self.object_joint_qpos_adr is not None and motion.get("initial_object_pos_w") is not None:
            self.data.qpos[self.object_joint_qpos_adr : self.object_joint_qpos_adr + 3] = motion["initial_object_pos_w"]
            object_quat_wxyz = np.asarray(motion["initial_object_quat_wxyz"], dtype=np.float64)
            self.data.qpos[self.object_joint_qpos_adr + 3 : self.object_joint_qpos_adr + 7] = object_quat_wxyz
        else:
            object_quat_wxyz = None
        if self.object_joint_qvel_adr is not None:
            self.data.qvel[self.object_joint_qvel_adr : self.object_joint_qvel_adr + 3] = motion.get(
                "initial_object_lin_vel_w", [0.0, 0.0, 0.0]
            )
            if object_quat_wxyz is None:
                object_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            self.data.qvel[self.object_joint_qvel_adr + 3 : self.object_joint_qvel_adr + 6] = quat_rotate_inverse_wxyz(
                object_quat_wxyz,
                np.asarray(motion.get("initial_object_ang_vel_w", [0.0, 0.0, 0.0]), dtype=np.float64),
            )

        self.motion_command_t = np.concatenate(
            [
                np.asarray(motion["initial_joint_pos"], dtype=np.float32),
                np.asarray(motion["initial_joint_vel"], dtype=np.float32),
            ]
        )
        ref_quat_wxyz = np.asarray(motion["initial_ref_quat_wxyz"], dtype=np.float32)
        self.ref_quat_xyzw_t = ref_quat_wxyz[[1, 2, 3, 0]].astype(np.float32)
        self.ref_pos_xyz_t = np.asarray(motion["initial_ref_pos_w"], dtype=np.float32)
        self.last_policy_action.fill(0.0)
        self.current_torques.fill(0.0)
        self.motion_timestep = 0
        self.current_step = 0
        self.robot_yaw_offset = 0.0
        self.motion_yaw_offset = 0.0

        mj.mj_forward(self.model, self.data)
        self._capture_yaw_offsets()

    def _get_root_pose_world(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.array(self.data.qpos[self.root_joint_qpos_adr : self.root_joint_qpos_adr + 3], dtype=np.float64, copy=True),
            np.array(
                self.data.qpos[self.root_joint_qpos_adr + 3 : self.root_joint_qpos_adr + 7],
                dtype=np.float64,
                copy=True,
            ),
        )

    def _get_root_state_world(self) -> dict:
        root_pos, root_quat = self._get_root_pose_world()
        angvel_local = np.asarray(self.data.qvel[self.root_joint_qvel_adr + 3 : self.root_joint_qvel_adr + 6], dtype=np.float64)
        return {
            "pos": root_pos,
            "quat_wxyz": root_quat,
            "lin_vel": np.asarray(self.data.qvel[self.root_joint_qvel_adr : self.root_joint_qvel_adr + 3], dtype=np.float64),
            "ang_vel_world": quat_apply_wxyz(root_quat, angvel_local),
        }

    def _get_ref_pose_world(self) -> dict:
        return {
            "pos": np.array(self.data.xpos[self.ref_body_id], dtype=np.float64, copy=True),
            "quat_wxyz": np.array(self.data.xquat[self.ref_body_id], dtype=np.float64, copy=True),
        }

    def _get_object_state_world(self) -> dict:
        quat_wxyz = np.asarray(self.data.xquat[self.object_body_id], dtype=np.float64)
        lin_vel = np.zeros(3, dtype=np.float64)
        ang_vel_world = np.zeros(3, dtype=np.float64)
        if self.object_joint_qvel_adr is not None:
            lin_vel = np.asarray(self.data.qvel[self.object_joint_qvel_adr : self.object_joint_qvel_adr + 3], dtype=np.float64)
            ang_vel_local = np.asarray(
                self.data.qvel[self.object_joint_qvel_adr + 3 : self.object_joint_qvel_adr + 6], dtype=np.float64
            )
            ang_vel_world = quat_apply_wxyz(quat_wxyz, ang_vel_local)
        return {
            "pos": np.array(self.data.xpos[self.object_body_id], dtype=np.float64, copy=True),
            "quat_wxyz": quat_wxyz,
            "lin_vel": lin_vel,
            "ang_vel_world": ang_vel_world,
        }

    def _capture_yaw_offsets(self) -> None:
        robot_ref = self._get_ref_pose_world()
        self.robot_yaw_offset = yaw_from_quat_wxyz(robot_ref["quat_wxyz"])
        motion_ref_wxyz = np.asarray([self.ref_quat_xyzw_t[3], *self.ref_quat_xyzw_t[:3]], dtype=np.float64)
        self.motion_yaw_offset = yaw_from_quat_wxyz(motion_ref_wxyz)

    def _pose_in_robot_ref_frame(
        self,
        robot_ref_pos_w: np.ndarray,
        robot_ref_quat_wxyz: np.ndarray,
        target_pos_w: np.ndarray,
        target_quat_wxyz: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        rel_pos_w = target_pos_w - robot_ref_pos_w
        rel_pos_b = quat_apply_wxyz(quat_inverse_wxyz(robot_ref_quat_wxyz), rel_pos_w)
        rel_quat_b = subtract_frame_transforms_wxyz(robot_ref_quat_wxyz, target_quat_wxyz)
        return rel_pos_b, rel_quat_b

    def build_obs(self) -> np.ndarray:
        motion_index = min(self.motion_timestep, self.config["motion"]["frame_count"] - 1)
        root_state = self._get_root_state_world()
        if motion_index == 0 and bool(self.config.get("use_root_reference_at_clip_start", False)):
            root_pos, root_quat_wxyz = self._get_root_pose_world()
            robot_ref = {"pos": root_pos, "quat_wxyz": root_quat_wxyz}
        else:
            robot_ref = self._get_ref_pose_world()
        object_state = self._get_object_state_world()

        motion_ref_quat_wxyz = np.asarray([self.ref_quat_xyzw_t[3], *self.ref_quat_xyzw_t[:3]], dtype=np.float64)
        motion_ref_quat_wxyz = remove_yaw_offset_wxyz(motion_ref_quat_wxyz, self.motion_yaw_offset)
        robot_ref_ori_obs = remove_yaw_offset_wxyz(robot_ref["quat_wxyz"], self.robot_yaw_offset)
        motion_ref_ori_b = quat_to_rot6d_wxyz(subtract_frame_transforms_wxyz(robot_ref_ori_obs, motion_ref_quat_wxyz))
        base_ang_vel_b = quat_apply_wxyz(quat_inverse_wxyz(root_state["quat_wxyz"]), root_state["ang_vel_world"])

        target_object_pos_w = np.asarray(
            self.config["motion"]["object_pos_w"][motion_index], dtype=np.float64
        )
        target_object_quat_wxyz = np.asarray(
            self.config["motion"]["object_quat_wxyz"][motion_index], dtype=np.float64
        )
        target_object_size = np.asarray(
            self.config["motion"]["object_size"][motion_index], dtype=np.float64
        )

        target_object_pos_b, target_object_quat_b = self._pose_in_robot_ref_frame(
            robot_ref["pos"],
            robot_ref["quat_wxyz"],
            target_object_pos_w,
            target_object_quat_wxyz,
        )
        current_object_pos_b, current_object_quat_b = self._pose_in_robot_ref_frame(
            robot_ref["pos"],
            robot_ref["quat_wxyz"],
            object_state["pos"],
            object_state["quat_wxyz"],
        )
        obj_target_rot6d = quat_to_rot6d_wxyz(target_object_quat_b)
        obj_rot6d = quat_to_rot6d_wxyz(current_object_quat_b)
        obj_lin_vel_b = quat_apply_wxyz(
            quat_inverse_wxyz(robot_ref["quat_wxyz"]),
            object_state["lin_vel"] - robot_ref["pos"],
        )
        obj_ang_vel_b = quat_apply_wxyz(quat_inverse_wxyz(robot_ref["quat_wxyz"]), object_state["ang_vel_world"])

        dof_pos = np.zeros(len(self.joint_bindings), dtype=np.float32)
        dof_vel = np.zeros(len(self.joint_bindings), dtype=np.float32)
        for binding in self.joint_bindings:
            idx = binding["index"]
            dof_pos[idx] = self.data.qpos[binding["qpos_adr"]] - self.config["default_dof_angles"][idx]
            dof_vel[idx] = self.data.qvel[binding["qvel_adr"]]

        term_buffers = {
            "actions": self.last_policy_action.astype(np.float32, copy=False),
            "base_ang_vel": np.asarray(base_ang_vel_b, dtype=np.float32),
            "dof_pos": dof_pos.astype(np.float32, copy=False),
            "dof_vel": dof_vel.astype(np.float32, copy=False),
            "motion_command": self.motion_command_t.astype(np.float32, copy=False),
            "motion_ref_ori_b": np.asarray(motion_ref_ori_b, dtype=np.float32),
            "obj_ang_vel_b": np.asarray(obj_ang_vel_b, dtype=np.float32),
            "obj_lin_vel_b": np.asarray(obj_lin_vel_b, dtype=np.float32),
            "obj_ori_b": np.asarray(obj_rot6d, dtype=np.float32),
            "obj_pos_b": np.asarray(current_object_pos_b, dtype=np.float32),
            "obj_target_pose_size_b": np.asarray(
                np.concatenate([target_object_pos_b, obj_target_rot6d, target_object_size]),
                dtype=np.float32,
            ),
        }

        term_order = self.config["observation"]["actor_obs_terms_sorted"]
        obs = np.zeros(self.config["onnx"]["obs_dim"], dtype=np.float32)
        cursor = 0
        for term in term_order:
            buffer = term_buffers[term]
            obs[cursor : cursor + buffer.size] = buffer
            cursor += buffer.size
        if cursor != obs.size:
            raise RuntimeError(f"Observation size mismatch: expected {obs.size}, filled {cursor}")
        return obs

    def step_policy(self) -> dict:
        obs = self.build_obs()
        outputs = self.session.run(
            None,
            {
                "obs": obs.reshape(1, -1),
                "time_step": np.asarray([[self.motion_timestep]], dtype=np.float32),
            },
        )
        output_names = [item.name for item in self.session.get_outputs()]
        output_dict = {name: value for name, value in zip(output_names, outputs, strict=False)}

        raw_actions = np.asarray(output_dict["actions"], dtype=np.float32).reshape(-1)
        clip_threshold = float(self.config["control"]["clip_actions_threshold"])
        self.last_policy_action[:] = np.clip(raw_actions, -clip_threshold, clip_threshold)

        if "joint_pos" in output_dict and "joint_vel" in output_dict:
            self.motion_command_t[: len(self.joint_bindings)] = np.asarray(output_dict["joint_pos"], dtype=np.float32).reshape(-1)
            self.motion_command_t[len(self.joint_bindings) :] = np.asarray(output_dict["joint_vel"], dtype=np.float32).reshape(-1)
        if "ref_quat_xyzw" in output_dict:
            self.ref_quat_xyzw_t[:] = np.asarray(output_dict["ref_quat_xyzw"], dtype=np.float32).reshape(-1)
        if "ref_pos_xyz" in output_dict:
            self.ref_pos_xyz_t[:] = np.asarray(output_dict["ref_pos_xyz"], dtype=np.float32).reshape(-1)

        for binding in self.joint_bindings:
            idx = binding["index"]
            scale = self.config["control"]["policy_action_scales"][idx]
            q_current = float(self.data.qpos[binding["qpos_adr"]])
            dq_current = float(self.data.qvel[binding["qvel_adr"]])
            q_target = self.config["default_dof_angles"][idx] + float(self.last_policy_action[idx]) * float(scale)
            q_target = clamp(q_target, binding["range_min"], binding["range_max"])
            torque = (
                float(self.config["kp"][idx]) * (q_target - q_current)
                + float(self.config["kd"][idx]) * (0.0 - dq_current)
            )
            self.current_torques[idx] = clamp(torque, -binding["ctrl_limit"], binding["ctrl_limit"])

        for _ in range(int(self.config["control"]["control_decimation"])):
            for binding in self.joint_bindings:
                if binding["actuator_id"] >= 0:
                    self.data.ctrl[binding["actuator_id"]] = self.current_torques[binding["index"]]
            mj.mj_step(self.model, self.data)
            self.current_step += 1

        self.motion_timestep += 1
        return {
            "action_rms": float(np.sqrt(np.mean(np.square(self.last_policy_action)))),
            "torque_rms": float(np.sqrt(np.mean(np.square(self.current_torques)))),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "public" / "demo-assets",
    )
    parser.add_argument("--clip-id", type=str, default=None)
    parser.add_argument("--policy-steps", type=int, default=150)
    parser.add_argument("--min-root-xy-displacement", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asset_root = args.asset_root.expanduser().resolve()
    clip_id, config = _load_clip_config(asset_root, args.clip_id)
    rollout = OfflineWebRollout(asset_root, config)
    rollout.reset()

    root_start, _ = rollout._get_root_pose_world()
    object_start = rollout._get_object_state_world()["pos"].copy()
    action_rms_values: list[float] = []
    torque_rms_values: list[float] = []
    root_xy_positions = [root_start[:2].copy()]
    root_z_positions = [float(root_start[2])]
    object_xy_positions = [object_start[:2].copy()]
    object_root_xy_offsets = [float(np.linalg.norm(object_start[:2] - root_start[:2]))]

    steps = min(int(args.policy_steps), int(config["motion"]["frame_count"]))
    for _ in range(steps):
        metrics = rollout.step_policy()
        root_pos, _ = rollout._get_root_pose_world()
        object_pos = rollout._get_object_state_world()["pos"]
        action_rms_values.append(metrics["action_rms"])
        torque_rms_values.append(metrics["torque_rms"])
        root_xy_positions.append(root_pos[:2].copy())
        root_z_positions.append(float(root_pos[2]))
        object_xy_positions.append(object_pos[:2].copy())
        object_root_xy_offsets.append(float(np.linalg.norm(object_pos[:2] - root_pos[:2])))

    root_end, _ = rollout._get_root_pose_world()
    object_end = rollout._get_object_state_world()["pos"].copy()
    root_xy_disp = float(np.linalg.norm(root_end[:2] - root_start[:2]))
    object_xy_disp = float(np.linalg.norm(object_end[:2] - object_start[:2]))
    carry_xy_gap_delta = float(object_root_xy_offsets[-1] - object_root_xy_offsets[0])
    summary = {
        "clip_id": clip_id,
        "policy_steps": steps,
        "motion_timestep": rollout.motion_timestep,
        "root_start_xyz": root_start.tolist(),
        "root_end_xyz": root_end.tolist(),
        "object_start_xyz": object_start.tolist(),
        "object_end_xyz": object_end.tolist(),
        "root_xy_displacement_m": root_xy_disp,
        "object_xy_displacement_m": object_xy_disp,
        "object_root_xy_offset_start_m": object_root_xy_offsets[0],
        "object_root_xy_offset_end_m": object_root_xy_offsets[-1],
        "object_root_xy_offset_delta_m": carry_xy_gap_delta,
        "root_z_min_m": float(min(root_z_positions)),
        "root_z_max_m": float(max(root_z_positions)),
        "action_rms_mean": float(np.mean(action_rms_values)) if action_rms_values else 0.0,
        "action_rms_max": float(np.max(action_rms_values)) if action_rms_values else 0.0,
        "torque_rms_mean": float(np.mean(torque_rms_values)) if torque_rms_values else 0.0,
        "torque_rms_max": float(np.max(torque_rms_values)) if torque_rms_values else 0.0,
        "actor_obs_terms_sorted": config["observation"]["actor_obs_terms_sorted"],
    }
    print(json.dumps(summary, indent=2))

    if not np.isfinite(root_xy_disp):
        raise SystemExit("Offline rollout produced non-finite root displacement.")
    if root_xy_disp < float(args.min_root_xy_displacement):
        raise SystemExit(
            f"Offline rollout root XY displacement {root_xy_disp:.4f} m is below the expected threshold "
            f"{args.min_root_xy_displacement:.4f} m."
        )


if __name__ == "__main__":
    main()
