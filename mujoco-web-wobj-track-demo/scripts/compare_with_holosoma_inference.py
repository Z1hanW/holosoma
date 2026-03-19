#!/usr/bin/env python3
"""Compare the staged web rollout against current holosoma_inference WBT semantics."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

try:
    import mujoco as mj
except ModuleNotFoundError as exc:
    raise SystemExit(
        "MuJoCo Python bindings are not available in the current interpreter. "
        "Use `/home/ubuntu/.holosoma_deps/miniconda3/envs/sim/bin/python` to run this script."
    ) from exc

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma"))
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma_inference"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from holosoma_inference.config.config_values.inference import DEFAULTS as INFERENCE_DEFAULTS
from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy
from smoke_rollout import (
    OfflineWebRollout,
    _load_clip_config,
    clamp,
    quat_apply_wxyz,
)


class DummyInterface:
    def __init__(self) -> None:
        self.backend = "dummy"
        self.no_action = 0
        self.kp_level = 1.0
        self.kd_level = 1.0
        self.robot_config = None
        self.state: np.ndarray | None = None

    def get_low_state(self) -> np.ndarray | None:
        return None if self.state is None else np.array(self.state, copy=True)

    def send_low_command(self, *args, **kwargs) -> None:  # noqa: D401, ANN002, ANN003
        del args, kwargs

    def close(self) -> None:
        return


def _sim_root_state(rollout: OfflineWebRollout) -> np.ndarray:
    root_pos, root_quat_wxyz = rollout._get_root_pose_world()
    root_lin_vel = np.asarray(
        rollout.data.qvel[rollout.root_joint_qvel_adr : rollout.root_joint_qvel_adr + 3],
        dtype=np.float32,
    )
    root_ang_vel_local = np.asarray(
        rollout.data.qvel[rollout.root_joint_qvel_adr + 3 : rollout.root_joint_qvel_adr + 6],
        dtype=np.float32,
    )
    root_ang_vel_world = quat_apply_wxyz(root_quat_wxyz, root_ang_vel_local).astype(np.float32, copy=False)
    root_quat_xyzw = np.asarray(
        [root_quat_wxyz[1], root_quat_wxyz[2], root_quat_wxyz[3], root_quat_wxyz[0]],
        dtype=np.float32,
    )
    return np.concatenate([root_pos, root_quat_xyzw, root_lin_vel, root_ang_vel_world]).astype(np.float32, copy=False)


def _sim_ref_state(rollout: OfflineWebRollout) -> np.ndarray:
    body_vel = np.zeros(6, dtype=np.float64)
    mj.mj_objectVelocity(rollout.model, rollout.data, mj.mjtObj.mjOBJ_BODY, rollout.ref_body_id, body_vel, 0)
    quat_wxyz = np.asarray(rollout.data.xquat[rollout.ref_body_id], dtype=np.float32)
    quat_xyzw = np.asarray([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)
    return np.concatenate(
        [
            np.asarray(rollout.data.xpos[rollout.ref_body_id], dtype=np.float32),
            quat_xyzw,
            np.asarray(body_vel[3:], dtype=np.float32),
            np.asarray(body_vel[:3], dtype=np.float32),
        ]
    ).astype(np.float32, copy=False)


def _sim_actor_state(rollout: OfflineWebRollout) -> np.ndarray:
    quat_wxyz = np.asarray(rollout.data.xquat[rollout.object_body_id], dtype=np.float32)
    quat_xyzw = np.asarray([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)
    lin_vel = np.zeros(3, dtype=np.float32)
    ang_vel_world = np.zeros(3, dtype=np.float32)
    if rollout.object_joint_qvel_adr is not None:
        lin_vel = np.asarray(
            rollout.data.qvel[rollout.object_joint_qvel_adr : rollout.object_joint_qvel_adr + 3],
            dtype=np.float32,
        )
        ang_vel_local = np.asarray(
            rollout.data.qvel[rollout.object_joint_qvel_adr + 3 : rollout.object_joint_qvel_adr + 6],
            dtype=np.float32,
        )
        ang_vel_world = quat_apply_wxyz(quat_wxyz, ang_vel_local).astype(np.float32, copy=False)
    return np.concatenate(
        [
            np.asarray(rollout.data.xpos[rollout.object_body_id], dtype=np.float32),
            quat_xyzw,
            lin_vel,
            ang_vel_world,
        ]
    ).astype(np.float32, copy=False)


def _robot_state_data(rollout: OfflineWebRollout) -> np.ndarray:
    root_pos, root_quat_wxyz = rollout._get_root_pose_world()
    root_lin_vel = np.asarray(
        rollout.data.qvel[rollout.root_joint_qvel_adr : rollout.root_joint_qvel_adr + 3],
        dtype=np.float32,
    )
    root_ang_vel_local = np.asarray(
        rollout.data.qvel[rollout.root_joint_qvel_adr + 3 : rollout.root_joint_qvel_adr + 6],
        dtype=np.float32,
    )
    root_ang_vel_world = quat_apply_wxyz(root_quat_wxyz, root_ang_vel_local).astype(np.float32, copy=False)

    dof_pos = np.zeros(len(rollout.joint_bindings), dtype=np.float32)
    dof_vel = np.zeros(len(rollout.joint_bindings), dtype=np.float32)
    for binding in rollout.joint_bindings:
        dof_pos[binding["index"]] = rollout.data.qpos[binding["qpos_adr"]]
        dof_vel[binding["index"]] = rollout.data.qvel[binding["qvel_adr"]]

    q = np.concatenate([root_pos.astype(np.float32), root_quat_wxyz.astype(np.float32), dof_pos], dtype=np.float32)
    dq = np.concatenate([root_lin_vel, root_ang_vel_world, dof_vel], dtype=np.float32)
    zeros = np.zeros_like(dq)
    return np.concatenate([q, dq, zeros, zeros], dtype=np.float32).reshape(1, -1)


def _sim_state_payload(rollout: OfflineWebRollout) -> dict:
    return {
        "robot_root_state": _sim_root_state(rollout),
        "robot_ref_state": _sim_ref_state(rollout),
        "robot_dof_pos": np.asarray(
            [rollout.data.qpos[binding["qpos_adr"]] for binding in rollout.joint_bindings],
            dtype=np.float32,
        ),
        "robot_dof_vel": np.asarray(
            [rollout.data.qvel[binding["qvel_adr"]] for binding in rollout.joint_bindings],
            dtype=np.float32,
        ),
        "actors": {"object": _sim_actor_state(rollout)},
    }


def _reference_policy_config(config: dict):
    base = INFERENCE_DEFAULTS["g1-29dof-w-obj"]
    return replace(
        base,
        task=replace(
            base.task,
            model_path=str((Path(__file__).resolve().parents[1] / "public" / "demo-assets" / config["model_path"]).resolve()),
            motion_file=str(Path(config["motion_file"]).resolve()),
            use_sim_state=True,
            use_sim_time=False,
            auto_start_motion=False,
            auto_start_motion_clip=False,
            use_zmq_lowcmd=True,
            use_root_reference_at_clip_start=True,
            prefer_sim_ref_from_sim_state=True,
        ),
    )


def _create_reference_policy(config: dict, initial_rollout: OfflineWebRollout) -> WholeBodyTrackingPolicy:
    policy = WholeBodyTrackingPolicy(_reference_policy_config(config))
    try:
        policy.interface.close()
    except Exception:
        pass
    dummy = DummyInterface()
    dummy.robot_config = policy.robot_config
    dummy.state = _robot_state_data(initial_rollout)
    policy.interface = dummy
    policy._latest_sim_state = _sim_state_payload(initial_rollout)
    policy._handle_stop_policy()
    policy._handle_start_policy()
    policy._handle_start_motion_clip()
    return policy


def _web_policy_step(rollout: OfflineWebRollout) -> tuple[np.ndarray, np.ndarray]:
    obs = rollout.build_obs()
    output_names = [item.name for item in rollout.session.get_outputs()]
    outputs = rollout.session.run(
        None,
        {
            "obs": obs.reshape(1, -1),
            "time_step": np.asarray([[rollout.motion_timestep]], dtype=np.float32),
        },
    )
    output_dict = {name: value for name, value in zip(output_names, outputs, strict=False)}

    raw_actions = np.asarray(output_dict["actions"], dtype=np.float32).reshape(-1)
    clip_threshold = float(rollout.config["control"]["clip_actions_threshold"])
    rollout.last_policy_action[:] = np.clip(raw_actions, -clip_threshold, clip_threshold)

    if "joint_pos" in output_dict and "joint_vel" in output_dict:
        rollout.motion_command_t[: len(rollout.joint_bindings)] = np.asarray(
            output_dict["joint_pos"], dtype=np.float32
        ).reshape(-1)
        rollout.motion_command_t[len(rollout.joint_bindings) :] = np.asarray(
            output_dict["joint_vel"], dtype=np.float32
        ).reshape(-1)
    if "ref_quat_xyzw" in output_dict:
        rollout.ref_quat_xyzw_t[:] = np.asarray(output_dict["ref_quat_xyzw"], dtype=np.float32).reshape(-1)
    if "ref_pos_xyz" in output_dict:
        rollout.ref_pos_xyz_t[:] = np.asarray(output_dict["ref_pos_xyz"], dtype=np.float32).reshape(-1)

    for binding in rollout.joint_bindings:
        idx = binding["index"]
        scale = float(rollout.config["control"]["policy_action_scales"][idx])
        q_current = float(rollout.data.qpos[binding["qpos_adr"]])
        dq_current = float(rollout.data.qvel[binding["qvel_adr"]])
        q_target = float(rollout.config["default_dof_angles"][idx]) + float(rollout.last_policy_action[idx]) * scale
        q_target = clamp(q_target, binding["range_min"], binding["range_max"])
        torque = float(rollout.config["kp"][idx]) * (q_target - q_current) + float(rollout.config["kd"][idx]) * (
            0.0 - dq_current
        )
        rollout.current_torques[idx] = clamp(torque, -binding["ctrl_limit"], binding["ctrl_limit"])

    for _ in range(int(rollout.config["control"]["control_decimation"])):
        for binding in rollout.joint_bindings:
            if binding["actuator_id"] >= 0:
                rollout.data.ctrl[binding["actuator_id"]] = rollout.current_torques[binding["index"]]
        mj.mj_step(rollout.model, rollout.data)
        rollout.current_step += 1
    rollout.motion_timestep += 1
    return obs, rollout.last_policy_action.copy()


def _reference_policy_step(
    rollout: OfflineWebRollout,
    policy: WholeBodyTrackingPolicy,
) -> tuple[np.ndarray, np.ndarray]:
    robot_state_data = _robot_state_data(rollout)
    policy.interface.state = robot_state_data
    policy._latest_sim_state = _sim_state_payload(rollout)

    obs = policy.prepare_obs_for_rl(robot_state_data)["actor_obs"].reshape(-1)
    scaled_action = policy.rl_inference(robot_state_data).reshape(-1)
    raw_action = policy.last_policy_action.reshape(-1).astype(np.float32, copy=True)

    q_target = policy.default_dof_angles.reshape(-1).astype(np.float32, copy=True) + scaled_action.astype(np.float32)
    if policy.q_min_arr is not None and policy.q_max_arr is not None:
        np.clip(q_target, policy.q_min_arr, policy.q_max_arr, out=q_target)

    for binding in rollout.joint_bindings:
        idx = binding["index"]
        q_current = float(rollout.data.qpos[binding["qpos_adr"]])
        dq_current = float(rollout.data.qvel[binding["qvel_adr"]])
        torque = float(rollout.config["kp"][idx]) * (float(q_target[idx]) - q_current) + float(
            rollout.config["kd"][idx]
        ) * (0.0 - dq_current)
        rollout.current_torques[idx] = clamp(torque, -binding["ctrl_limit"], binding["ctrl_limit"])

    for _ in range(int(rollout.config["control"]["control_decimation"])):
        for binding in rollout.joint_bindings:
            if binding["actuator_id"] >= 0:
                rollout.data.ctrl[binding["actuator_id"]] = rollout.current_torques[binding["index"]]
        mj.mj_step(rollout.model, rollout.data)
        rollout.current_step += 1
    return obs, raw_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "public" / "demo-assets",
    )
    parser.add_argument("--clip-id", type=str, default=None)
    parser.add_argument("--policy-steps", type=int, default=64)
    parser.add_argument("--max-obs-linf", type=float, default=5e-3)
    parser.add_argument("--max-action-linf", type=float, default=5e-4)
    parser.add_argument("--max-root-pos-linf", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asset_root = args.asset_root.expanduser().resolve()
    clip_id, config = _load_clip_config(asset_root, args.clip_id)

    web_rollout = OfflineWebRollout(asset_root, config)
    ref_rollout = OfflineWebRollout(asset_root, config)
    web_rollout.reset()
    ref_rollout.reset()
    policy = _create_reference_policy(config, ref_rollout)

    max_obs_linf = 0.0
    max_action_linf = 0.0
    max_root_pos_linf = 0.0

    steps = min(int(args.policy_steps), int(config["motion"]["frame_count"]))
    for _ in range(steps):
        web_obs, web_action = _web_policy_step(web_rollout)
        ref_obs, ref_action = _reference_policy_step(ref_rollout, policy)

        max_obs_linf = max(max_obs_linf, float(np.max(np.abs(web_obs - ref_obs))))
        max_action_linf = max(max_action_linf, float(np.max(np.abs(web_action - ref_action))))

        web_root_pos, _ = web_rollout._get_root_pose_world()
        ref_root_pos, _ = ref_rollout._get_root_pose_world()
        max_root_pos_linf = max(max_root_pos_linf, float(np.max(np.abs(web_root_pos - ref_root_pos))))

    summary = {
        "clip_id": clip_id,
        "policy_steps": steps,
        "max_obs_linf": max_obs_linf,
        "max_action_linf": max_action_linf,
        "max_root_pos_linf": max_root_pos_linf,
        "web_policy_action_scale_unique": sorted(
            {round(float(value), 6) for value in config["control"]["policy_action_scales"]}
        ),
    }
    print(json.dumps(summary, indent=2))

    if max_obs_linf > float(args.max_obs_linf):
        raise SystemExit(
            f"Web rollout obs divergence {max_obs_linf:.6g} exceeded threshold {args.max_obs_linf:.6g}."
        )
    if max_action_linf > float(args.max_action_linf):
        raise SystemExit(
            f"Web rollout action divergence {max_action_linf:.6g} exceeded threshold {args.max_action_linf:.6g}."
        )
    if max_root_pos_linf > float(args.max_root_pos_linf):
        raise SystemExit(
            f"Web rollout root-pos divergence {max_root_pos_linf:.6g} exceeded threshold {args.max_root_pos_linf:.6g}."
        )


if __name__ == "__main__":
    main()
