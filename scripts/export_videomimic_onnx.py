#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch
from torch import nn


def _add_repo_paths(repo_root: Path) -> None:
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "VideoMimic" / "simulation" / "videomimic_gym"))
    sys.path.insert(0, str(repo_root / "holosoma"))


def _find_input_dim(module: nn.Module) -> int:
    for submodule in module.modules():
        if isinstance(submodule, nn.Linear):
            return submodule.in_features
    raise ValueError("Unable to infer actor input dim from module.")


def _load_videomimic_actor(task: str, checkpoint_path: str, device: str) -> nn.Module:
    from legged_gym.utils import task_registry
    from legged_gym.utils.helpers import get_args

    args, _ = get_args()
    args.task = task
    args.headless = True
    args.num_envs = 1
    args.rl_device = device
    args.sim_device = device

    env_cfg, train_cfg = task_registry.get_cfgs(name=task)
    env, _ = task_registry.make_env(name=task, args=args, env_cfg=env_cfg)
    runner, _ = task_registry.make_alg_runner(env=env, name=task, args=args, train_cfg=train_cfg)
    runner.load(checkpoint_path)

    actor_critic = runner.alg.actor_critic
    actor_critic.eval()
    actor = getattr(actor_critic, "actor", actor_critic)
    actor.eval()
    return actor


class VideoMimicAdapter(nn.Module):
    def __init__(
        self,
        actor: nn.Module,
        dof_names: list[str],
        history_len: int,
        actor_input_dim: int,
        heightmap_dim: int | None,
    ):
        super().__init__()
        self.actor = actor
        self.dof_names = list(dof_names)
        self.history_len = int(history_len)
        self.actor_input_dim = int(actor_input_dim)

        drop_names = {
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        }
        self.keep_dof_indices = [i for i, name in enumerate(self.dof_names) if name not in drop_names]
        if len(self.keep_dof_indices) != len(self.dof_names) - 6:
            raise ValueError("Unexpected DOF layout; wrist joints not found where expected.")

        dof_count = len(self.dof_names)
        torso_real_dim = 6 + dof_count * 3
        self.torso_real_dim = torso_real_dim
        self.holosoma_actor_obs_dim = (
            self.history_len * (torso_real_dim + 2 + 1) + dof_count + 2
        )

        base_dim = self._reduced_obs_dim()
        if heightmap_dim is None:
            heightmap_dim = max(0, self.actor_input_dim - base_dim)
        self.heightmap_dim = int(heightmap_dim)
        if base_dim + self.heightmap_dim != self.actor_input_dim:
            raise ValueError(
                f"Actor expects {self.actor_input_dim} dims but reduced obs {base_dim} + "
                f"heightmap {self.heightmap_dim} does not match."
            )

        self._torso_keep = self._build_torso_keep_indices(dof_count)
        self._target_keep = torch.tensor(self.keep_dof_indices, dtype=torch.long)

    def _build_torso_keep_indices(self, dof_count: int) -> torch.Tensor:
        keep = list(range(6))
        for offset in (6, 6 + dof_count, 6 + 2 * dof_count):
            keep.extend([offset + idx for idx in self.keep_dof_indices])
        return torch.tensor(keep, dtype=torch.long)

    def _reduced_obs_dim(self) -> int:
        dof_keep = len(self.keep_dof_indices)
        torso_real_reduced = 6 + dof_keep * 3
        torso_hist = self.history_len * torso_real_reduced
        rel_hist = self.history_len * (2 + 1)
        return torso_hist + rel_hist + dof_keep + 2

    def _reduce_obs(self, actor_obs: torch.Tensor) -> torch.Tensor:
        batch = actor_obs.shape[0]
        if actor_obs.shape[1] != self.holosoma_actor_obs_dim:
            raise ValueError(
                f"Expected actor_obs dim {self.holosoma_actor_obs_dim}, got {actor_obs.shape[1]}."
            )

        offset = 0
        torso_len = self.history_len * self.torso_real_dim
        torso_real = actor_obs[:, offset : offset + torso_len].reshape(
            batch, self.history_len, self.torso_real_dim
        )
        offset += torso_len

        torso_xy = actor_obs[:, offset : offset + self.history_len * 2].reshape(batch, self.history_len, 2)
        offset += self.history_len * 2

        torso_yaw = actor_obs[:, offset : offset + self.history_len].reshape(batch, self.history_len, 1)
        offset += self.history_len

        target_joints = actor_obs[:, offset : offset + len(self.dof_names)]
        offset += len(self.dof_names)

        target_root_pitch = actor_obs[:, offset : offset + 1]
        offset += 1

        target_root_roll = actor_obs[:, offset : offset + 1]

        torso_real_reduced = torch.index_select(torso_real, dim=2, index=self._torso_keep)
        torso_real_flat = torso_real_reduced.reshape(batch, -1)
        torso_xy_flat = torso_xy.reshape(batch, -1)
        torso_yaw_flat = torso_yaw.reshape(batch, -1)
        target_joints_reduced = torch.index_select(target_joints, dim=1, index=self._target_keep)

        base = torch.cat(
            [torso_real_flat, torso_xy_flat, torso_yaw_flat, target_joints_reduced, target_root_pitch, target_root_roll],
            dim=1,
        )
        if self.heightmap_dim > 0:
            zeros = torch.zeros((batch, self.heightmap_dim), device=base.device, dtype=base.dtype)
            base = torch.cat([base, zeros], dim=1)
        return base

    def _call_actor(self, obs: torch.Tensor) -> torch.Tensor:
        out = self.actor(obs)
        if isinstance(out, tuple):
            return out[0]
        return out

    def forward(self, actor_obs: torch.Tensor) -> torch.Tensor:
        obs = self._reduce_obs(actor_obs)
        action_23 = self._call_actor(obs)
        batch = action_23.shape[0]
        action_29 = torch.zeros((batch, len(self.dof_names)), device=action_23.device, dtype=action_23.dtype)
        action_29[:, self.keep_dof_indices] = action_23
        return action_29


def main() -> None:
    parser = argparse.ArgumentParser(description="Export VideoMimic checkpoint to holosoma-compatible ONNX.")
    parser.add_argument("--checkpoint", required=True, help="Path to VideoMimic checkpoint (.pt).")
    parser.add_argument("--task", default="g1_deepmimic_proj_heightfield", help="VideoMimic task name.")
    parser.add_argument("--motion-file", required=True, help="Holosoma motion npz used for VideoMimic obs.")
    parser.add_argument("--output", required=True, help="Output ONNX path.")
    parser.add_argument("--device", default="cuda:0", help="Torch device for export.")
    parser.add_argument("--history-len", type=int, default=5, help="History length for torso_* obs.")
    parser.add_argument("--heightmap-dim", type=int, default=None, help="Heightmap dim to append (auto if None).")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    _add_repo_paths(repo_root)

    from holosoma.config_values import robot as robot_cfg
    from holosoma.utils.inference_helpers import (
        attach_onnx_metadata,
        get_control_gains_from_config,
        get_urdf_text_from_robot_config,
    )

    actor = _load_videomimic_actor(args.task, args.checkpoint, args.device)
    actor_input_dim = _find_input_dim(actor)

    dof_names = list(robot_cfg.g1_29dof.dof_names)
    adapter = VideoMimicAdapter(
        actor=actor,
        dof_names=dof_names,
        history_len=args.history_len,
        actor_input_dim=actor_input_dim,
        heightmap_dim=args.heightmap_dim,
    )
    adapter.eval()

    example_obs_dim = adapter.holosoma_actor_obs_dim
    example_obs = torch.zeros((1, example_obs_dim), dtype=torch.float32, device="cpu")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        adapter,
        example_obs,
        str(output_path),
        input_names=["actor_obs"],
        output_names=["action"],
        opset_version=13,
        dynamic_axes={"actor_obs": {0: "batch"}, "action": {0: "batch"}},
        verbose=False,
        dynamo=False,
    )

    urdf_text = get_urdf_text_from_robot_config(robot_cfg.g1_29dof)
    kp, kd = get_control_gains_from_config(robot_cfg.g1_29dof)
    motion_cfg = {
        "motion_file": str(Path(args.motion_file).expanduser()),
        "body_name_ref": ["torso_link"],
        "body_names_to_track": ["torso_link"],
        "align_motion_to_init_yaw": True,
    }
    metadata = {
        "robot_urdf": urdf_text,
        "kp": kp,
        "kd": kd,
        "dof_names": dof_names,
        "motion_config": motion_cfg,
    }
    attach_onnx_metadata(str(output_path), metadata)


if __name__ == "__main__":
    main()
