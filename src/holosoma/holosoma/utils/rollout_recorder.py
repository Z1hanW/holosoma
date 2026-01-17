from __future__ import annotations

import atexit
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from holosoma.utils.safe_torch_import import torch


class RolloutRecorder:
    """Record physics rollouts to disk for later Viser playback.

    Enable with: HOLOSOMA_ROLLOUT_DIR=/abs/path
    Optional:
      - HOLOSOMA_ROLLOUT_RECORD_ENV_ID (default: simulator video record env)
      - HOLOSOMA_ROLLOUT_PREFIX (default: "rollout")
      - HOLOSOMA_ROLLOUT_MAX_FRAMES (default: 0 -> no limit)
      - HOLOSOMA_ROLLOUT_SAVE_ON_RESET (default: 1)
    """

    def __init__(self, env: Any) -> None:
        self._env = env
        self._enabled = bool(os.environ.get("HOLOSOMA_ROLLOUT_DIR"))
        if not self._enabled:
            return

        self._save_dir = Path(os.environ["HOLOSOMA_ROLLOUT_DIR"]).expanduser()
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._prefix = os.environ.get("HOLOSOMA_ROLLOUT_PREFIX", "rollout")
        self._record_env_id = int(
            os.environ.get(
                "HOLOSOMA_ROLLOUT_RECORD_ENV_ID",
                getattr(getattr(env.simulator, "video_config", None), "record_env_id", 0),
            )
        )
        self._max_frames = int(os.environ.get("HOLOSOMA_ROLLOUT_MAX_FRAMES", "0"))
        self._save_on_reset = os.environ.get("HOLOSOMA_ROLLOUT_SAVE_ON_RESET", "1") not in ("0", "false", "False")

        self._episode_idx = 0
        self._frames: list[np.ndarray] = []
        self._stopped = False

        self._fps = float(1.0 / getattr(env, "dt", 0.02))
        self._terrain_obj_path = self._resolve_terrain_path()
        self._terrain_rows, self._terrain_cols = self._resolve_terrain_grid()
        self._env_origin = self._resolve_env_origin()

        atexit.register(self._finalize, "exit")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def record_step(self) -> None:
        if not self._enabled or self._stopped:
            return

        frame = self._build_frame()
        if frame is None:
            return

        self._frames.append(frame)
        if self._max_frames > 0 and len(self._frames) >= self._max_frames:
            self._finalize("max_frames")
            self._stopped = True

    def on_reset(self, env_ids: torch.Tensor | np.ndarray | list[int]) -> None:
        if not self._enabled or not self._save_on_reset or self._stopped:
            return

        env_id_list = self._normalize_env_ids(env_ids)
        if self._record_env_id not in env_id_list:
            return
        self._finalize("reset")

    def _finalize(self, reason: str) -> None:
        if not self._enabled or not self._frames:
            return

        clip_name = self._get_clip_name() or "rollout"
        safe_clip = re.sub(r"[^A-Za-z0-9_.-]+", "_", clip_name).strip("_") or "rollout"
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{self._prefix}_{self._episode_idx:04d}_{safe_clip}_{timestamp}.npz"
        path = self._save_dir / filename

        qpos = np.stack(self._frames, axis=0).astype(np.float32, copy=False)
        payload: dict[str, Any] = {
            "qpos": qpos,
            "fps": np.array([self._fps], dtype=np.float32),
            "clip_name": np.array([clip_name]),
            "record_env_id": np.array([self._record_env_id], dtype=np.int64),
            "reason": np.array([reason]),
        }

        if self._env_origin is not None:
            payload["env_origin"] = self._env_origin.astype(np.float32, copy=False)
        if self._terrain_obj_path:
            payload["terrain_obj_path"] = np.array([self._terrain_obj_path])
        if self._terrain_rows is not None and self._terrain_cols is not None:
            payload["terrain_num_rows"] = np.array([self._terrain_rows], dtype=np.int64)
            payload["terrain_num_cols"] = np.array([self._terrain_cols], dtype=np.int64)

        np.savez(path, **payload)
        self._frames.clear()
        self._episode_idx += 1

    def _build_frame(self) -> np.ndarray | None:
        root_pos, root_quat_wxyz = self._get_root_state_wxyz()
        if root_pos is None or root_quat_wxyz is None:
            return None

        dof_pos = self._get_dof_pos()
        if dof_pos is None:
            return None

        parts = [root_pos, root_quat_wxyz, dof_pos]
        obj_state = self._get_object_state_wxyz()
        if obj_state is not None:
            obj_pos, obj_quat = obj_state
            parts.extend([obj_pos, obj_quat])

        return np.concatenate(parts, axis=0).astype(np.float32, copy=False)

    def _get_root_state_wxyz(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        root_states = getattr(self._env.simulator, "robot_root_states", None)
        if root_states is None:
            return None, None

        if hasattr(root_states, "tensor_wxyz"):
            root = root_states.tensor_wxyz[self._record_env_id]
            pos = root[0:3]
            quat_wxyz = root[3:7]
        else:
            root = root_states[self._record_env_id]
            pos = root[0:3]
            quat_xyzw = root[3:7]
            quat_wxyz = quat_xyzw[[3, 0, 1, 2]]

        return (
            pos.detach().cpu().numpy(),
            quat_wxyz.detach().cpu().numpy(),
        )

    def _get_dof_pos(self) -> np.ndarray | None:
        dof_pos = getattr(self._env.simulator, "dof_pos", None)
        if dof_pos is None:
            return None
        return dof_pos[self._record_env_id].detach().cpu().numpy()

    def _get_object_state_wxyz(self) -> tuple[np.ndarray, np.ndarray] | None:
        if not getattr(self._env.robot_config, "object", None):
            return None
        if not getattr(self._env.robot_config.object, "object_urdf_path", None):
            return None

        env_ids = torch.tensor([self._record_env_id], device=self._env.device, dtype=torch.long)
        sim = self._env.simulator
        states = None
        if hasattr(sim, "_get_object_states"):
            try:
                states = sim._get_object_states("object", env_ids)
            except Exception:
                states = None
        if states is None and hasattr(sim, "get_actor_states") and getattr(sim, "has_scene_objects", False):
            try:
                states = sim.get_actor_states(["object"], env_ids)
            except Exception:
                states = None
        if states is None or states.numel() == 0:
            return None

        state = states[0]
        pos = state[0:3]
        quat_xyzw = state[3:7]
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
        return pos.detach().cpu().numpy(), quat_wxyz.detach().cpu().numpy()

    def _get_clip_name(self) -> str | None:
        command_mgr = getattr(self._env, "command_manager", None)
        if command_mgr is None:
            return None
        motion_cmd = command_mgr.get_state("motion_command")
        if motion_cmd is None or not hasattr(motion_cmd, "clip_ids"):
            return None

        clip_ids = motion_cmd.clip_ids
        if clip_ids is None:
            return None
        try:
            clip_idx = int(clip_ids[self._record_env_id].item())
        except Exception:
            return None
        clip_names = getattr(motion_cmd.motion, "clip_ids", None)
        if not clip_names or clip_idx < 0 or clip_idx >= len(clip_names):
            return None
        return str(clip_names[clip_idx])

    def _resolve_terrain_path(self) -> str | None:
        terrain_mgr = getattr(self._env, "terrain_manager", None)
        if terrain_mgr is None:
            return None
        terrain_cfg = getattr(terrain_mgr, "cfg", None)
        if terrain_cfg is None:
            return None
        terrain_term = getattr(terrain_cfg, "terrain_term", None)
        if terrain_term is None:
            return None
        obj_path = getattr(terrain_term, "obj_file_path", None)
        if not obj_path:
            return None
        return str(obj_path)

    def _resolve_terrain_grid(self) -> tuple[int | None, int | None]:
        terrain_mgr = getattr(self._env, "terrain_manager", None)
        if terrain_mgr is None:
            return None, None
        terrain_cfg = getattr(terrain_mgr, "cfg", None)
        if terrain_cfg is None:
            return None, None
        terrain_term = getattr(terrain_cfg, "terrain_term", None)
        if terrain_term is None:
            return None, None
        rows = getattr(terrain_term, "num_rows", None)
        cols = getattr(terrain_term, "num_cols", None)
        if rows is None or cols is None:
            return None, None
        return int(rows), int(cols)

    def _resolve_env_origin(self) -> np.ndarray | None:
        scene = getattr(self._env.simulator, "scene", None)
        if scene is None or not hasattr(scene, "env_origins"):
            return None
        try:
            origin = scene.env_origins[self._record_env_id]
        except Exception:
            return None
        if isinstance(origin, torch.Tensor):
            return origin.detach().cpu().numpy()
        return np.asarray(origin)

    @staticmethod
    def _normalize_env_ids(env_ids: torch.Tensor | np.ndarray | list[int]) -> list[int]:
        if isinstance(env_ids, torch.Tensor):
            return [int(idx.item()) for idx in env_ids.flatten()]
        if isinstance(env_ids, np.ndarray):
            return [int(idx) for idx in env_ids.flatten().tolist()]
        return [int(idx) for idx in env_ids]
