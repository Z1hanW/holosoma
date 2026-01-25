from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from holosoma.utils.module_utils import get_holosoma_root
from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.safe_torch_import import torch


def _ensure_viser_on_path() -> None:
    root = Path(get_holosoma_root())
    repo_root = root.parents[2] if len(root.parents) > 2 else root
    viser_src = repo_root / "viser" / "src"
    if viser_src.exists() and str(viser_src) not in sys.path:
        sys.path.insert(0, str(viser_src))


def _import_viser() -> tuple[Any | None, Any | None, str | None]:
    _ensure_viser_on_path()
    try:
        import viser  # type: ignore[import-not-found]
        from viser.extras import ViserUrdf  # type: ignore[import-not-found]
    except Exception as exc:
        return None, None, str(exc)
    return viser, ViserUrdf, None


def _resolve_data_path(path: str) -> str:
    if path.startswith("@holosoma/"):
        return str(Path(get_holosoma_root()) / path[len("@holosoma/") :])
    return resolve_data_file_path(path)


def _resolve_robot_urdf_path(robot_config: Any) -> str:
    asset_root = _resolve_data_path(robot_config.asset.asset_root)
    urdf_path = os.path.join(asset_root, robot_config.asset.urdf_file)
    return _resolve_data_path(urdf_path)


def _resolve_object_urdf_path(robot_config: Any) -> str | None:
    obj_path = getattr(getattr(robot_config, "object", None), "object_urdf_path", None)
    if not obj_path:
        return None
    return _resolve_data_path(obj_path)


def _is_rank0() -> bool:
    try:
        return int(os.environ.get("RANK", "0")) == 0
    except ValueError:
        return True


def _load_terrain_mesh(
    obj_path: str | None,
    *,
    obj_metadata_path: str | None,
    num_rows: int | None,
    num_cols: int | None,
):
    if not obj_path:
        return None

    try:
        import trimesh  # type: ignore[import-not-found]
    except Exception as exc:
        logger.warning("Viser terrain disabled (trimesh unavailable): {}", exc)
        return None

    terrain_path = Path(_resolve_data_path(obj_path))
    if terrain_path.is_dir():
        candidates = sorted(list(terrain_path.glob("*.obj")) + list(terrain_path.glob("*.OBJ")))
        if not candidates:
            return None
        terrain_path = candidates[0]

    if not terrain_path.exists():
        return None

    base_mesh = trimesh.load(str(terrain_path), process=False)
    if isinstance(base_mesh, trimesh.Scene):
        base_mesh = base_mesh.dump(concatenate=True)
    if not isinstance(base_mesh, trimesh.Trimesh):
        raise ValueError(f"Loaded terrain is not a trimesh: {type(base_mesh)}")

    if obj_metadata_path:
        return base_mesh

    rows = int(num_rows or 1)
    cols = int(num_cols or 1)
    if rows * cols <= 1:
        return base_mesh

    gap = 1e-4
    stride = (base_mesh.bounds[1] - base_mesh.bounds[0]) + gap
    tiles = []
    for r in range(rows):
        for c in range(cols):
            tile = base_mesh.copy()
            tile.apply_translation([c * stride[0], r * stride[1], 0.0])
            tiles.append(tile)
    return trimesh.util.concatenate(tiles)


class ViserLiveViewer:
    def __init__(self, env: Any) -> None:
        self._env = env
        self._enabled = bool(getattr(env.training_config, "enable_viser", False))
        self._server = None
        self._vr = None
        self._vo = None
        self._robot_root = None
        self._object_root = None
        self._joint_order: np.ndarray | None = None
        self._joint_count = 0
        self._offset: np.ndarray | None = None
        self._last_update = 0.0

        if not self._enabled:
            return
        if not _is_rank0():
            self._enabled = False
            return

        cfg = env.training_config
        self._env_id = int(getattr(cfg, "viser_env_id", 0))
        if self._env_id < 0 or self._env_id >= getattr(env, "num_envs", 1):
            logger.warning("Viser env_id {} out of range; defaulting to 0.", self._env_id)
            self._env_id = 0

        update_hz = float(getattr(cfg, "viser_update_hz", 30.0))
        self._update_period = 0.0 if update_hz <= 0 else 1.0 / update_hz
        self._recenter = bool(getattr(cfg, "viser_recenter", True))

        viser_mod, viser_urdf_cls, err = _import_viser()
        if err is not None or viser_mod is None or viser_urdf_cls is None:
            logger.warning("Viser live viewer disabled: {}", err or "missing dependency")
            self._enabled = False
            return

        port_cfg = int(getattr(cfg, "viser_port", 6060))
        port = int(os.environ.get("HOLOSOMA_VISER_PORT", str(port_cfg)))
        self._server = viser_mod.ViserServer(port=port)

        self._robot_root = self._server.scene.add_frame("/robot", show_axes=False)
        self._object_root = self._server.scene.add_frame("/object", show_axes=False)
        self._server.scene.add_grid("/grid", width=8.0, height=8.0, position=(0.0, 0.0, 0.0))

        robot_urdf = _resolve_robot_urdf_path(env.robot_config)
        self._vr = viser_urdf_cls(self._server, urdf_or_path=Path(robot_urdf), root_node_name="/robot")

        object_urdf = _resolve_object_urdf_path(env.robot_config)
        if object_urdf:
            self._vo = viser_urdf_cls(self._server, urdf_or_path=Path(object_urdf), root_node_name="/object")

        self._setup_joint_order()
        self._load_terrain()

        logger.info("Viser live viewer listening on port {}", port)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def on_reset(self, env_ids) -> None:
        if not self._enabled or not self._recenter:
            return
        if self._env_id not in _normalize_env_ids(env_ids):
            return
        self._offset = self._resolve_env_origin()

    def record_step(self) -> None:
        if not self._enabled or self._vr is None or self._robot_root is None:
            return

        now = time.perf_counter()
        if self._update_period > 0 and (now - self._last_update) < self._update_period:
            return
        self._last_update = now

        root_pos, root_quat_wxyz = self._get_root_state_wxyz()
        if root_pos is None or root_quat_wxyz is None:
            return

        dof_pos = self._get_dof_pos()
        if dof_pos is None:
            return

        if self._offset is None:
            self._offset = self._resolve_env_origin() if self._recenter else np.zeros(3, dtype=np.float32)
            if self._offset is None:
                self._offset = root_pos.copy()
        offset = self._offset if self._recenter else np.zeros(3, dtype=np.float32)

        self._robot_root.position = root_pos - offset
        self._robot_root.wxyz = root_quat_wxyz

        joints = dof_pos
        if self._joint_order is not None:
            joints = joints[self._joint_order]
        if joints.shape[0] != self._joint_count:
            return
        self._vr.update_cfg(joints.astype(np.float32, copy=False))

        if self._vo is None or self._object_root is None:
            return
        obj_state = self._get_object_state_wxyz()
        if obj_state is None:
            self._vo.show_visual = False
            return
        obj_pos, obj_quat_wxyz = obj_state
        self._object_root.position = obj_pos - offset
        self._object_root.wxyz = obj_quat_wxyz
        self._vo.show_visual = True

    def _setup_joint_order(self) -> None:
        if self._vr is None:
            return
        viser_joint_names = list(self._vr.get_actuated_joint_names())
        self._joint_count = len(viser_joint_names)
        name_to_idx = {name: idx for idx, name in enumerate(self._env.robot_config.dof_names)}
        missing = [name for name in viser_joint_names if name not in name_to_idx]
        if missing:
            if len(viser_joint_names) == len(self._env.robot_config.dof_names):
                logger.warning("Viser joint names mismatch; using robot DOF order.")
                self._joint_order = None
            else:
                logger.warning("Viser joints missing in robot config: {}", missing)
                self._enabled = False
        else:
            self._joint_order = np.array([name_to_idx[name] for name in viser_joint_names], dtype=np.int64)

    def _load_terrain(self) -> None:
        if self._server is None:
            return
        terrain_mgr = getattr(self._env, "terrain_manager", None)
        if terrain_mgr is None:
            return
        terrain_cfg = getattr(terrain_mgr, "cfg", None)
        terrain_term = getattr(terrain_cfg, "terrain_term", None) if terrain_cfg is not None else None
        if terrain_term is None:
            return
        obj_path = getattr(terrain_term, "obj_file_path", None) or ""
        if not obj_path:
            return
        obj_meta = getattr(terrain_term, "obj_metadata_path", None)
        rows = getattr(terrain_term, "num_rows", None)
        cols = getattr(terrain_term, "num_cols", None)

        mesh = _load_terrain_mesh(obj_path, obj_metadata_path=obj_meta, num_rows=rows, num_cols=cols)
        if mesh is None:
            return
        self._server.scene.add_mesh_trimesh("/terrain", mesh)

    def _resolve_env_origin(self) -> np.ndarray | None:
        origins = None
        terrain_mgr = getattr(self._env, "terrain_manager", None)
        if terrain_mgr is not None and hasattr(terrain_mgr, "get_state"):
            terrain_state = terrain_mgr.get_state("locomotion_terrain")
            if terrain_state is not None and hasattr(terrain_state, "env_origins"):
                origins = terrain_state.env_origins
        if origins is None:
            scene = getattr(self._env.simulator, "scene", None)
            if scene is not None and hasattr(scene, "env_origins"):
                origins = scene.env_origins
        if origins is None:
            return None
        try:
            origin = origins[self._env_id]
        except Exception:
            return None
        if isinstance(origin, torch.Tensor):
            return origin.detach().cpu().numpy()
        return np.asarray(origin, dtype=np.float32)

    def _get_root_state_wxyz(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        root_states = getattr(self._env.simulator, "robot_root_states", None)
        if root_states is None:
            return None, None

        if hasattr(root_states, "tensor_wxyz"):
            root = root_states.tensor_wxyz[self._env_id]
            pos = root[0:3]
            quat_wxyz = root[3:7]
        else:
            root = root_states[self._env_id]
            pos = root[0:3]
            quat_xyzw = root[3:7]
            quat_wxyz = quat_xyzw[[3, 0, 1, 2]]

        return pos.detach().cpu().numpy(), quat_wxyz.detach().cpu().numpy()

    def _get_dof_pos(self) -> np.ndarray | None:
        dof_pos = getattr(self._env.simulator, "dof_pos", None)
        if dof_pos is None:
            return None
        return dof_pos[self._env_id].detach().cpu().numpy()

    def _get_object_state_wxyz(self) -> tuple[np.ndarray, np.ndarray] | None:
        if not getattr(self._env.robot_config, "object", None):
            return None
        if not getattr(self._env.robot_config.object, "object_urdf_path", None):
            return None

        env_ids = torch.tensor([self._env_id], device=self._env.device, dtype=torch.long)
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


def _normalize_env_ids(env_ids) -> list[int]:
    if isinstance(env_ids, torch.Tensor):
        return [int(idx.item()) for idx in env_ids.flatten()]
    if isinstance(env_ids, np.ndarray):
        return [int(idx) for idx in env_ids.flatten().tolist()]
    return [int(idx) for idx in env_ids]
