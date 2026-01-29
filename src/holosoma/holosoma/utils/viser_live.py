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
from holosoma.utils.viser_utils import ensure_viser_on_path, resolve_viser_port

LIGHT_BLUE = (130, 180, 235)


def _import_viser() -> tuple[Any | None, Any | None, str | None]:
    ensure_viser_on_path()
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

    def _load_mesh(path: Path) -> trimesh.Trimesh:
        mesh = trimesh.load(str(path), process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Loaded terrain is not a trimesh: {type(mesh)}")
        return mesh

    def _resolve_obj_paths(path_str: str) -> list[Path]:
        path = Path(path_str)
        if path.is_dir():
            matches = list(path.glob("*.obj")) + list(path.glob("*.OBJ"))
            return sorted(matches)
        if any(char in path_str for char in ("*", "?", "[")):
            import glob

            return sorted(Path(p) for p in glob.glob(path_str))
        return [path] if path.exists() else []

    def _tile_single_mesh(mesh: trimesh.Trimesh, rows: int, cols: int) -> trimesh.Trimesh:
        if rows * cols <= 1:
            return mesh
        gap = 1e-4
        stride = (mesh.bounds[1] - mesh.bounds[0]) + gap
        tiles = []
        for r in range(rows):
            for c in range(cols):
                tile = mesh.copy()
                tile.apply_translation([c * stride[0], r * stride[1], 0.0])
                tiles.append(tile)
        return trimesh.util.concatenate(tiles)

    terrain_path = Path(_resolve_data_path(obj_path))
    obj_paths = _resolve_obj_paths(str(terrain_path))
    if not obj_paths:
        return None

    if obj_metadata_path:
        if len(obj_paths) != 1:
            logger.warning("OBJ metadata requires a single OBJ file; ignoring metadata for directory input.")
        base_mesh = _load_mesh(obj_paths[0])
        return base_mesh

    rows = int(num_rows or 1)
    cols = int(num_cols or 1)

    if len(obj_paths) == 1:
        base_mesh = _load_mesh(obj_paths[0])
        return _tile_single_mesh(base_mesh, rows, cols)

    meshes = []
    spans = []
    for path in obj_paths:
        mesh = _load_mesh(path)
        meshes.append(mesh)
        spans.append(mesh.bounds[1] - mesh.bounds[0])

    spans = np.vstack(spans)
    gap = 1e-4
    stride = spans.max(axis=0) + gap

    tiles = []
    for col, mesh in enumerate(meshes):
        col_offset = np.array([col * stride[0], 0.0, 0.0], dtype=np.float64)
        for row in range(rows):
            offset = col_offset + np.array([0.0, row * stride[1], 0.0], dtype=np.float64)
            tile = mesh.copy()
            tile.apply_translation(offset)
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
        self._scandots_handle = None
        self._scandots_enabled = False
        self._scandots_point_size = 0.02
        self._scandots_color = np.array([0, 255, 255], dtype=np.uint8)
        self._scandots_warned = False
        self._play_control = None
        self._step_button = None
        self._reset_button = None
        self._step_requested = False
        self._reset_requested = False
        self._clip_dropdown = None
        self._clip_apply = None
        self._clip_label = None
        self._clip_names: list[str] = []
        self._pending_clip_idx: int | None = None
        self._control_sleep_s = 0.02
        self._pending_clip_start: int | None = None
        self._grid_handle = None
        self._terrain_handle = None
        self._ground_handle = None
        self._show_robot_cb = None
        self._show_object_cb = None
        self._show_terrain_cb = None
        self._show_grid_cb = None
        self._show_scandots_cb = None
        self._recenter_cb = None
        self._scandots_size_slider = None
        self._contact_force_cb = None
        self._contact_force_scale_slider = None
        self._contact_force_threshold_slider = None
        self._clip_start_slider = None
        self._clip_lock_cb = None

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
        sync_to_sim = bool(getattr(cfg, "viser_sync_to_sim", True))
        force_dt = bool(getattr(cfg, "viser_force_dt", True))
        sim_hz = None
        sim_cfg = getattr(env.simulator, "simulator_config", None)
        if sim_cfg is not None and hasattr(sim_cfg, "sim"):
            sim_fps = getattr(sim_cfg.sim, "fps", None)
            sim_decimation = getattr(sim_cfg.sim, "control_decimation", None)
            if sim_fps:
                sim_decimation = int(sim_decimation or 1)
                sim_hz = float(sim_fps) / max(1, sim_decimation)

        if sync_to_sim and sim_hz:
            update_hz = sim_hz
        elif update_hz <= 0 and sim_hz:
            update_hz = sim_hz

        self._update_period = 0.0 if update_hz <= 0 else 1.0 / update_hz
        self._force_dt = force_dt and self._update_period > 0
        self._next_tick = time.perf_counter()
        self._recenter = bool(getattr(cfg, "viser_recenter", True))
        self._scandots_enabled = bool(getattr(cfg, "viser_show_scandots", False))
        self._scandots_point_size = float(getattr(cfg, "viser_scandots_point_size", 0.02))

        viser_mod, viser_urdf_cls, err = _import_viser()
        if err is not None or viser_mod is None or viser_urdf_cls is None:
            logger.warning("Viser live viewer disabled: {}", err or "missing dependency")
            self._enabled = False
            return

        port_cfg = int(getattr(cfg, "viser_port", 0) or 0)
        port = resolve_viser_port(port_cfg)
        self._server = viser_mod.ViserServer(port=port)

        self._robot_root = self._server.scene.add_frame("/robot", show_axes=False)
        self._object_root = self._server.scene.add_frame("/object", show_axes=False)
        self._grid_handle = self._server.scene.add_grid(
            "/grid", width=8.0, height=8.0, position=(0.0, 0.0, 0.0)
        )
        self._grid_handle.visible = False

        robot_urdf = _resolve_robot_urdf_path(env.robot_config)
        self._vr = viser_urdf_cls(
            self._server,
            urdf_or_path=Path(robot_urdf),
            root_node_name="/robot",
            mesh_color_override=LIGHT_BLUE,
        )

        object_urdf = _resolve_object_urdf_path(env.robot_config)
        if object_urdf:
            self._vo = viser_urdf_cls(
                self._server,
                urdf_or_path=Path(object_urdf),
                root_node_name="/object",
                mesh_color_override=LIGHT_BLUE,
            )

        self._setup_joint_order()
        self._load_terrain()
        self._setup_controls()

        logger.info("Viser live viewer listening on port {}", port)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _set_handle_visible(self, handle, visible: bool) -> None:
        if handle is None:
            return
        try:
            handle.visible = bool(visible)
        except Exception:
            return

    def _set_sim_cfg(self, field: str, value) -> None:
        sim_cfg = getattr(self._env.simulator, "simulator_config", None)
        if sim_cfg is None:
            return
        try:
            object.__setattr__(sim_cfg, field, value)
        except Exception:
            return

    def wait_if_paused(self) -> None:
        if not self._enabled or self._play_control is None:
            return
        while not bool(self._play_control.value):
            if self._step_requested:
                self._step_requested = False
                return
            if self._reset_requested or self._pending_clip_idx is not None or self._pending_clip_start is not None:
                self.apply_pending_controls()
            time.sleep(self._control_sleep_s)

    def apply_pending_controls(self) -> None:
        if not self._enabled:
            return
        if self._reset_requested:
            self._reset_requested = False
            self._reset_env()
        if self._pending_clip_idx is not None or self._pending_clip_start is not None:
            self._apply_clip_selection()

    def on_reset(self, env_ids) -> None:
        if not self._enabled or not self._recenter:
            return
        if self._env_id not in _normalize_env_ids(env_ids):
            return
        self._offset = self._resolve_env_origin()

    def record_step(self) -> None:
        if not self._enabled or self._vr is None or self._robot_root is None:
            return

        if self._force_dt:
            now = time.perf_counter()
            if now < self._next_tick:
                time.sleep(self._next_tick - now)
            now = time.perf_counter()
            self._next_tick = now + self._update_period
        else:
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

        with self._server.atomic():
            self._robot_root.position = root_pos - offset
            self._robot_root.wxyz = root_quat_wxyz

            joints = dof_pos
            if self._joint_order is not None:
                joints = joints[self._joint_order]
            if joints.shape[0] != self._joint_count:
                return
            self._vr.update_cfg(joints.astype(np.float32, copy=False))

            if self._scandots_enabled:
                self._update_scandots(offset)

            if self._vo is None or self._object_root is None:
                return
            if self._show_object_cb is not None and not bool(self._show_object_cb.value):
                self._vo.show_visual = False
                return
            obj_state = self._get_object_state_wxyz()
            if obj_state is None:
                self._vo.show_visual = False
                return
            obj_pos, obj_quat_wxyz = obj_state
            self._object_root.position = obj_pos - offset
            self._object_root.wxyz = obj_quat_wxyz
            self._vo.show_visual = True

        if self._clip_label is not None:
            motion_cmd = self._get_motion_command()
            if motion_cmd is not None and hasattr(motion_cmd, "clip_ids"):
                try:
                    clip_idx = int(motion_cmd.clip_ids[self._env_id].item())
                    clip_names = getattr(motion_cmd.motion, "clip_ids", [])
                    clip_name = clip_names[clip_idx] if 0 <= clip_idx < len(clip_names) else str(clip_idx)
                    self._clip_label.content = f"Current clip: `{clip_name}`"
                except Exception:
                    pass

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
        terrain_state = terrain_mgr.get_state("locomotion_terrain")
        mesh = getattr(terrain_state, "mesh", None)
        if mesh is None:
            terrain = getattr(terrain_state, "terrain", None)
            mesh = getattr(terrain, "mesh", None) if terrain is not None else None
        if mesh is not None:
            try:
                import trimesh  # type: ignore[import-not-found]
            except Exception:
                mesh = None
            else:
                if isinstance(mesh, trimesh.Scene):
                    mesh = mesh.dump(concatenate=True)
                if not isinstance(mesh, trimesh.Trimesh):
                    try:
                        mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces))
                    except Exception:
                        mesh = None

        if mesh is None:
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
            try:
                import trimesh  # type: ignore[import-not-found]
            except Exception:
                return
            ground_mesh = trimesh.creation.box(extents=(8.0, 8.0, 0.01))
            ground_mesh.apply_translation([0.0, 0.0, -0.005])
            self._ground_handle = self._server.scene.add_mesh_simple(
                "/ground",
                ground_mesh.vertices,
                ground_mesh.faces,
                color=LIGHT_BLUE,
                side="double",
            )
            return
        self._terrain_handle = self._server.scene.add_mesh_simple(
            "/terrain",
            mesh.vertices,
            mesh.faces,
            color=LIGHT_BLUE,
            side="double",
        )

    def _setup_controls(self) -> None:
        if self._server is None:
            return

        with self._server.gui.add_folder("Visualization"):
            self._show_scandots_cb = self._server.gui.add_checkbox(
                "Show Scandots",
                initial_value=self._scandots_enabled,
                hint="Toggle mesh-sampled point cloud",
            )
            self._scandots_size_slider = self._server.gui.add_slider(
                "Scandots Size",
                min=0.001,
                max=0.05,
                step=0.001,
                initial_value=float(self._scandots_point_size),
                hint="Point size for scandots visualization",
            )

        @self._show_scandots_cb.on_update
        def _(_evt) -> None:
            self._scandots_enabled = bool(self._show_scandots_cb.value)
            if not self._scandots_enabled and self._scandots_handle is not None:
                self._scandots_handle.visible = False

        @self._scandots_size_slider.on_update
        def _(_evt) -> None:
            self._scandots_point_size = float(self._scandots_size_slider.value)
            if self._scandots_handle is not None:
                self._scandots_handle.point_size = float(self._scandots_point_size)

        sim_cfg = getattr(self._env.simulator, "simulator_config", None)
        if sim_cfg is not None and hasattr(sim_cfg, "contact_force_viz"):
            with self._server.gui.add_folder("Contact Forces"):
                self._contact_force_cb = self._server.gui.add_checkbox(
                    "Show Contact Forces",
                    initial_value=bool(getattr(sim_cfg, "contact_force_viz", False)),
                    hint="Toggle contact force debug lines",
                )
                self._contact_force_scale_slider = self._server.gui.add_slider(
                    "Force Scale",
                    min=0.0,
                    max=0.01,
                    step=0.0001,
                    initial_value=float(getattr(sim_cfg, "contact_force_viz_scale", 0.001)),
                    hint="Scale factor for contact force lines",
                )
                self._contact_force_threshold_slider = self._server.gui.add_slider(
                    "Force Threshold",
                    min=0.0,
                    max=50.0,
                    step=0.1,
                    initial_value=float(getattr(sim_cfg, "contact_force_viz_threshold", 1.0)),
                    hint="Minimum force magnitude to display",
                )

            def _sync_contact_cfg() -> None:
                if self._contact_force_cb is None:
                    return
                self._set_sim_cfg("contact_force_viz", bool(self._contact_force_cb.value))
                if self._contact_force_scale_slider is not None:
                    self._set_sim_cfg("contact_force_viz_scale", float(self._contact_force_scale_slider.value))
                if self._contact_force_threshold_slider is not None:
                    self._set_sim_cfg(
                        "contact_force_viz_threshold",
                        float(self._contact_force_threshold_slider.value),
                    )

            @self._contact_force_cb.on_update
            def _(_evt) -> None:
                _sync_contact_cfg()

            @self._contact_force_scale_slider.on_update
            def _(_evt) -> None:
                _sync_contact_cfg()

            @self._contact_force_threshold_slider.on_update
            def _(_evt) -> None:
                _sync_contact_cfg()

        with self._server.gui.add_folder("Advanced", expand_by_default=False):
            with self._server.gui.add_folder("Simulation Control"):
                self._play_control = self._server.gui.add_checkbox(
                    "Play",
                    initial_value=True,
                    hint="Toggle simulation play/pause",
                )
                self._step_button = self._server.gui.add_button(
                    "Step",
                    hint="Step the simulation forward by one frame",
                )
                self._reset_button = self._server.gui.add_button(
                    "Reset",
                    hint="Reset the selected environment",
                )

            @self._step_button.on_click
            def _(_evt) -> None:
                self._step_requested = True

            @self._reset_button.on_click
            def _(_evt) -> None:
                self._reset_requested = True

            with self._server.gui.add_folder("World Viz"):
                self._show_robot_cb = self._server.gui.add_checkbox(
                    "Show Robot",
                    initial_value=bool(getattr(self._vr, "show_visual", True)),
                    hint="Toggle robot mesh visibility",
                )
                if self._vo is not None:
                    self._show_object_cb = self._server.gui.add_checkbox(
                        "Show Object",
                        initial_value=bool(getattr(self._vo, "show_visual", True)),
                        hint="Toggle object mesh visibility",
                    )
                if self._terrain_handle is not None or self._ground_handle is not None:
                    self._show_terrain_cb = self._server.gui.add_checkbox(
                        "Show Terrain",
                        initial_value=bool(
                            getattr(self._terrain_handle or self._ground_handle, "visible", True)
                        ),
                        hint="Toggle terrain mesh visibility",
                    )
                if self._grid_handle is not None:
                    self._show_grid_cb = self._server.gui.add_checkbox(
                        "Show Grid",
                        initial_value=bool(getattr(self._grid_handle, "visible", False)),
                        hint="Toggle the ground grid",
                    )
                self._recenter_cb = self._server.gui.add_checkbox(
                    "Recenter to Env Origin",
                    initial_value=bool(self._recenter),
                    hint="Keep the selected env centered in view",
                )

            def _apply_world_vis() -> None:
                if self._show_robot_cb is not None and self._vr is not None:
                    try:
                        self._vr.show_visual = bool(self._show_robot_cb.value)
                    except Exception:
                        pass
                if self._show_object_cb is not None and self._vo is not None:
                    try:
                        self._vo.show_visual = bool(self._show_object_cb.value)
                    except Exception:
                        pass
                if self._show_terrain_cb is not None:
                    handle = self._terrain_handle or self._ground_handle
                    self._set_handle_visible(handle, bool(self._show_terrain_cb.value))
                if self._show_grid_cb is not None:
                    self._set_handle_visible(self._grid_handle, bool(self._show_grid_cb.value))

            if self._show_robot_cb is not None:
                @self._show_robot_cb.on_update
                def _(_evt) -> None:
                    _apply_world_vis()

            if self._show_object_cb is not None:
                @self._show_object_cb.on_update
                def _(_evt) -> None:
                    _apply_world_vis()

            if self._show_terrain_cb is not None:
                @self._show_terrain_cb.on_update
                def _(_evt) -> None:
                    _apply_world_vis()

            if self._show_grid_cb is not None:
                @self._show_grid_cb.on_update
                def _(_evt) -> None:
                    _apply_world_vis()

            if self._recenter_cb is not None:
                @self._recenter_cb.on_update
                def _(_evt) -> None:
                    self._recenter = bool(self._recenter_cb.value)
                    if self._recenter:
                        offset = self._resolve_env_origin()
                        if offset is None:
                            offset = np.zeros(3, dtype=np.float32)
                        self._offset = offset
                    else:
                        self._offset = np.zeros(3, dtype=np.float32)

            clip_gui_enabled = os.environ.get("VISER_ENABLE_CLIP_GUI", "1").lower() not in (
                "0",
                "false",
                "no",
            )
            if clip_gui_enabled:
                motion_cmd = self._get_motion_command()
                if motion_cmd is not None and hasattr(motion_cmd, "motion"):
                    clip_names = list(getattr(motion_cmd.motion, "clip_ids", []))
                    if clip_names:
                        self._clip_names = clip_names
                        with self._server.gui.add_folder("Clip Playback"):
                            if len(clip_names) > 1:
                                self._clip_dropdown = self._server.gui.add_dropdown(
                                    "Clip",
                                    options=clip_names,
                                    initial_value=clip_names[0],
                                    hint="Select which motion clip to visualize",
                                )
                            else:
                                self._clip_dropdown = None
                                self._clip_label = self._server.gui.add_markdown(
                                    f"Clip: `{clip_names[0]}`"
                                )
                            self._clip_start_slider = self._server.gui.add_slider(
                                "Clip Start Frame",
                                min=0,
                                max=10000,
                                step=1,
                                initial_value=0,
                                hint="Select starting frame in the clip",
                            )
                            self._clip_lock_cb = self._server.gui.add_checkbox(
                                "Lock Clip",
                                initial_value=True,
                                hint="Keep the selected clip fixed across resets",
                            )
                            self._clip_apply = self._server.gui.add_button("Apply Clip")
                            if self._clip_label is None:
                                self._clip_label = self._server.gui.add_markdown("")

                        def _update_clip_slider(idx: int | None) -> None:
                            if self._clip_start_slider is None or idx is None:
                                return
                            length = self._get_clip_length(motion_cmd, idx)
                            if length is None:
                                return
                            max_frame = max(0, int(length) - 2)
                            self._clip_start_slider.max = max_frame
                            if int(self._clip_start_slider.value) > max_frame:
                                self._clip_start_slider.value = max_frame

                        def _queue_clip_change() -> None:
                            idx = self._current_clip_index(motion_cmd)
                            if idx is None:
                                return
                            self._pending_clip_idx = idx
                            if self._clip_start_slider is not None:
                                self._pending_clip_start = int(self._clip_start_slider.value)
                            _update_clip_slider(idx)

                        if self._clip_dropdown is not None:
                            @self._clip_dropdown.on_update
                            def _(_evt) -> None:
                                _queue_clip_change()

                        @self._clip_start_slider.on_update
                        def _(_evt) -> None:
                            _queue_clip_change()

                        @self._clip_lock_cb.on_update
                        def _(_evt) -> None:
                            if bool(self._clip_lock_cb.value):
                                _queue_clip_change()
                            else:
                                try:
                                    motion_cmd.set_forced_clip(None)
                                except Exception:
                                    motion_cmd._forced_clip_idx = None

                        @self._clip_apply.on_click
                        def _(_evt) -> None:
                            _queue_clip_change()

                        # Force the initial clip so we don't randomize across the bank.
                        self._pending_clip_idx = 0
                        _update_clip_slider(0)

    def _get_motion_command(self):
        cmd_mgr = getattr(self._env, "command_manager", None)
        if cmd_mgr is None:
            return None
        return cmd_mgr.get_state("motion_command")

    def _current_clip_index(self, motion_cmd) -> int | None:
        if self._clip_dropdown is not None:
            try:
                return self._clip_names.index(str(self._clip_dropdown.value))
            except Exception:
                return None
        if motion_cmd is None or not hasattr(motion_cmd, "clip_ids"):
            return None
        try:
            return int(motion_cmd.clip_ids[self._env_id].item())
        except Exception:
            return None

    def _get_clip_length(self, motion_cmd, clip_idx: int) -> int | None:
        if motion_cmd is None or not hasattr(motion_cmd, "motion"):
            return None
        lengths = getattr(motion_cmd.motion, "clip_lengths", None)
        if lengths is None:
            return None
        try:
            if isinstance(lengths, torch.Tensor):
                return int(lengths[clip_idx].item())
            return int(lengths[clip_idx])
        except Exception:
            return None

    def _reset_env(self) -> None:
        if not hasattr(self._env, "reset_envs_idx"):
            return
        env_ids = torch.tensor([self._env_id], device=self._env.device, dtype=torch.long)
        self._env.reset_envs_idx(env_ids)
        if hasattr(self._env, "reset_buf"):
            self._env.reset_buf[env_ids] = 0
        if hasattr(self._env, "time_out_buf"):
            self._env.time_out_buf[env_ids] = 0

    def _apply_clip_selection(self) -> None:
        motion_cmd = self._get_motion_command()
        if motion_cmd is None:
            self._pending_clip_idx = None
            self._pending_clip_start = None
            return
        clip_idx = self._pending_clip_idx
        clip_start = self._pending_clip_start
        self._pending_clip_idx = None
        self._pending_clip_start = None
        if clip_idx is None:
            return
        lock_enabled = True
        if self._clip_lock_cb is not None:
            lock_enabled = bool(self._clip_lock_cb.value)
        if lock_enabled:
            try:
                motion_cmd.set_forced_clip(int(clip_idx))
            except Exception:
                motion_cmd._forced_clip_idx = int(clip_idx)
        else:
            try:
                motion_cmd.set_forced_clip(None)
            except Exception:
                motion_cmd._forced_clip_idx = None
        self._force_clip_state(motion_cmd, int(clip_idx), clip_start)

    def _force_clip_state(self, motion_cmd, clip_idx: int, clip_start: int | None) -> None:
        env_ids = torch.tensor([self._env_id], device=self._env.device, dtype=torch.long)
        self._env.reset_envs_idx(env_ids)

        clip_length = self._get_clip_length(motion_cmd, clip_idx)
        max_valid = max(0, int(clip_length or 1) - 2)
        start_frame = max(0, min(int(clip_start or 0), max_valid))

        motion_cmd.clip_ids[env_ids] = int(clip_idx)
        motion_cmd.time_steps[env_ids] = start_frame
        if motion_cmd.motion_cfg.align_motion_to_init_yaw:
            motion_cmd._update_motion_alignment(env_ids)

        root_pos = motion_cmd.root_pos_w[env_ids].clone()
        root_rot = motion_cmd.root_quat_w[env_ids].clone()

        motion_idx = motion_cmd._get_motion_indices(motion_cmd.time_steps[env_ids], env_ids)
        root_lin_vel = motion_cmd.motion.body_lin_vel_w[motion_idx, 0].clone()
        root_ang_vel = motion_cmd.motion.body_ang_vel_w[motion_idx, 0].clone()
        if motion_cmd.motion_cfg.align_motion_to_init_yaw:
            root_lin_vel = motion_cmd._apply_motion_alignment_vec(root_lin_vel)
            root_ang_vel = motion_cmd._apply_motion_alignment_vec(root_ang_vel)

        dof_pos = motion_cmd.joint_pos[env_ids].clone()
        dof_vel = motion_cmd.joint_vel[env_ids].clone()

        sim = self._env.simulator
        sim.dof_pos[env_ids] = dof_pos
        sim.dof_vel[env_ids] = dof_vel
        sim.robot_root_states[env_ids, :3] = root_pos
        sim.robot_root_states[env_ids, 3:7] = root_rot
        sim.robot_root_states[env_ids, 7:10] = root_lin_vel
        sim.robot_root_states[env_ids, 10:13] = root_ang_vel

        if hasattr(sim, "set_actor_root_state_tensor_robots"):
            sim.set_actor_root_state_tensor_robots(env_ids, sim.robot_root_states)
        else:
            sim.set_actor_root_state_tensor(env_ids, sim.all_root_states)

        if hasattr(sim, "set_dof_state_tensor_robots"):
            sim.set_dof_state_tensor_robots(env_ids, sim.dof_state)
        else:
            sim.set_dof_state_tensor(env_ids, sim.dof_state)

        if motion_cmd.motion.has_object:
            obj_pos = motion_cmd.object_pos_w[env_ids]
            obj_ori = motion_cmd.object_quat_w[env_ids]
            obj_lin_vel = motion_cmd.object_lin_vel_w[env_ids]
            obj_states = torch.cat([obj_pos, obj_ori, obj_lin_vel, torch.zeros_like(obj_lin_vel)], dim=-1)
            sim.set_actor_states([motion_cmd.object_name], env_ids, obj_states)

        if hasattr(sim, "scene") and hasattr(sim.scene, "write_data_to_sim"):
            sim.scene.write_data_to_sim()
        sim.refresh_sim_tensors()

        motion_cmd._update_future_target_poses()
        if hasattr(self._env, "_refresh_envs_after_reset"):
            self._env._refresh_envs_after_reset(env_ids)

        if hasattr(self._env, "reset_buf"):
            self._env.reset_buf[env_ids] = 0
        if hasattr(self._env, "time_out_buf"):
            self._env.time_out_buf[env_ids] = 0

        self._env._compute_observations()
        self._env._post_compute_observations_callback()
        self._env._clip_observations()

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

    def _update_scandots(self, offset: np.ndarray) -> None:
        if not self._server or not self._scandots_enabled:
            return

        perception_mgr = getattr(self._env, "perception_manager", None)
        if perception_mgr is None:
            if not self._scandots_warned:
                logger.warning("Viser scandots requested but perception_manager is unavailable.")
                self._scandots_warned = True
            return

        env_ids = torch.tensor([self._env_id], device=self._env.device, dtype=torch.long)
        try:
            with torch.no_grad():
                result = perception_mgr.get_camera_scandots_points(env_ids, include_misses=False)
        except Exception as exc:
            if not self._scandots_warned:
                logger.warning("Viser scandots disabled: {}", exc)
                self._scandots_warned = True
            return

        if result is None:
            if not self._scandots_warned:
                logger.warning("Viser scandots disabled: perception is not using mesh_raycast_scandots.")
                self._scandots_warned = True
            self._scandots_enabled = False
            return

        points, mask = result
        if points.numel() == 0:
            return
        points_env = points[0]
        mask_env = mask[0]
        if mask_env.numel() > 0:
            points_env = points_env[mask_env]
        if points_env.numel() == 0:
            if self._scandots_handle is not None:
                self._scandots_handle.visible = False
            return

        pts = points_env.detach().cpu().numpy()
        if self._recenter:
            pts = pts - offset

        if self._scandots_handle is None:
            self._scandots_handle = self._server.scene.add_point_cloud(
                "/scandots",
                points=pts.astype(np.float32, copy=False),
                colors=self._scandots_color,
                point_size=float(self._scandots_point_size),
                point_shape="circle",
            )
        else:
            self._scandots_handle.visible = True
            self._scandots_handle.points = pts.astype(np.float32, copy=False)


def _normalize_env_ids(env_ids) -> list[int]:
    if isinstance(env_ids, torch.Tensor):
        return [int(idx.item()) for idx in env_ids.flatten()]
    if isinstance(env_ids, np.ndarray):
        return [int(idx) for idx in env_ids.flatten().tolist()]
    return [int(idx) for idx in env_ids]
