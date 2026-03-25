from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path

import numpy as np
import trimesh
import tyro
from loguru import logger

# Ensure local packages are importable when running from source.
SRC_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
INFER_SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "holosoma_inference"
DEFAULT_TRACKING_MOTION_FILE = REPO_ROOT / "src" / "holosoma" / "holosoma" / "data" / "motions" / "g1_29dof" / "whole_body_tracking" / "sub3_largebox_003_mj_w_obj.npz"
DEFAULT_TRACKING_MODEL_PATH = Path(
    "/data/logs_new/boxer/20260316_200048-g1_29dof_wbt_w_object_extend_20260316_200027_s01_scale_1p0-g1_29dof_wbt_w_object_extend_20260316_200027/model_23500.onnx"
)
for path in (SRC_ROOT, INFER_SRC_ROOT):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from holosoma.utils.viser_utils import ensure_viser_on_path, resolve_viser_port  # noqa: E402

ensure_viser_on_path()

import viser  # type: ignore[import-not-found]  # noqa: E402
from viser.extras import ViserUrdf  # type: ignore[import-not-found]  # noqa: E402

from holosoma.config_types.robot import RobotConfig  # noqa: E402
from holosoma.config_values import robot as robot_values  # noqa: E402
from holosoma.utils.module_utils import get_holosoma_root  # noqa: E402
from holosoma.utils.path import resolve_data_file_path  # noqa: E402
from holosoma_inference.utils.policy_control import PolicyControlPush  # noqa: E402
from holosoma_inference.utils.sim_control import SimControlPush  # noqa: E402
from holosoma_inference.utils.sim_state import SimStateSub  # noqa: E402


@dataclass(frozen=True)
class MujocoSimStateViewerConfig:
    robot: str = "g1_29dof_w_object"
    state_port: int = int(os.environ.get("SIM_STATE_PORT", "5657"))
    control_port: int = int(os.environ.get("SIM_CONTROL_PORT", "5659"))
    policy_control_port: int = int(os.environ.get("POLICY_CONTROL_PORT", "5660"))
    object_actor_name: str = "object"
    port: int = 0
    rate_hz: float = 30.0
    recenter_xy: bool = True
    show_object: bool = True
    object_mesh_mode: str = ""
    show_robot_collision: bool = False
    show_object_collision: bool = False
    mujoco_object_geom_snapshot_path: str = str(REPO_ROOT / "logs" / "live_debug" / "viser_mujoco_object_geoms.json")
    show_ref_body: bool = True
    grid_size: float = 8.0
    launch_rollout: bool = False
    run_script: str = str(REPO_ROOT / "mj_track.sh")
    motion_file: str = str(DEFAULT_TRACKING_MOTION_FILE)
    model_path: str = str(DEFAULT_TRACKING_MODEL_PATH)
    launch_run_seconds: int = 0
    training_headless: bool = True
    rollout_log_path: str = str(REPO_ROOT / "logs" / "live_debug" / "viser_mujoco_sim_state.log")
    auto_reset_after_first_state_sec: float = 0.0


def _resolve_data_path(path: str) -> Path:
    if path.startswith("@holosoma/"):
        return Path(get_holosoma_root()) / path[len("@holosoma/") :]
    return Path(resolve_data_file_path(path)).expanduser().resolve()


def _resolve_robot_config(name: str) -> RobotConfig:
    defaults = robot_values.DEFAULTS
    if name not in defaults:
        raise ValueError(f"Unknown robot '{name}'. Available: {sorted(defaults.keys())}")
    return defaults[name]


def _resolve_repo_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def _resolve_robot_urdf_path(robot_config: RobotConfig) -> Path:
    asset_root = _resolve_data_path(robot_config.asset.asset_root)
    return _resolve_data_path(str(asset_root / robot_config.asset.urdf_file))


def _xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32).reshape(4)
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)


def _normalize_quaternion_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32).reshape(4)
    quat_norm = float(np.linalg.norm(quat_wxyz))
    if quat_norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return quat_wxyz / quat_norm


OBJECT_MESH_MODE_OPTIONS = ("visual", "collision")


def _resolve_object_mesh_mode(mode_raw: object, *, show_object_collision: bool = False) -> str:
    mode = str(mode_raw).strip().lower()
    if mode in OBJECT_MESH_MODE_OPTIONS:
        return mode
    return "collision" if show_object_collision else "visual"


def _geom_supports_visual_mesh(geom_entry: dict) -> bool:
    rgba = np.asarray(geom_entry.get("rgba", [0.75, 0.75, 0.75, 1.0]), dtype=np.float32).reshape(-1)
    alpha = float(rgba[3]) if rgba.shape[0] >= 4 else 1.0
    return alpha > 1e-4


def _mesh_arrays_from_mujoco_geom(geom_entry: dict, collision_view: bool) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int], float] | None:
    geom_type = str(geom_entry.get("type", ""))
    geom_size = np.asarray(geom_entry.get("size", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(-1)
    rgba = np.asarray(geom_entry.get("rgba", [0.75, 0.75, 0.75, 1.0]), dtype=np.float32).reshape(-1)

    mesh: trimesh.Trimesh | None = None
    if geom_type == "mesh":
        mesh_payload = geom_entry.get("mesh")
        if not isinstance(mesh_payload, dict):
            return None
        vertices = np.asarray(mesh_payload.get("vertices", []), dtype=np.float32)
        faces = np.asarray(mesh_payload.get("faces", []), dtype=np.int32)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
            return None
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    elif geom_type == "box" and geom_size.shape[0] >= 3:
        mesh = trimesh.creation.box(extents=2.0 * geom_size[:3])
    elif geom_type == "sphere" and geom_size.shape[0] >= 1:
        mesh = trimesh.creation.uv_sphere(radius=float(geom_size[0]))
    elif geom_type == "capsule" and geom_size.shape[0] >= 2:
        mesh = trimesh.creation.capsule(height=2.0 * float(geom_size[1]), radius=float(geom_size[0]))
    elif geom_type == "cylinder" and geom_size.shape[0] >= 2:
        mesh = trimesh.creation.cylinder(radius=float(geom_size[0]), height=2.0 * float(geom_size[1]))
    elif geom_type == "ellipsoid" and geom_size.shape[0] >= 3:
        mesh = trimesh.creation.uv_sphere(radius=1.0)
        mesh.vertices *= geom_size[:3]

    if mesh is None:
        return None

    if collision_view:
        color = (255, 72, 72)
        opacity = 0.30
    else:
        rgb = np.clip(np.round(rgba[:3] * 255.0), 0, 255).astype(np.int32)
        color = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        opacity = float(np.clip(rgba[3] if rgba.shape[0] >= 4 else 1.0, 0.0, 1.0))

    return (
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.int32),
        color,
        opacity,
    )


def _load_mujoco_object_geom_snapshot(snapshot_path: Path, actor_name: str) -> dict[str, object] | None:
    if not snapshot_path.is_file():
        return None
    try:
        with snapshot_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load MuJoCo geom snapshot {}: {}", snapshot_path, exc)
        return None

    actors = payload.get("actors")
    if not isinstance(actors, dict) or not actors:
        return None
    actor_payload = actors.get(actor_name)
    if actor_payload is None and len(actors) == 1:
        actor_payload = next(iter(actors.values()))
    return actor_payload if isinstance(actor_payload, dict) else None


def _select_actor_state(state: dict, actor_name: str) -> tuple[str | None, np.ndarray | None]:
    actors = state.get("actors")
    if not isinstance(actors, dict) or not actors:
        return None, None

    actor_state = actors.get(actor_name)
    actor_key = actor_name
    if actor_state is None and len(actors) == 1:
        actor_key, actor_state = next(iter(actors.items()))
    if actor_state is None:
        return None, None

    actor_state_np = np.asarray(actor_state, dtype=np.float32).reshape(-1)
    if actor_state_np.shape[0] < 7:
        return None, None
    return actor_key, actor_state_np


def _build_default_joint_viser(robot_config: RobotConfig, viser_joint_names: list[str]) -> np.ndarray:
    default_joint_angles = getattr(robot_config.init_state, "default_joint_angles", {}) or {}
    name_to_robot_idx = {name: idx for idx, name in enumerate(robot_config.dof_names)}
    missing = [name for name in viser_joint_names if name not in name_to_robot_idx]
    if missing:
        raise ValueError(f"Viser URDF joints missing in robot config: {missing}")
    default_joint_robot = np.zeros(len(robot_config.dof_names), dtype=np.float32)
    for idx, name in enumerate(robot_config.dof_names):
        default_joint_robot[idx] = float(default_joint_angles.get(name, 0.0))
    return np.asarray([default_joint_robot[name_to_robot_idx[name]] for name in viser_joint_names], dtype=np.float32)


def _terminate_process_group(proc: subprocess.Popen[bytes] | subprocess.Popen[str] | None, timeout_sec: float = 10.0) -> None:
    if proc is None or proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.1)
    os.killpg(proc.pid, signal.SIGKILL)
    proc.wait(timeout=5.0)


def _build_rollout_command(cfg: MujocoSimStateViewerConfig) -> list[str]:
    run_script = _resolve_repo_path(cfg.run_script)
    if not run_script.is_file():
        raise FileNotFoundError(f"run script not found: {run_script}")
    command = [str(run_script)]
    if cfg.motion_file:
        command.append(str(_resolve_repo_path(cfg.motion_file)))
    if cfg.model_path:
        command.append(str(_resolve_repo_path(cfg.model_path)))
    return command


def view_sim_state(cfg: MujocoSimStateViewerConfig) -> None:
    robot_config = _resolve_robot_config(cfg.robot)
    robot_urdf_path = _resolve_robot_urdf_path(robot_config)
    snapshot_path_default = _resolve_repo_path(cfg.mujoco_object_geom_snapshot_path)

    port = resolve_viser_port(cfg.port)
    server = viser.ViserServer(port=port)
    robot_root = server.scene.add_frame("/robot", show_axes=False)
    ref_root = server.scene.add_frame("/robot_ref", show_axes=bool(cfg.show_ref_body))
    object_root = server.scene.add_frame("/object", show_axes=False)
    object_visual_root = server.scene.add_frame("/object/visual_geoms", show_axes=False)
    object_collision_root = server.scene.add_frame("/object/collision_geoms", show_axes=False)
    server.scene.add_grid("/grid", width=cfg.grid_size, height=cfg.grid_size, position=(0.0, 0.0, 0.0))

    vr = ViserUrdf(
        server,
        urdf_or_path=robot_urdf_path,
        root_node_name="/robot",
        load_collision_meshes=True,
        collision_mesh_color_override=(0.15, 0.7, 1.0, 0.28),
    )
    viser_joint_names = list(vr.get_actuated_joint_names())
    default_joint_viser = _build_default_joint_viser(robot_config, viser_joint_names)
    vr.update_cfg(default_joint_viser)
    vr.show_collision = bool(cfg.show_robot_collision)
    object_visual_handles: list[object] = []
    object_collision_handles: list[object] = []
    loaded_object_snapshot_path: Path | None = None
    object_visual_uses_collision_fallback = False

    def _clear_object_geom_handles() -> None:
        nonlocal object_visual_handles, object_collision_handles, loaded_object_snapshot_path, object_visual_uses_collision_fallback
        for handle in [*object_visual_handles, *object_collision_handles]:
            try:
                handle.remove()
            except Exception:
                pass
        object_visual_handles = []
        object_collision_handles = []
        loaded_object_snapshot_path = None
        object_visual_uses_collision_fallback = False
        object_visual_root.visible = False
        object_collision_root.visible = False

    def _set_object_mesh_visibility(*, show_object: bool, mesh_mode: str) -> None:
        effective_show_visual = False
        effective_show_collision = False
        normalized_mesh_mode = _resolve_object_mesh_mode(mesh_mode)

        if show_object:
            if normalized_mesh_mode == "collision":
                effective_show_collision = bool(object_collision_handles)
            else:
                effective_show_visual = bool(object_visual_handles)
                if not effective_show_visual and object_visual_uses_collision_fallback:
                    effective_show_collision = bool(object_collision_handles)

        object_visual_root.visible = effective_show_visual
        object_collision_root.visible = effective_show_collision
        for handle in object_visual_handles:
            handle.visible = effective_show_visual
        for handle in object_collision_handles:
            handle.visible = effective_show_collision

    def _load_object_geom_handles(snapshot_path: Path) -> bool:
        nonlocal loaded_object_snapshot_path, object_visual_handles, object_collision_handles, object_visual_uses_collision_fallback
        actor_payload = _load_mujoco_object_geom_snapshot(snapshot_path, cfg.object_actor_name)
        if actor_payload is None:
            return False

        geoms = actor_payload.get("geoms")
        if not isinstance(geoms, list):
            return False

        _clear_object_geom_handles()
        visual_count = 0
        collision_count = 0
        for geom_idx, geom_entry_raw in enumerate(geoms):
            if not isinstance(geom_entry_raw, dict):
                continue
            geom_name = str(geom_entry_raw.get("name", f"geom_{geom_idx}"))
            geom_pos = np.asarray(geom_entry_raw.get("relative_pos", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3)
            geom_quat_wxyz = _normalize_quaternion_wxyz(geom_entry_raw.get("relative_quat_wxyz", [1.0, 0.0, 0.0, 0.0]))
            is_collision = bool(geom_entry_raw.get("is_collision", False))

            visual_mesh = _mesh_arrays_from_mujoco_geom(geom_entry_raw, collision_view=False)
            if visual_mesh is not None and (not is_collision or _geom_supports_visual_mesh(geom_entry_raw)):
                vertices, faces, color, opacity = visual_mesh
                object_visual_handles.append(
                    server.scene.add_mesh_simple(
                        f"/object/visual_geoms/{geom_name}_{geom_idx}",
                        vertices=vertices,
                        faces=faces,
                        color=color,
                        opacity=opacity,
                        side="double",
                        position=tuple(geom_pos.tolist()),
                        wxyz=tuple(geom_quat_wxyz.tolist()),
                        visible=False,
                    )
                )
                visual_count += 1

            if is_collision:
                collision_mesh = _mesh_arrays_from_mujoco_geom(geom_entry_raw, collision_view=True)
                if collision_mesh is not None:
                    vertices, faces, color, opacity = collision_mesh
                    object_collision_handles.append(
                        server.scene.add_mesh_simple(
                            f"/object/collision_geoms/{geom_name}_{geom_idx}",
                            vertices=vertices,
                            faces=faces,
                            color=color,
                            opacity=opacity,
                            side="double",
                            position=tuple(geom_pos.tolist()),
                            wxyz=tuple(geom_quat_wxyz.tolist()),
                            visible=False,
                        )
                    )
                    collision_count += 1

        object_visual_uses_collision_fallback = not object_visual_handles and bool(object_collision_handles)
        loaded_object_snapshot_path = snapshot_path
        logger.info(
            "Loaded MuJoCo object geoms from {} (visual={}, collision={}, collision_fallback={})",
            snapshot_path,
            visual_count,
            collision_count,
            object_visual_uses_collision_fallback,
        )
        return True

    with server.gui.add_folder("Sim State"):
        state_md = server.gui.add_markdown("Waiting for simulator state...")
        actor_md = server.gui.add_markdown("")

    with server.gui.add_folder("Display"):
        recenter_cb = server.gui.add_checkbox("Recenter XY", initial_value=bool(cfg.recenter_xy))
        show_object_cb = server.gui.add_checkbox("Show object (MuJoCo)", initial_value=bool(cfg.show_object))
        object_mesh_mode_dropdown = server.gui.add_dropdown(
            "Object mesh",
            options=OBJECT_MESH_MODE_OPTIONS,
            initial_value=_resolve_object_mesh_mode(
                cfg.object_mesh_mode,
                show_object_collision=bool(cfg.show_object_collision),
            ),
        )
        show_robot_collision_cb = server.gui.add_checkbox(
            "Show robot collision (URDF)",
            initial_value=bool(cfg.show_robot_collision),
        )
        show_ref_cb = server.gui.add_checkbox("Show ref body", initial_value=bool(cfg.show_ref_body))
        reset_offset_btn = server.gui.add_button("Reset offset")

    with server.gui.add_folder("Rollout"):
        rollout_md = server.gui.add_markdown("Viewer only")
        reset_rollout_btn = server.gui.add_button("Reset rollout")

    manual_gui_enabled = os.environ.get("VISER_ENABLE_MANUAL_GUI", "1").lower() not in ("0", "false", "no")
    if manual_gui_enabled:
        with server.gui.add_folder("Manual Control", expand_by_default=False):
            policy_md = server.gui.add_markdown("Policy control: `idle`")
            start_policy_btn = server.gui.add_button("Start policy")
            stop_policy_btn = server.gui.add_button("Stop policy")
            init_state_btn = server.gui.add_button("Init state")
            start_motion_clip_btn = server.gui.add_button("Start motion clip")
    else:
        policy_md = None
        start_policy_btn = None
        stop_policy_btn = None
        init_state_btn = None
        start_motion_clip_btn = None

    sub = SimStateSub(port=cfg.state_port)
    sub.start()
    control_pub = SimControlPush(port=cfg.control_port)
    control_pub.start()
    policy_control_pub = PolicyControlPush(port=cfg.policy_control_port)
    policy_control_pub.start()
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

    def _handle_sigterm(_signum, _frame) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _handle_sigterm)

    offset_xy = np.zeros(2, dtype=np.float32)
    offset_initialized = False
    received_first_state = False
    rollout_proc: subprocess.Popen | None = None
    rollout_log_handle: TextIOWrapper | None = None
    rollout_restart_count = 0
    pending_restart_reason = "startup" if cfg.launch_rollout else None
    last_rollout_reason = "idle"
    rollout_log_path = _resolve_repo_path(cfg.rollout_log_path)
    auto_reset_scheduled_at: float | None = None
    auto_reset_done = False
    reset_request_time_monotonic: float | None = None
    reset_pending_clock_rewind = False
    pre_reset_sim_time_ms: int | None = None
    last_seen_sim_time_ms: int | None = None

    def _refresh_rollout_md() -> None:
        if not cfg.launch_rollout:
            rollout_md.content = "launch_rollout: `False`"
            return
        if rollout_proc is None:
            proc_state = "stopped"
            pid = "n/a"
        else:
            poll = rollout_proc.poll()
            proc_state = "running" if poll is None else f"exited({poll})"
            pid = str(rollout_proc.pid)
        rollout_md.content = (
            f"status: `{proc_state}`\n\n"
            f"pid: `{pid}`\n\n"
            f"restart_count: `{rollout_restart_count}`\n\n"
            f"last_reason: `{last_rollout_reason}`\n\n"
            f"reset_mode: `sim-control`\n\n"
            f"log_path: `{rollout_log_path}`"
        )

    def _stop_rollout() -> None:
        nonlocal rollout_proc, rollout_log_handle
        if rollout_proc is not None:
            logger.info("Stopping rollout pid={}", rollout_proc.pid)
            _terminate_process_group(rollout_proc)
            rollout_proc = None
        if rollout_log_handle is not None:
            rollout_log_handle.close()
            rollout_log_handle = None

    def _restart_rollout(reason: str) -> None:
        nonlocal rollout_proc, rollout_log_handle, rollout_restart_count, offset_initialized, received_first_state, pending_restart_reason, last_rollout_reason, auto_reset_scheduled_at, auto_reset_done, reset_request_time_monotonic, reset_pending_clock_rewind, pre_reset_sim_time_ms, last_seen_sim_time_ms
        _stop_rollout()
        _clear_object_geom_handles()
        command = _build_rollout_command(cfg)
        env = os.environ.copy()
        env["RUN_SECONDS"] = str(cfg.launch_run_seconds)
        env["TRAINING_HEADLESS"] = "True" if cfg.training_headless else "False"
        env["HOLOSOMA_MUJOCO_OBJECT_GEOM_SNAPSHOT_PATH"] = str(snapshot_path_default)
        try:
            snapshot_path_default.unlink()
        except FileNotFoundError:
            pass
        rollout_log_path.parent.mkdir(parents=True, exist_ok=True)
        rollout_log_handle = rollout_log_path.open("a", encoding="utf-8")
        rollout_proc = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            preexec_fn=os.setsid,
            stdout=rollout_log_handle,
            stderr=subprocess.STDOUT,
        )
        rollout_restart_count += 1
        last_rollout_reason = reason
        pending_restart_reason = None
        offset_xy[:] = 0.0
        offset_initialized = False
        received_first_state = False
        sub.last_state = None
        auto_reset_scheduled_at = None
        auto_reset_done = False
        reset_request_time_monotonic = None
        reset_pending_clock_rewind = False
        pre_reset_sim_time_ms = None
        last_seen_sim_time_ms = None
        state_md.content = "Waiting for simulator state after reset..."
        actor_md.content = ""
        logger.info("Started rollout pid={} reason={}", rollout_proc.pid, reason)
        _refresh_rollout_md()

    def _request_sim_reset(reason: str) -> None:
        nonlocal pending_restart_reason, offset_initialized, received_first_state, auto_reset_scheduled_at, auto_reset_done, reset_request_time_monotonic, reset_pending_clock_rewind, pre_reset_sim_time_ms
        if control_pub.enabled:
            control_pub.request_reset(reason)
            offset_xy[:] = 0.0
            state_md.content = f"Reset requested over sim-control ({reason})..."
            actor_md.content = ""
            sub.last_state = None
            pending_restart_reason = None
            if bool(recenter_cb.value):
                offset_initialized = False
            received_first_state = False
            auto_reset_scheduled_at = None
            auto_reset_done = True
            reset_request_time_monotonic = time.monotonic()
            reset_pending_clock_rewind = True
            pre_reset_sim_time_ms = last_seen_sim_time_ms
            logger.info("Requested simulator reset over sim-control ({})", reason)
        elif cfg.launch_rollout:
            pending_restart_reason = "gui_restart_fallback"
            state_md.content = "Control channel unavailable, falling back to full restart..."
        else:
            logger.warning("Reset rollout requested, but sim-control is unavailable")

    def _request_policy_action(action: str, label: str) -> None:
        if policy_md is not None:
            policy_md.content = f"Policy control: `{label}`"
        policy_control_pub.request_action(action, source="viser_mujoco_sim_state")
        logger.info("Requested policy action '{}' over policy-control", action)

    @show_object_cb.on_update
    def _(_evt) -> None:
        _set_object_mesh_visibility(
            show_object=bool(show_object_cb.value and object_root.visible),
            mesh_mode=str(object_mesh_mode_dropdown.value),
        )

    @show_robot_collision_cb.on_update
    def _(_evt) -> None:
        vr.show_collision = bool(show_robot_collision_cb.value)

    @object_mesh_mode_dropdown.on_update
    def _(_evt) -> None:
        _set_object_mesh_visibility(
            show_object=bool(show_object_cb.value and object_root.visible),
            mesh_mode=str(object_mesh_mode_dropdown.value),
        )

    @show_ref_cb.on_update
    def _(_evt) -> None:
        ref_root.visible = bool(show_ref_cb.value)

    @reset_offset_btn.on_click
    def _(_evt) -> None:
        nonlocal offset_initialized
        offset_xy[:] = 0.0
        offset_initialized = False

    @reset_rollout_btn.on_click
    def _(_evt) -> None:
        _request_sim_reset("gui_reset")

    if start_policy_btn is not None:
        @start_policy_btn.on_click
        def _(_evt) -> None:
            _request_policy_action("start_policy", "start_policy")

    if stop_policy_btn is not None:
        @stop_policy_btn.on_click
        def _(_evt) -> None:
            _request_policy_action("stop_policy", "stop_policy")

    if init_state_btn is not None:
        @init_state_btn.on_click
        def _(_evt) -> None:
            _request_policy_action("init_state", "init_state")

    if start_motion_clip_btn is not None:
        @start_motion_clip_btn.on_click
        def _(_evt) -> None:
            _request_policy_action("start_motion_clip", "start_motion_clip")

    logger.info("Open viser at http://localhost:{}", port)
    logger.info("Reading split MuJoCo sim-state from tcp://localhost:{}", cfg.state_port)
    logger.info("Sending split MuJoCo policy-control to tcp://localhost:{}", cfg.policy_control_port)
    _refresh_rollout_md()

    try:
        while True:
            if pending_restart_reason is not None:
                _restart_rollout(pending_restart_reason)
            if auto_reset_scheduled_at is not None and time.monotonic() >= auto_reset_scheduled_at:
                _request_sim_reset("auto_test_reset")
            _refresh_rollout_md()

            state = sub.get_state()
            if state is None:
                time.sleep(1.0 / max(cfg.rate_hz, 1.0))
                continue

            robot_root_state = state.get("robot_root_state")
            robot_dof_pos = state.get("robot_dof_pos")
            if robot_root_state is None or robot_dof_pos is None:
                time.sleep(1.0 / max(cfg.rate_hz, 1.0))
                continue

            sim_time_ms = int(state.get("sim_time_ms", 0))
            if reset_pending_clock_rewind and pre_reset_sim_time_ms is not None and sim_time_ms >= pre_reset_sim_time_ms:
                time.sleep(1.0 / max(cfg.rate_hz, 1.0))
                continue

            root_state = np.asarray(robot_root_state, dtype=np.float32).reshape(-1)
            dof_pos = np.asarray(robot_dof_pos, dtype=np.float32).reshape(-1)
            if root_state.shape[0] < 7 or dof_pos.shape[0] < len(robot_config.dof_names):
                time.sleep(1.0 / max(cfg.rate_hz, 1.0))
                continue

            if not received_first_state:
                if reset_request_time_monotonic is None:
                    logger.info(
                        "Received first sim-state: sim_time_ms={}, ref_body={}",
                        int(state.get("sim_time_ms", 0)),
                        state.get("robot_ref_body_name", "n/a"),
                    )
                else:
                    reset_latency_ms = (time.monotonic() - reset_request_time_monotonic) * 1000.0
                    logger.info(
                        "Received first sim-state after reset: sim_time_ms={}, ref_body={}, latency_ms={:.1f}",
                        sim_time_ms,
                        state.get("robot_ref_body_name", "n/a"),
                        reset_latency_ms,
                    )
                    reset_request_time_monotonic = None
                    reset_pending_clock_rewind = False
                    pre_reset_sim_time_ms = None
                received_first_state = True
                if cfg.auto_reset_after_first_state_sec > 0.0 and not auto_reset_done:
                    auto_reset_scheduled_at = time.monotonic() + float(cfg.auto_reset_after_first_state_sec)

            last_seen_sim_time_ms = sim_time_ms

            if bool(recenter_cb.value) and not offset_initialized:
                offset_xy[:] = root_state[:2]
                offset_initialized = True

            joint_viser = dof_pos[: len(robot_config.dof_names)]
            name_to_robot_idx = {name: idx for idx, name in enumerate(robot_config.dof_names)}
            joint_viser = np.asarray([joint_viser[name_to_robot_idx[name]] for name in viser_joint_names], dtype=np.float32)

            root_pos = root_state[:3].copy()
            if bool(recenter_cb.value):
                root_pos[:2] -= offset_xy
            root_quat_wxyz = _xyzw_to_wxyz(root_state[3:7])

            ref_state = state.get("robot_ref_state")
            ref_state_np = None
            if ref_state is not None:
                ref_state_np = np.asarray(ref_state, dtype=np.float32).reshape(-1)

            actor_key, object_state = _select_actor_state(state, cfg.object_actor_name)
            snapshot_path_raw = state.get("mujoco_object_geom_snapshot_path")
            snapshot_path = snapshot_path_default
            if isinstance(snapshot_path_raw, str) and snapshot_path_raw.strip():
                snapshot_path = Path(snapshot_path_raw).expanduser().resolve()
            if loaded_object_snapshot_path is None and snapshot_path.is_file():
                _load_object_geom_handles(snapshot_path)

            with server.atomic():
                robot_root.position = tuple(root_pos.tolist())
                robot_root.wxyz = tuple(root_quat_wxyz.tolist())
                vr.update_cfg(joint_viser)

                if ref_state_np is not None and ref_state_np.shape[0] >= 7:
                    ref_pos = ref_state_np[:3].copy()
                    if bool(recenter_cb.value):
                        ref_pos[:2] -= offset_xy
                    ref_root.position = tuple(ref_pos.tolist())
                    ref_root.wxyz = tuple(_xyzw_to_wxyz(ref_state_np[3:7]).tolist())
                    ref_root.visible = bool(show_ref_cb.value)
                else:
                    ref_root.visible = False

                if object_state is not None:
                    object_pos = object_state[:3].copy()
                    if bool(recenter_cb.value):
                        object_pos[:2] -= offset_xy
                    object_root.position = tuple(object_pos.tolist())
                    object_root.wxyz = tuple(_xyzw_to_wxyz(object_state[3:7]).tolist())
                    object_root.visible = True
                    _set_object_mesh_visibility(
                        show_object=bool(show_object_cb.value),
                        mesh_mode=str(object_mesh_mode_dropdown.value),
                    )
                else:
                    object_root.visible = False
                    _set_object_mesh_visibility(show_object=False, mesh_mode=str(object_mesh_mode_dropdown.value))

            ref_body_name = state.get("robot_ref_body_name", "n/a")
            object_robot_contacts = int(state.get("object_robot_contact_count", 0))
            object_scene_contacts = int(state.get("object_scene_contact_count", 0))
            state_md.content = (
                f"sim_time_ms: `{sim_time_ms}`\n\n"
                f"ref_body: `{ref_body_name}`\n\n"
                f"robot_root_xyz: `{np.array2string(root_state[:3], precision=4)}`\n\n"
                f"object_robot_contacts: `{object_robot_contacts}`\n\n"
                f"object_scene_contacts: `{object_scene_contacts}`"
            )
            actor_label = actor_key if actor_key is not None else "none"
            snapshot_label = str(loaded_object_snapshot_path) if loaded_object_snapshot_path is not None else "pending"
            requested_mesh_mode = _resolve_object_mesh_mode(str(object_mesh_mode_dropdown.value))
            resolved_mesh_mode = requested_mesh_mode
            if requested_mesh_mode == "visual" and object_visual_uses_collision_fallback:
                resolved_mesh_mode = "visual -> collision fallback"
            available_mesh_modes = []
            if object_visual_handles:
                available_mesh_modes.append("visual")
            if object_collision_handles:
                available_mesh_modes.append("collision")
            object_geom_mode = (
                ", ".join(available_mesh_modes)
                if available_mesh_modes
                else "pending"
            )
            actor_md.content = (
                f"object_actor: `{actor_label}`\n\n"
                f"object_geom_source: `MuJoCo geom snapshot`\n\n"
                f"object_mesh_view: `{resolved_mesh_mode}`\n\n"
                f"object_geom_mode: `{object_geom_mode}`\n\n"
                f"snapshot_path: `{snapshot_label}`"
            )

            time.sleep(1.0 / max(cfg.rate_hz, 1.0))
    except KeyboardInterrupt:
        logger.info("Stopping viser MuJoCo sim-state viewer")
    finally:
        _stop_rollout()
        policy_control_pub.close()
        control_pub.close()
        sub.close()
        signal.signal(signal.SIGTERM, previous_sigterm_handler)


def main() -> None:
    cfg = tyro.cli(MujocoSimStateViewerConfig)
    view_sim_state(cfg)


if __name__ == "__main__":
    main()
