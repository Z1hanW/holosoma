from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path

import mujoco
import numpy as np
import tyro
from loguru import logger

# Ensure local packages are importable when running from source.
SRC_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
INFER_SRC_ROOT = REPO_ROOT / "src" / "holosoma_inference"
MJVISER_SRC_ROOT = INFER_SRC_ROOT / "mjviser" / "src"
DEFAULT_TRACKING_MOTION_FILE = REPO_ROOT / "src" / "holosoma" / "holosoma" / "data" / "motions" / "g1_29dof" / "whole_body_tracking" / "sub3_largebox_003_mj_w_obj.npz"
DEFAULT_TRACKING_MODEL_PATH = Path(
    "/data/logs_new/boxer/20260316_200048-g1_29dof_wbt_w_object_extend_20260316_200027_s01_scale_1p0-g1_29dof_wbt_w_object_extend_20260316_200027/model_23500.onnx"
)
for path in (SRC_ROOT, INFER_SRC_ROOT, MJVISER_SRC_ROOT):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from holosoma.config_types.robot import RobotConfig  # noqa: E402
from holosoma.config_values import robot as robot_values  # noqa: E402
from holosoma.simulator.mujoco.tensor_views import quat_rotate_inverse_mujoco  # noqa: E402
from holosoma.utils.path import resolve_data_file_path  # noqa: E402
from holosoma.utils.module_utils import get_holosoma_root  # noqa: E402
from holosoma.utils.viser_utils import resolve_viser_port  # noqa: E402
from holosoma_inference.utils.policy_control import PolicyControlPush  # noqa: E402
from holosoma_inference.utils.sim_control import SimControlPush  # noqa: E402
from holosoma_inference.utils.sim_state import SimStateSub  # noqa: E402
from mjviser import ViserMujocoScene  # noqa: E402
import viser  # type: ignore[import-not-found]  # noqa: E402


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class MjviserMujocoSimStateViewerConfig:
    robot: str = "g1_29dof_w_object"
    state_port: int = int(os.environ.get("SIM_STATE_PORT", "5657"))
    control_port: int = int(os.environ.get("SIM_CONTROL_PORT", "5659"))
    policy_control_port: int = int(os.environ.get("POLICY_CONTROL_PORT", "5660"))
    object_actor_name: str = "object"
    port: int = 0
    rate_hz: float = 30.0
    launch_rollout: bool = False
    run_script: str = str(REPO_ROOT / "mj_track.sh")
    motion_file: str = str(DEFAULT_TRACKING_MOTION_FILE)
    model_path: str = str(DEFAULT_TRACKING_MODEL_PATH)
    launch_run_seconds: int = 0
    training_headless: bool = True
    rollout_log_path: str = str(REPO_ROOT / "logs" / "live_debug" / "mjviser_mujoco_sim_state.log")
    mujoco_scene_xml_snapshot_path: str = str(REPO_ROOT / "logs" / "live_debug" / "mjviser_mujoco_scene.xml")
    auto_reset_after_first_state_sec: float = 0.0
    show_ref_body: bool = True
    default_pose_init: bool = _env_flag(
        "HOLOSOMA_DEFAULT_POSE_INIT",
        default=os.environ.get("SIM_MOTION_INIT_MODE", "").strip().lower() == "training_default_pose",
    )


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


def _xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32).reshape(4)
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)


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


def _build_rollout_command(cfg: MjviserMujocoSimStateViewerConfig) -> list[str]:
    run_script = _resolve_repo_path(cfg.run_script)
    if not run_script.is_file():
        raise FileNotFoundError(f"run script not found: {run_script}")
    command = [str(run_script)]
    if cfg.motion_file:
        command.append(str(_resolve_repo_path(cfg.motion_file)))
    if cfg.model_path:
        command.append(str(_resolve_repo_path(cfg.model_path)))
    return command


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
    if actor_state_np.shape[0] < 13:
        return None, None
    return actor_key, actor_state_np


def _infer_prefixed_joint_name(model: mujoco.MjModel, clean_name: str) -> str:
    exact_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, clean_name)
    if exact_id != -1:
        return clean_name
    matches = []
    for joint_id in range(model.njnt):
        joint_name = model.joint(joint_id).name
        if joint_name == clean_name or joint_name.endswith(f"_{clean_name}"):
            matches.append(joint_name)
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Could not uniquely resolve MuJoCo joint name for '{clean_name}'")


def _resolve_robot_state_layout(state: dict, model: mujoco.MjModel, robot_config: RobotConfig) -> tuple[int, int, np.ndarray, np.ndarray]:
    qpos_addr = state.get("mujoco_robot_qpos_addr")
    qvel_addr = state.get("mujoco_robot_qvel_addr")
    dof_qpos_addrs = state.get("mujoco_robot_dof_qpos_addrs")
    dof_qvel_addrs = state.get("mujoco_robot_dof_qvel_addrs")
    if (
        isinstance(qpos_addr, int)
        and isinstance(qvel_addr, int)
        and isinstance(dof_qpos_addrs, list)
        and isinstance(dof_qvel_addrs, list)
        and len(dof_qpos_addrs) == len(robot_config.dof_names)
        and len(dof_qvel_addrs) == len(robot_config.dof_names)
    ):
        return (
            int(qpos_addr),
            int(qvel_addr),
            np.asarray(dof_qpos_addrs, dtype=np.int32),
            np.asarray(dof_qvel_addrs, dtype=np.int32),
        )

    freejoint_name = _infer_prefixed_joint_name(model, "floating_base_joint")
    freejoint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, freejoint_name)
    if freejoint_id == -1:
        raise ValueError(f"Robot freejoint '{freejoint_name}' not found in model")
    if model.jnt_type[freejoint_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"Joint '{freejoint_name}' is not a freejoint")
    qpos_addr = int(model.jnt_qposadr[freejoint_id])
    qvel_addr = int(model.jnt_dofadr[freejoint_id])
    dof_qpos = []
    dof_qvel = []
    for dof_name in robot_config.dof_names:
        joint_name = _infer_prefixed_joint_name(model, dof_name)
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id == -1:
            raise ValueError(f"Robot DOF joint '{joint_name}' not found in model")
        dof_qpos.append(int(model.jnt_qposadr[joint_id]))
        dof_qvel.append(int(model.jnt_dofadr[joint_id]))
    return qpos_addr, qvel_addr, np.asarray(dof_qpos, dtype=np.int32), np.asarray(dof_qvel, dtype=np.int32)


def _resolve_actor_root_layouts(state: dict) -> dict[str, tuple[int, int]]:
    payload = state.get("mujoco_actor_root_metadata")
    if not isinstance(payload, dict):
        return {}
    resolved: dict[str, tuple[int, int]] = {}
    for name, meta in payload.items():
        if not isinstance(name, str) or not isinstance(meta, dict):
            continue
        qpos_addr = meta.get("qpos_addr")
        qvel_addr = meta.get("qvel_addr")
        if isinstance(qpos_addr, int) and isinstance(qvel_addr, int):
            resolved[name] = (int(qpos_addr), int(qvel_addr))
    return resolved


def _apply_freejoint_state(data: mujoco.MjData, qpos_addr: int, qvel_addr: int, actor_state: np.ndarray) -> None:
    quat_mj = np.array(
        [actor_state[6], actor_state[3], actor_state[4], actor_state[5]],
        dtype=np.float64,
    )
    ang_vel_local = quat_rotate_inverse_mujoco(quat_mj, actor_state[10:13])
    data.qpos[qpos_addr : qpos_addr + 3] = actor_state[:3]
    data.qpos[qpos_addr + 3 : qpos_addr + 7] = quat_mj
    data.qvel[qvel_addr : qvel_addr + 3] = actor_state[7:10]
    data.qvel[qvel_addr + 3 : qvel_addr + 6] = ang_vel_local


def view_sim_state(cfg: MjviserMujocoSimStateViewerConfig) -> None:
    robot_config = _resolve_robot_config(cfg.robot)
    port = resolve_viser_port(cfg.port)
    server = viser.ViserServer(port=port)
    ref_root = server.scene.add_frame("/holosoma_ref", show_axes=bool(cfg.show_ref_body))

    with server.gui.add_folder("Split Sim"):
        state_md = server.gui.add_markdown("Waiting for simulator state...")
        actor_md = server.gui.add_markdown("")
        show_ref_cb = server.gui.add_checkbox("Show ref body", initial_value=bool(cfg.show_ref_body))

    with server.gui.add_folder("Rollout"):
        rollout_md = server.gui.add_markdown("Viewer only")
        reset_rollout_btn = server.gui.add_button("Reset rollout")
        default_pose_init_cb = server.gui.add_checkbox(
            "Default pose init",
            initial_value=bool(cfg.default_pose_init),
            hint="Restart/reset rollout from the robot default pose instead of the motion pose.",
        )

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

    rollout_proc: subprocess.Popen | None = None
    rollout_log_handle: TextIOWrapper | None = None
    rollout_restart_count = 0
    pending_restart_reason = "startup" if cfg.launch_rollout else None
    last_rollout_reason = "idle"
    rollout_log_path = _resolve_repo_path(cfg.rollout_log_path)
    scene_xml_snapshot_path_default = _resolve_repo_path(cfg.mujoco_scene_xml_snapshot_path)
    auto_reset_scheduled_at: float | None = None
    auto_reset_done = False
    reset_request_time_monotonic: float | None = None
    reset_pending_clock_rewind = False
    pre_reset_sim_time_ms: int | None = None
    last_seen_sim_time_ms: int | None = None
    received_first_state = False

    loaded_scene_path: Path | None = None
    scene: ViserMujocoScene | None = None
    model: mujoco.MjModel | None = None
    data: mujoco.MjData | None = None
    default_qpos: np.ndarray | None = None
    robot_qpos_addr: int | None = None
    robot_qvel_addr: int | None = None
    dof_qpos_addrs: np.ndarray | None = None
    dof_qvel_addrs: np.ndarray | None = None
    actor_layouts: dict[str, tuple[int, int]] = {}

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
            f"default_pose_init: `{bool(default_pose_init_cb.value)}`\n\n"
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
        nonlocal rollout_proc, rollout_log_handle, rollout_restart_count, pending_restart_reason, last_rollout_reason
        nonlocal auto_reset_scheduled_at, auto_reset_done, reset_request_time_monotonic, reset_pending_clock_rewind
        nonlocal pre_reset_sim_time_ms, last_seen_sim_time_ms, received_first_state, loaded_scene_path
        _stop_rollout()
        command = _build_rollout_command(cfg)
        env = os.environ.copy()
        env["RUN_SECONDS"] = str(cfg.launch_run_seconds)
        env["TRAINING_HEADLESS"] = "True" if cfg.training_headless else "False"
        env["HOLOSOMA_MUJOCO_SCENE_XML_SNAPSHOT_PATH"] = str(scene_xml_snapshot_path_default)
        env["HOLOSOMA_DEFAULT_POSE_INIT"] = "1" if bool(default_pose_init_cb.value) else "0"
        env["SIM_MOTION_INIT_MODE"] = "training_default_pose" if bool(default_pose_init_cb.value) else "raw_motion"
        env["HOLOSOMA_RESET_TO_DEFAULT_POSE"] = "1" if bool(default_pose_init_cb.value) else "0"
        try:
            scene_xml_snapshot_path_default.unlink()
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
        auto_reset_scheduled_at = None
        auto_reset_done = False
        reset_request_time_monotonic = None
        reset_pending_clock_rewind = False
        pre_reset_sim_time_ms = None
        last_seen_sim_time_ms = None
        received_first_state = False
        loaded_scene_path = None
        state_md.content = "Waiting for simulator state after reset..."
        actor_md.content = ""
        logger.info("Started rollout pid={} reason={}", rollout_proc.pid, reason)
        _refresh_rollout_md()

    def _request_sim_reset(reason: str) -> None:
        nonlocal pending_restart_reason, auto_reset_scheduled_at, auto_reset_done
        nonlocal reset_request_time_monotonic, reset_pending_clock_rewind, pre_reset_sim_time_ms, received_first_state
        if control_pub.enabled:
            control_pub.request_reset(reason)
            state_md.content = f"Reset requested over sim-control ({reason})..."
            actor_md.content = ""
            sub.last_state = None
            pending_restart_reason = None
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
        policy_control_pub.request_action(action, source="mjviser_mujoco_sim_state")
        logger.info("Requested policy action '{}' over policy-control", action)

    def _load_scene_from_snapshot(snapshot_path: Path, state: dict) -> bool:
        nonlocal loaded_scene_path, scene, model, data, default_qpos
        nonlocal robot_qpos_addr, robot_qvel_addr, dof_qpos_addrs, dof_qvel_addrs, actor_layouts
        try:
            loaded_model = mujoco.MjModel.from_xml_path(str(snapshot_path))
            loaded_data = mujoco.MjData(loaded_model)
            mujoco.mj_resetData(loaded_model, loaded_data)
            robot_qpos_addr_val, robot_qvel_addr_val, dof_qpos_addrs_val, dof_qvel_addrs_val = _resolve_robot_state_layout(
                state,
                loaded_model,
                robot_config,
            )
            actor_layouts_val = _resolve_actor_root_layouts(state)
        except Exception as exc:
            logger.warning("Failed to load mjviser scene from {}: {}", snapshot_path, exc)
            return False

        for path in ("/sim", "/holosoma_ref"):
            try:
                server.scene.remove_by_name(path)
            except Exception:
                pass
        ref_root.wxyz = (1.0, 0.0, 0.0, 0.0)
        ref_root.position = (0.0, 0.0, 0.0)
        loaded_scene = ViserMujocoScene(server, loaded_model, num_envs=1)
        loaded_scene.create_visualization_gui()
        loaded_scene.server.scene.set_up_direction("+z")
        model = loaded_model
        data = loaded_data
        default_qpos = loaded_data.qpos.copy()
        scene = loaded_scene
        loaded_scene_path = snapshot_path
        robot_qpos_addr = robot_qpos_addr_val
        robot_qvel_addr = robot_qvel_addr_val
        dof_qpos_addrs = dof_qpos_addrs_val
        dof_qvel_addrs = dof_qvel_addrs_val
        actor_layouts = actor_layouts_val
        logger.info("Loaded mjviser MuJoCo scene from {}", snapshot_path)
        return True

    @show_ref_cb.on_update
    def _(_evt) -> None:
        ref_root.visible = bool(show_ref_cb.value)

    @reset_rollout_btn.on_click
    def _(_evt) -> None:
        _request_sim_reset("gui_reset")

    @default_pose_init_cb.on_update
    def _(_evt) -> None:
        nonlocal pending_restart_reason
        if not cfg.launch_rollout:
            _refresh_rollout_md()
            return
        pending_restart_reason = "default_pose_init_toggle"
        state_md.content = "Restarting rollout to apply default-pose init preference..."
        _refresh_rollout_md()

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

    logger.info("Open mjviser at http://localhost:{}", port)
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

            sim_time_ms = int(state.get("sim_time_ms", 0))
            if reset_pending_clock_rewind and pre_reset_sim_time_ms is not None and sim_time_ms >= pre_reset_sim_time_ms:
                time.sleep(1.0 / max(cfg.rate_hz, 1.0))
                continue

            robot_root_state = state.get("robot_root_state")
            robot_dof_pos = state.get("robot_dof_pos")
            robot_dof_vel = state.get("robot_dof_vel")
            if robot_root_state is None or robot_dof_pos is None or robot_dof_vel is None:
                time.sleep(1.0 / max(cfg.rate_hz, 1.0))
                continue

            snapshot_path = scene_xml_snapshot_path_default
            snapshot_path_raw = state.get("mujoco_scene_xml_snapshot_path")
            if isinstance(snapshot_path_raw, str) and snapshot_path_raw.strip():
                snapshot_path = Path(snapshot_path_raw).expanduser().resolve()
            if scene is None and snapshot_path.is_file():
                _load_scene_from_snapshot(snapshot_path, state)
            if scene is None or model is None or data is None or default_qpos is None:
                time.sleep(1.0 / max(cfg.rate_hz, 1.0))
                continue

            root_state = np.asarray(robot_root_state, dtype=np.float32).reshape(-1)
            dof_pos = np.asarray(robot_dof_pos, dtype=np.float32).reshape(-1)
            dof_vel = np.asarray(robot_dof_vel, dtype=np.float32).reshape(-1)
            if root_state.shape[0] < 13 or robot_qpos_addr is None or robot_qvel_addr is None:
                time.sleep(1.0 / max(cfg.rate_hz, 1.0))
                continue

            if not received_first_state:
                if reset_request_time_monotonic is None:
                    logger.info(
                        "Received first sim-state: sim_time_ms={}, ref_body={}",
                        sim_time_ms,
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

            data.qpos[:] = default_qpos
            data.qvel[:] = 0.0
            _apply_freejoint_state(data, robot_qpos_addr, robot_qvel_addr, root_state[:13])
            if dof_qpos_addrs is not None and dof_qvel_addrs is not None:
                qpos_count = min(len(dof_qpos_addrs), dof_pos.shape[0])
                qvel_count = min(len(dof_qvel_addrs), dof_vel.shape[0])
                data.qpos[dof_qpos_addrs[:qpos_count]] = dof_pos[:qpos_count]
                data.qvel[dof_qvel_addrs[:qvel_count]] = dof_vel[:qvel_count]

            actors = state.get("actors")
            if isinstance(actors, dict):
                for actor_name, actor_state_raw in actors.items():
                    actor_layout = actor_layouts.get(str(actor_name))
                    if actor_layout is None:
                        continue
                    actor_state = np.asarray(actor_state_raw, dtype=np.float32).reshape(-1)
                    if actor_state.shape[0] < 13:
                        continue
                    _apply_freejoint_state(data, actor_layout[0], actor_layout[1], actor_state[:13])

            mujoco.mj_forward(model, data)
            scene.update_from_mjdata(data)

            ref_state = state.get("robot_ref_state")
            if ref_state is not None:
                ref_state_np = np.asarray(ref_state, dtype=np.float32).reshape(-1)
                if ref_state_np.shape[0] >= 7:
                    ref_root.position = tuple(ref_state_np[:3].tolist())
                    ref_root.wxyz = tuple(_xyzw_to_wxyz(ref_state_np[3:7]).tolist())
                    ref_root.visible = bool(show_ref_cb.value)
            else:
                ref_root.visible = False

            ref_body_name = state.get("robot_ref_body_name", "n/a")
            object_robot_contacts = int(state.get("object_robot_contact_count", 0))
            object_scene_contacts = int(state.get("object_scene_contact_count", 0))
            state_md.content = (
                f"sim_time_ms: `{sim_time_ms}`\n\n"
                f"ref_body: `{ref_body_name}`\n\n"
                f"robot_root_xyz: `{np.array2string(root_state[:3], precision=4)}`\n\n"
                f"object_robot_contacts: `{object_robot_contacts}`\n\n"
                f"object_scene_contacts: `{object_scene_contacts}`\n\n"
                f"scene_xml: `{loaded_scene_path}`"
            )
            actor_key, object_state = _select_actor_state(state, cfg.object_actor_name)
            actor_label = actor_key if actor_key is not None else "none"
            object_xyz_label = "n/a"
            if object_state is not None:
                object_xyz_label = np.array2string(object_state[:3], precision=4)
            actor_md.content = f"object_actor: `{actor_label}`\n\nobject_xyz: `{object_xyz_label}`"

            time.sleep(1.0 / max(cfg.rate_hz, 1.0))
    except KeyboardInterrupt:
        logger.info("Stopping mjviser MuJoCo sim-state viewer")
    finally:
        _stop_rollout()
        policy_control_pub.close()
        control_pub.close()
        sub.close()
        server.stop()
        signal.signal(signal.SIGTERM, previous_sigterm_handler)


def main() -> None:
    cfg = tyro.cli(MjviserMujocoSimStateViewerConfig)
    view_sim_state(cfg)


if __name__ == "__main__":
    main()
