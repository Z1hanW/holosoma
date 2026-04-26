"""Record a MuJoCo split rollout debug video from sim_state and depth shm.

The recorder is intentionally independent of the browser/viser frontend.  It
starts a rollout through the command web API, subscribes to the simulator state,
reads the policy perception_obs shared-memory buffer, and renders a compact
diagnostic MP4 with side/top views plus the depth panel.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from multiprocessing import resource_tracker
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import zmq
from loguru import logger


def _json_post(url: str, payload: dict[str, Any], *, timeout: float = 3.0) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _json_get(url: str, *, timeout: float = 1.0) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def _safe_json_post(url: str, payload: dict[str, Any], *, timeout: float = 3.0) -> dict[str, Any] | None:
    try:
        return _json_post(url, payload, timeout=timeout)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def _wait_for_web_state(base_url: str, *, timeout_s: float, poll_s: float = 0.1) -> dict[str, Any] | None:
    deadline = time.monotonic() + max(float(timeout_s), 0.0)
    last_state: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        state = _json_get(f"{base_url}/state")
        if state is not None:
            last_state = state
            return state
        time.sleep(max(float(poll_s), 0.01))
    return last_state


def _wait_for_clip_active(base_url: str, *, timeout_s: float, poll_s: float = 0.1) -> dict[str, Any] | None:
    deadline = time.monotonic() + max(float(timeout_s), 0.0)
    last_state: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        state = _json_get(f"{base_url}/state")
        if state is not None:
            last_state = state
            if bool(state.get("clip_active", False)):
                return state
        time.sleep(max(float(poll_s), 0.01))
    return last_state


def _prepare_rollout_start(
    base_url: str,
    *,
    reset_reason: str,
    reset_to_default_pose: bool | None = None,
    timeout_s: float = 45.0,
    settle_s: float = 0.8,
) -> dict[str, Any]:
    """Reset and start only after the policy control path is actually ready."""

    initial_state = _wait_for_web_state(base_url, timeout_s=timeout_s)
    if initial_state is None:
        raise RuntimeError(f"Timed out waiting for command web state at {base_url}/state")
    if reset_to_default_pose is None:
        reset_mode = (
            os.environ.get("MJ_ENV_RECORD_MOTION_INIT_MODE", "")
            or os.environ.get("SIM_MOTION_INIT_MODE", "")
        ).strip().lower().replace("-", "_")
        if reset_mode in {"raw_motion", "training_default_pose"}:
            reset_to_default_pose = reset_mode == "training_default_pose"
        else:
            reset_to_default_pose = bool(initial_state.get("reset_to_default_pose", False))

    _safe_json_post(
        f"{base_url}/command",
        {"enabled": False, "keys": [], "reset_to_default_pose": bool(reset_to_default_pose), "mode": "manual"},
    )
    _safe_json_post(f"{base_url}/policy", {"action": "stop"})
    reset_response = _safe_json_post(
        f"{base_url}/reset",
        {"reason": reset_reason, "reset_to_default_pose": bool(reset_to_default_pose)},
    )
    if reset_response is None:
        raise RuntimeError("Failed to request simulator reset through command web")
    time.sleep(max(float(settle_s), 0.0))

    deadline = time.monotonic() + max(float(timeout_s), 0.0)
    last_response: dict[str, Any] | None = None
    last_state: dict[str, Any] | None = None
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        last_response = _safe_json_post(f"{base_url}/policy", {"action": "rollout_start"})
        if last_response is not None and bool(last_response.get("sent", False)):
            last_state = _wait_for_clip_active(base_url, timeout_s=1.5)
            if last_state is not None and bool(last_state.get("clip_active", False)):
                logger.info("Rollout start confirmed after {} attempt(s): {}", attempt, last_state)
                return last_state
        else:
            last_state = _json_get(f"{base_url}/state") or last_state
        time.sleep(0.35)

    raise RuntimeError(
        "Timed out waiting for rollout_start to reach an active motion clip "
        f"(last_response={last_response}, last_state={last_state})"
    )


class SimStateSub:
    def __init__(self, port: int) -> None:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.connect(f"tcp://localhost:{int(port)}")
        self.last_state: dict[str, Any] | None = None

    def get(self) -> dict[str, Any] | None:
        while True:
            try:
                self.last_state = json.loads(self.socket.recv_string(zmq.NOBLOCK))
            except zmq.Again:
                break
            except json.JSONDecodeError:
                continue
        return self.last_state

    def close(self) -> None:
        self.socket.close(0)
        self.context.term()


class DepthShmReader:
    def __init__(self, name: str, expected_dim: int) -> None:
        self.name = str(name)
        self.expected_dim = int(expected_dim)
        self.shm: shared_memory.SharedMemory | None = None
        self.array: np.ndarray | None = None

    def _attach(self) -> bool:
        if self.shm is not None and self.array is not None:
            return True
        try:
            self.shm = shared_memory.SharedMemory(name=self.name, create=False)
        except FileNotFoundError:
            return False
        try:
            resource_tracker.unregister(self.shm._name, "shared_memory")
        except Exception:
            pass
        expected_bytes = self.expected_dim * np.dtype(np.float32).itemsize
        if len(self.shm.buf) < expected_bytes:
            self.shm.close()
            self.shm = None
            return False
        self.array = np.ndarray((self.expected_dim,), dtype=np.float32, buffer=self.shm.buf)
        return True

    def read(self) -> np.ndarray | None:
        if not self._attach() or self.array is None:
            return None
        return self.array.copy()

    def close(self) -> None:
        if self.shm is not None:
            self.shm.close()
        self.shm = None
        self.array = None


def _blank(width: int, height: int) -> np.ndarray:
    return np.full((height, width, 3), 245, dtype=np.uint8)


def _put(img: np.ndarray, text: str, xy: tuple[int, int], *, scale: float = 0.45, color=(30, 30, 30)) -> None:
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _project(value: tuple[float, float], bounds: tuple[float, float, float, float], rect: tuple[int, int, int, int]) -> tuple[int, int]:
    xmin, xmax, ymin, ymax = bounds
    x0, y0, w, h = rect
    x, y = value
    sx = (x - xmin) / max(1.0e-6, xmax - xmin)
    sy = (y - ymin) / max(1.0e-6, ymax - ymin)
    px = int(x0 + np.clip(sx, 0.0, 1.0) * w)
    py = int(y0 + (1.0 - np.clip(sy, 0.0, 1.0)) * h)
    return px, py


def _state_pos(state: dict[str, Any], name: str) -> np.ndarray | None:
    if name == "root":
        values = state.get("robot_root_state") or []
    elif name == "object":
        values = (state.get("actors") or {}).get("object") or []
    else:
        values = (state.get("key_body_states") or {}).get(name) or []
    if len(values) < 3:
        return None
    return np.asarray(values[:3], dtype=np.float64)


def _as_float_array(values: Any) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=np.float64)
    try:
        return np.asarray(values, dtype=np.float64).reshape(-1)
    except Exception:
        return np.asarray([], dtype=np.float64)


def _xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    if quat_xyzw.size < 4:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return np.asarray([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)


class MujocoStateRenderer:
    """Render published sim_state by replaying qpos into an exported MuJoCo model."""

    def __init__(self, xml_path: str | Path, *, width: int, height: int) -> None:
        if "MUJOCO_GL" not in os.environ and "DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ:
            os.environ["MUJOCO_GL"] = "egl"
        import mujoco  # Imported after MUJOCO_GL is selected.

        self.mujoco = mujoco
        self.path = self._resolve_mesh_paths(Path(xml_path).expanduser())
        self.width = int(width)
        self.height = int(height)
        self.model = mujoco.MjModel.from_xml_path(str(self.path))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.azimuth = 135.0
        self.camera.elevation = -18.0
        self.camera.distance = 3.0
        self._robot_root_qpos_addr = self._find_robot_freejoint_qpos_addr()
        self._robot_root_qvel_addr = self._find_robot_freejoint_qvel_addr()
        self._object_qpos_addr = self._find_object_freejoint_qpos_addr()
        self._object_qvel_addr = self._find_object_freejoint_qvel_addr()
        self._cached_dof_key: tuple[str, ...] | None = None
        self._cached_dof_addrs: list[tuple[int, int]] = []
        logger.info(
            "Loaded MuJoCo render model '{}' (nq={}, nv={}, joints={})",
            self.path,
            self.model.nq,
            self.model.nv,
            self.model.njnt,
        )

    @staticmethod
    def _resolve_mesh_paths(xml_path: Path) -> Path:
        repo_root = Path(__file__).resolve().parents[3]
        robot_roots = [
            repo_root / "src" / "holosoma" / "holosoma" / "data" / "robots" / "g1",
            repo_root / "src" / "holosoma" / "holosoma" / "data" / "robots" / "t1",
        ]
        tree = ET.parse(xml_path)
        root = tree.getroot()
        changed = False
        for mesh in root.findall(".//mesh"):
            file_attr = mesh.get("file")
            if not file_attr:
                continue
            mesh_path = Path(file_attr)
            if mesh_path.is_absolute() or (xml_path.parent / mesh_path).exists():
                continue
            candidates = [repo_root / mesh_path]
            candidates.extend(robot_root / mesh_path for robot_root in robot_roots)
            resolved = next((candidate.resolve() for candidate in candidates if candidate.exists()), None)
            if resolved is None:
                continue
            mesh.set("file", str(resolved))
            changed = True
        if not changed:
            return xml_path
        resolved_xml = xml_path.with_name(f"{xml_path.stem}_resolved{xml_path.suffix}")
        tree.write(resolved_xml, encoding="utf-8", xml_declaration=False)
        logger.info("Wrote resolved MuJoCo render XML to {}", resolved_xml)
        return resolved_xml

    def close(self) -> None:
        self.renderer.close()

    def render(self, state: dict[str, Any] | None) -> np.ndarray:
        if state is None:
            frame = _blank(self.width, self.height)
            _put(frame, "waiting for sim_state", (24, self.height // 2), scale=0.75, color=(80, 80, 80))
            return frame
        self._apply_state(state)
        self._update_camera(state)
        self.renderer.update_scene(self.data, camera=self.camera)
        rgb = self.renderer.render()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _joint_name(self, joint_id: int) -> str:
        name = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, int(joint_id))
        return name or ""

    def _body_name(self, body_id: int) -> str:
        name = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_BODY, int(body_id))
        return name or ""

    def _is_free_joint(self, joint_id: int) -> bool:
        return int(self.model.jnt_type[int(joint_id)]) == int(self.mujoco.mjtJoint.mjJNT_FREE)

    def _freejoint_ids(self) -> list[int]:
        return [joint_id for joint_id in range(int(self.model.njnt)) if self._is_free_joint(joint_id)]

    def _find_robot_freejoint_qpos_addr(self) -> int | None:
        for joint_id in self._freejoint_ids():
            name = self._joint_name(joint_id)
            body_name = self._body_name(int(self.model.jnt_bodyid[joint_id]))
            if "floating_base_joint" in name or body_name in {"robot_pelvis", "pelvis", "pelvis_link", "base_link"}:
                return int(self.model.jnt_qposadr[joint_id])
        for joint_id in self._freejoint_ids():
            name = self._joint_name(joint_id).lower()
            body_name = self._body_name(int(self.model.jnt_bodyid[joint_id])).lower()
            if "object" not in name and "object" not in body_name:
                return int(self.model.jnt_qposadr[joint_id])
        return None

    def _find_robot_freejoint_qvel_addr(self) -> int | None:
        qpos_addr = self._robot_root_qpos_addr
        if qpos_addr is None:
            return None
        for joint_id in self._freejoint_ids():
            if int(self.model.jnt_qposadr[joint_id]) == qpos_addr:
                return int(self.model.jnt_dofadr[joint_id])
        return None

    def _find_object_freejoint_qpos_addr(self) -> int | None:
        for joint_id in self._freejoint_ids():
            name = self._joint_name(joint_id).lower()
            body_name = self._body_name(int(self.model.jnt_bodyid[joint_id])).lower()
            if "object" in name or "object" in body_name:
                return int(self.model.jnt_qposadr[joint_id])
        for joint_id in self._freejoint_ids():
            addr = int(self.model.jnt_qposadr[joint_id])
            if self._robot_root_qpos_addr is None or addr != self._robot_root_qpos_addr:
                return addr
        return None

    def _find_object_freejoint_qvel_addr(self) -> int | None:
        qpos_addr = self._object_qpos_addr
        if qpos_addr is None:
            return None
        for joint_id in self._freejoint_ids():
            if int(self.model.jnt_qposadr[joint_id]) == qpos_addr:
                return int(self.model.jnt_dofadr[joint_id])
        return None

    def _find_joint_id_for_dof(self, dof_name: str) -> int | None:
        candidates = [
            dof_name,
            f"robot_{dof_name}",
            f"robot/{dof_name}",
            f"robot:{dof_name}",
            "robot_" + dof_name.replace("/", "_"),
        ]
        for candidate in candidates:
            joint_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, candidate)
            if joint_id >= 0:
                return int(joint_id)
        for joint_id in range(int(self.model.njnt)):
            if self._joint_name(joint_id).endswith(dof_name):
                return int(joint_id)
        return None

    def _fallback_dof_addrs(self, expected: int) -> list[tuple[int, int]]:
        addrs: list[tuple[int, int]] = []
        for joint_id in range(int(self.model.njnt)):
            if self._is_free_joint(joint_id):
                continue
            name = self._joint_name(joint_id).lower()
            body_name = self._body_name(int(self.model.jnt_bodyid[joint_id])).lower()
            if "object" in name or "object" in body_name:
                continue
            addrs.append((int(self.model.jnt_qposadr[joint_id]), int(self.model.jnt_dofadr[joint_id])))
        return addrs[:expected]

    def _dof_addrs(self, state: dict[str, Any], expected: int) -> list[tuple[int, int]]:
        names = state.get("robot_dof_names")
        key = tuple(str(name) for name in names) if isinstance(names, list) else tuple()
        if self._cached_dof_key == key and self._cached_dof_addrs:
            return self._cached_dof_addrs[:expected]
        addrs: list[tuple[int, int]] = []
        if key:
            for dof_name in key:
                joint_id = self._find_joint_id_for_dof(dof_name)
                if joint_id is None:
                    continue
                addrs.append((int(self.model.jnt_qposadr[joint_id]), int(self.model.jnt_dofadr[joint_id])))
        if len(addrs) != expected:
            if key:
                logger.warning(
                    "Could only resolve {}/{} robot DOF names in render model; falling back to model joint order",
                    len(addrs),
                    expected,
                )
            addrs = self._fallback_dof_addrs(expected)
        self._cached_dof_key = key
        self._cached_dof_addrs = addrs
        return addrs[:expected]

    def _apply_free_state(self, qpos_addr: int | None, qvel_addr: int | None, values: np.ndarray) -> None:
        if qpos_addr is None or values.size < 7:
            return
        self.data.qpos[qpos_addr : qpos_addr + 3] = values[:3]
        self.data.qpos[qpos_addr + 3 : qpos_addr + 7] = _xyzw_to_wxyz(values[3:7])
        if qvel_addr is not None and values.size >= 13:
            self.data.qvel[qvel_addr : qvel_addr + 3] = values[7:10]
            self.data.qvel[qvel_addr + 3 : qvel_addr + 6] = values[10:13]

    def _object_state(self, state: dict[str, Any]) -> np.ndarray:
        actors = state.get("actors")
        if not isinstance(actors, dict) or not actors:
            return np.asarray([], dtype=np.float64)
        values = actors.get("object")
        if values is None and len(actors) == 1:
            values = next(iter(actors.values()))
        return _as_float_array(values)

    def _apply_state(self, state: dict[str, Any]) -> None:
        root_state = _as_float_array(state.get("robot_root_state"))
        self._apply_free_state(self._robot_root_qpos_addr, self._robot_root_qvel_addr, root_state)

        dof_pos = _as_float_array(state.get("robot_dof_pos"))
        dof_vel = _as_float_array(state.get("robot_dof_vel"))
        dof_addrs = self._dof_addrs(state, int(dof_pos.size))
        for i, (qpos_addr, qvel_addr) in enumerate(dof_addrs):
            if i < dof_pos.size:
                self.data.qpos[qpos_addr] = dof_pos[i]
            if i < dof_vel.size:
                self.data.qvel[qvel_addr] = dof_vel[i]

        self._apply_free_state(self._object_qpos_addr, self._object_qvel_addr, self._object_state(state))
        self.mujoco.mj_forward(self.model, self.data)

    def _update_camera(self, state: dict[str, Any]) -> None:
        root = _state_pos(state, "root")
        obj = _state_pos(state, "object")
        if root is None and obj is None:
            return
        if root is not None and obj is not None:
            look = 0.65 * root + 0.35 * obj
            span = float(np.linalg.norm((root - obj)[:2]))
        else:
            look = root if root is not None else obj
            span = 0.0
        assert look is not None
        self.camera.lookat[:] = [float(look[0]), float(look[1]), max(0.35, float(look[2]) * 0.55)]
        self.camera.distance = float(np.clip(2.4 + span, 2.4, 4.5))


def _draw_view(
    img: np.ndarray,
    state: dict[str, Any] | None,
    *,
    rect: tuple[int, int, int, int],
    mode: str,
    title: str,
) -> None:
    x0, y0, w, h = rect
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (210, 210, 210), 1)
    _put(img, title, (x0 + 8, y0 + 20), scale=0.55)
    if state is None:
        _put(img, "waiting for sim_state", (x0 + 20, y0 + h // 2), color=(80, 80, 80))
        return

    root = _state_pos(state, "root")
    if root is None:
        return
    if mode == "side":
        bounds = (root[0] - 1.3, root[0] + 1.3, -0.05, 1.45)
        axis = lambda p: (float(p[0]), float(p[2]))
        cv2.line(img, _project((bounds[0], 0.0), bounds, rect), _project((bounds[1], 0.0), bounds, rect), (180, 180, 180), 1)
    else:
        bounds = (root[0] - 1.3, root[0] + 1.3, root[1] - 1.3, root[1] + 1.3)
        axis = lambda p: (float(p[0]), float(p[1]))

    body_names = [
        "torso_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "left_rubber_hand",
        "right_elbow_link",
        "right_wrist_yaw_link",
        "right_rubber_hand",
        "left_foot_contact_point",
        "right_foot_contact_point",
    ]
    points = {name: _state_pos(state, name) for name in body_names}
    edges = [
        ("torso_link", "left_elbow_link"),
        ("left_elbow_link", "left_wrist_yaw_link"),
        ("left_wrist_yaw_link", "left_rubber_hand"),
        ("torso_link", "right_elbow_link"),
        ("right_elbow_link", "right_wrist_yaw_link"),
        ("right_wrist_yaw_link", "right_rubber_hand"),
        ("torso_link", "left_foot_contact_point"),
        ("torso_link", "right_foot_contact_point"),
    ]
    for a, b in edges:
        pa = points.get(a)
        pb = points.get(b)
        if pa is None or pb is None:
            continue
        cv2.line(img, _project(axis(pa), bounds, rect), _project(axis(pb), bounds, rect), (70, 120, 210), 2)
    for name, pos in points.items():
        if pos is None:
            continue
        color = (40, 90, 210)
        if "rubber_hand" in name:
            color = (30, 150, 70)
        if "foot" in name:
            color = (90, 90, 90)
        cv2.circle(img, _project(axis(pos), bounds, rect), 5, color, -1)

    obj = _state_pos(state, "object")
    if obj is not None:
        obj_xy = _project(axis(obj), bounds, rect)
        cv2.circle(img, obj_xy, 10, (30, 70, 220), -1)
        cv2.circle(img, obj_xy, 14, (30, 70, 220), 2)


def _depth_panel(depth: np.ndarray | None, *, width: int, height: int, depth_shape: tuple[int, int]) -> np.ndarray:
    panel = _blank(width, height)
    _put(panel, "policy perception_obs", (8, 22), scale=0.55)
    if depth is None:
        _put(panel, "waiting for depth shm", (20, height // 2), color=(80, 80, 80))
        return panel
    try:
        depth_img = depth.reshape(depth_shape)
    except ValueError:
        _put(panel, f"depth dim mismatch: {depth.size}", (20, height // 2), color=(0, 0, 180))
        return panel
    finite = depth_img[np.isfinite(depth_img)]
    if finite.size == 0:
        norm = np.zeros(depth_shape, dtype=np.uint8)
        stats = "all non-finite"
    else:
        vmin = float(finite.min())
        vmax = float(finite.max())
        mean = float(finite.mean())
        if vmax - vmin < 1.0e-6:
            norm = np.zeros(depth_shape, dtype=np.uint8)
        else:
            norm = np.clip((depth_img - vmin) / (vmax - vmin) * 255.0, 0, 255).astype(np.uint8)
        stats = f"min={vmin:.3f} max={vmax:.3f} mean={mean:.3f}"
    color = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)
    color = cv2.resize(color, (width - 24, height - 70), interpolation=cv2.INTER_NEAREST)
    panel[40 : 40 + color.shape[0], 12 : 12 + color.shape[1]] = color
    _put(panel, stats, (12, height - 12), scale=0.45)
    return panel


def _render_frame(
    state: dict[str, Any] | None,
    web_state: dict[str, Any] | None,
    depth: np.ndarray | None,
    *,
    frame_no: int,
    elapsed_s: float,
    output_size: tuple[int, int],
    depth_shape: tuple[int, int],
    mujoco_renderer: MujocoStateRenderer | None = None,
) -> np.ndarray:
    width, height = output_size
    if mujoco_renderer is not None:
        render = mujoco_renderer.render(state)
        render_h, render_w = render.shape[:2]
        img = _blank(width, height)
        img[: min(height, render_h), : min(width, render_w)] = render[: min(height, render_h), : min(width, render_w)]
        cv2.rectangle(img, (0, 0), (min(render_w, width), 112), (0, 0, 0), -1)
        _put(img, "MuJoCo rendered rollout", (16, 28), scale=0.72, color=(245, 245, 245))
        _put(img, f"wall={elapsed_s:5.2f}s frame={frame_no}", (16, 54), scale=0.48, color=(230, 230, 230))
        if web_state:
            cmd = web_state.get("sparse_effective_command")
            src = web_state.get("sparse_command_source")
            timestep = web_state.get("motion_timestep")
            _put(
                img,
                f"policy: clip={web_state.get('clip_active')} source={src} motion_timestep={timestep}",
                (16, 76),
                scale=0.44,
                color=(230, 230, 230),
            )
            if isinstance(cmd, list):
                _put(
                    img,
                    "effective_cmd=[" + ", ".join(f"{float(v):.3f}" for v in cmd[:3]) + "]",
                    (16, 98),
                    scale=0.44,
                    color=(230, 230, 230),
                )
        if state:
            obj = _state_pos(state, "object")
            root = _state_pos(state, "root")
            contacts = state.get("object_robot_contact_count", 0)
            obj_z = float(obj[2]) if obj is not None else math.nan
            root_z = float(root[2]) if root is not None else math.nan
            _put(
                img,
                f"sim_ms={state.get('sim_time_ms')} root_z={root_z:.3f} obj_z={obj_z:.3f} contacts={contacts}",
                (render_w + 14, 28),
                scale=0.45,
                color=(30, 30, 30),
            )
            bodies = ",".join((state.get("object_robot_contact_bodies") or [])[:4])
            _put(img, f"contact_bodies={bodies or '<none>'}", (render_w + 14, 50), scale=0.42)
        panel_width = max(1, width - render_w)
        panel = _depth_panel(depth, width=panel_width, height=height - 90, depth_shape=depth_shape)
        img[90 : 90 + panel.shape[0], render_w : render_w + panel.shape[1]] = panel
        return img

    img = _blank(width, height)
    _put(img, "MuJoCo rollout debug recording", (16, 30), scale=0.75)
    _put(img, f"wall={elapsed_s:5.2f}s frame={frame_no}", (16, 56), scale=0.5)
    if web_state:
        cmd = web_state.get("sparse_effective_command")
        src = web_state.get("sparse_command_source")
        timestep = web_state.get("motion_timestep")
        _put(img, f"policy: clip={web_state.get('clip_active')} source={src} motion_timestep={timestep}", (16, 78))
        if isinstance(cmd, list):
            _put(img, "effective_cmd=[" + ", ".join(f"{float(v):.3f}" for v in cmd[:3]) + "]", (16, 100))
    if state:
        obj = _state_pos(state, "object")
        root = _state_pos(state, "root")
        contacts = state.get("object_robot_contact_count", 0)
        bodies = ",".join((state.get("object_robot_contact_bodies") or [])[:4])
        obj_z = float(obj[2]) if obj is not None else math.nan
        root_z = float(root[2]) if root is not None else math.nan
        _put(img, f"sim_ms={state.get('sim_time_ms')} root_z={root_z:.3f} obj_z={obj_z:.3f} contacts={contacts}", (16, 122))
        _put(img, f"contact_bodies={bodies or '<none>'}", (16, 144))

    _draw_view(img, state, rect=(20, 170, 600, 500), mode="side", title="side view x-z")
    _draw_view(img, state, rect=(650, 170, 360, 360), mode="top", title="top view x-y")
    panel = _depth_panel(depth, width=240, height=360, depth_shape=depth_shape)
    img[170 : 170 + panel.shape[0], 1020 : 1020 + panel.shape[1]] = panel
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web-port", type=int, default=4477)
    parser.add_argument("--state-port", type=int, default=5657)
    parser.add_argument("--depth-shm-name", default="depth_img_shm")
    parser.add_argument("--depth-height", type=int, default=58)
    parser.add_argument("--depth-width", type=int, default=87)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument(
        "--sim-duration",
        type=float,
        default=0.0,
        help="Optional simulator-time duration in seconds; wall duration remains the safety timeout.",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--output", default="")
    parser.add_argument("--mujoco-xml", default="", help="Exported MuJoCo XML to render instead of the schematic view.")
    parser.add_argument("--render-width", type=int, default=960, help="Width of the MuJoCo render region in the MP4.")
    parser.add_argument(
        "--replay-state-trace",
        default="",
        help="Render an existing sim_state JSONL trace offline instead of subscribing to the live simulator.",
    )
    parser.add_argument("--no-auto-start", action="store_true")
    parser.add_argument("--reset-reason", default="record_mujoco_rollout_video")
    args = parser.parse_args()

    out_path = Path(args.output).expanduser() if args.output else Path("logs/live_debug") / (
        "mujoco_rollout_" + time.strftime("%Y%m%d_%H%M%S") + ".mp4"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base_url = f"http://127.0.0.1:{int(args.web_port)}"
    size = (1280, 720)

    if args.replay_state_trace:
        trace_path = Path(args.replay_state_trace).expanduser()
        if not trace_path.is_file():
            raise FileNotFoundError(f"Replay state trace not found: {trace_path}")
        if not args.mujoco_xml:
            raise ValueError("--replay-state-trace requires --mujoco-xml for real MuJoCo rendering")

        states: list[dict[str, Any]] = []
        with trace_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    state = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(state, dict):
                    states.append(state)
        if not states:
            raise ValueError(f"Replay state trace has no valid states: {trace_path}")

        duration = float(args.duration)
        if duration <= 0.0:
            first_ms = float(states[0].get("sim_time_ms") or 0.0)
            last_ms = float(states[-1].get("sim_time_ms") or first_ms)
            duration = max(1.0 / max(float(args.fps), 1.0), (last_ms - first_ms) / 1000.0)
        frame_count = max(1, int(math.ceil(duration * float(args.fps))))
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(args.fps), size)
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {out_path}")

        mujoco_renderer = MujocoStateRenderer(
            args.mujoco_xml,
            width=max(320, min(int(args.render_width), size[0] - 160)),
            height=size[1],
        )
        metadata: list[dict[str, Any]] = []
        try:
            for frame_no in range(frame_count):
                elapsed = frame_no / max(float(args.fps), 1.0e-6)
                state_idx = min(len(states) - 1, int(round(frame_no * (len(states) - 1) / max(frame_count - 1, 1))))
                state = states[state_idx]
                frame = _render_frame(
                    state,
                    None,
                    None,
                    frame_no=frame_no,
                    elapsed_s=elapsed,
                    output_size=size,
                    depth_shape=(int(args.depth_height), int(args.depth_width)),
                    mujoco_renderer=mujoco_renderer,
                )
                writer.write(frame)
                obj = _state_pos(state, "object")
                root = _state_pos(state, "root")
                metadata.append(
                    {
                        "frame_no": frame_no,
                        "trace_index": state_idx,
                        "sim_time_ms": state.get("sim_time_ms"),
                        "root_z": None if root is None else float(root[2]),
                        "object_z": None if obj is None else float(obj[2]),
                        "contacts": int(state.get("object_robot_contact_count", 0)),
                    }
                )
        finally:
            writer.release()
            mujoco_renderer.close()

        meta_path = out_path.with_suffix(".json")
        meta_path.write_text(
            json.dumps({"video": str(out_path), "trace": str(trace_path), "samples": metadata}, indent=2),
            encoding="utf-8",
        )
        logger.info("Wrote offline MuJoCo rollout video: {}", out_path)
        logger.info("Wrote rollout metadata: {}", meta_path)
        return

    if not args.no_auto_start:
        web_state = _prepare_rollout_start(base_url, reset_reason=args.reset_reason)
    else:
        web_state = _json_get(f"{base_url}/state")

    sub = SimStateSub(args.state_port)
    depth_reader = DepthShmReader(args.depth_shm_name, int(args.depth_height) * int(args.depth_width))
    mujoco_renderer: MujocoStateRenderer | None = None
    if args.mujoco_xml:
        mujoco_renderer = MujocoStateRenderer(
            args.mujoco_xml,
            width=max(320, min(int(args.render_width), size[0] - 160)),
            height=size[1],
        )
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(args.fps), size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_path}")

    metadata: list[dict[str, Any]] = []
    start = time.monotonic()
    start_sim_ms: float | None = None
    sim_duration_s = float(args.sim_duration)
    frame_period = 1.0 / max(float(args.fps), 1.0e-6)
    next_frame = start
    frame_no = 0
    next_web = start
    try:
        while True:
            now = time.monotonic()
            elapsed = now - start
            if elapsed >= float(args.duration):
                break
            state = sub.get()
            if state is not None and sim_duration_s > 0.0:
                try:
                    sim_ms = float(state.get("sim_time_ms"))
                except (TypeError, ValueError):
                    sim_ms = math.nan
                if math.isfinite(sim_ms):
                    if start_sim_ms is None:
                        start_sim_ms = sim_ms
                    elif (sim_ms - start_sim_ms) >= sim_duration_s * 1000.0 and frame_no > 0:
                        break
            if now >= next_web:
                web_state = _json_get(f"{base_url}/state") or web_state
                next_web = now + 0.25
            if now < next_frame:
                time.sleep(min(0.005, next_frame - now))
                continue
            depth = depth_reader.read()
            frame = _render_frame(
                state,
                web_state,
                depth,
                frame_no=frame_no,
                elapsed_s=elapsed,
                output_size=size,
                depth_shape=(int(args.depth_height), int(args.depth_width)),
                mujoco_renderer=mujoco_renderer,
            )
            writer.write(frame)
            if state is not None:
                obj = _state_pos(state, "object")
                root = _state_pos(state, "root")
                metadata.append(
                    {
                        "frame_no": frame_no,
                        "wall_s": elapsed,
                        "sim_time_ms": state.get("sim_time_ms"),
                        "root_z": None if root is None else float(root[2]),
                        "object_z": None if obj is None else float(obj[2]),
                        "contacts": int(state.get("object_robot_contact_count", 0)),
                    }
                )
            frame_no += 1
            next_frame += frame_period
    finally:
        writer.release()
        if mujoco_renderer is not None:
            mujoco_renderer.close()
        sub.close()
        depth_reader.close()

    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "video": str(out_path),
                "wall_duration_limit_s": float(args.duration),
                "sim_duration_target_s": sim_duration_s,
                "samples": metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote rollout video: {}", out_path)
    logger.info("Wrote rollout metadata: {}", meta_path)


if __name__ == "__main__":
    main()
