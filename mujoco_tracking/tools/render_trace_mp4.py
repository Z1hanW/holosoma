#!/usr/bin/env python3
"""Render a MuJoCo physics trace to MP4 from a co-tracking camera."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import mujoco
import numpy as np


def _encode_h264(raw_mp4: Path, output: Path) -> None:
    """Convert OpenCV's mp4v output into broadly compatible H.264 MP4."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raw_mp4.replace(output)
        return

    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(raw_mp4),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-preset",
            "medium",
            "-crf",
            "20",
            str(output),
        ],
        check=True,
    )
    raw_mp4.unlink(missing_ok=True)


def _quat_xyzw_to_wxyz(q: list[float] | np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def _find_joint(model: mujoco.MjModel, names: tuple[str, ...]) -> int:
    for name in names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id != -1:
            return int(joint_id)
    raise RuntimeError(f"none of these joints exist in model: {names}")


def _joint_qpos_addr(model: mujoco.MjModel, name: str) -> int | None:
    for candidate in (name, f"robot_{name}"):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, candidate)
        if joint_id != -1:
            return int(model.jnt_qposadr[joint_id])
    return None


def _joint_qvel_addr(model: mujoco.MjModel, name: str) -> int | None:
    for candidate in (name, f"robot_{name}"):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, candidate)
        if joint_id != -1:
            return int(model.jnt_dofadr[joint_id])
    return None


def _set_freejoint(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_id: int,
    state: list[float],
) -> None:
    qpos_addr = int(model.jnt_qposadr[joint_id])
    qvel_addr = int(model.jnt_dofadr[joint_id])
    data.qpos[qpos_addr : qpos_addr + 3] = np.asarray(state[:3], dtype=np.float64)
    data.qpos[qpos_addr + 3 : qpos_addr + 7] = _quat_xyzw_to_wxyz(state[3:7])
    if len(state) >= 13:
        data.qvel[qvel_addr : qvel_addr + 3] = np.asarray(state[7:10], dtype=np.float64)
        data.qvel[qvel_addr + 3 : qvel_addr + 6] = np.asarray(state[10:13], dtype=np.float64)


def _object_state(row: dict) -> list[float] | None:
    actors = row.get("actors") or {}
    state = actors.get("object")
    if isinstance(state, list):
        return state
    for value in actors.values():
        if isinstance(value, list):
            return value
    return None


def _object_body_ids(model: mujoco.MjModel) -> list[int]:
    ids = []
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if name.startswith("object"):
            ids.append(body_id)
    return ids


def _update_camera(cam: mujoco.MjvCamera, root: np.ndarray, obj: np.ndarray | None) -> None:
    if obj is None:
        target = root.copy()
        separation = 0.0
    else:
        target = 0.5 * (root + obj)
        separation = float(np.linalg.norm(root[:2] - obj[:2]))
    target[2] = max(float(target[2]), 0.65)
    cam.lookat[:] = target
    cam.distance = max(3.0, 2.4 + 0.8 * separation)
    cam.azimuth = 135.0
    cam.elevation = -18.0


def render_trace(
    model_xml: Path,
    trace_path: Path,
    output: Path,
    width: int,
    height: int,
    fps: float,
    stride: int,
) -> int:
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=width, height=height)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)

    robot_joint = _find_joint(model, ("robot_floating_base_joint", "robot_freejoint", "floating_base_joint"))
    object_joint = _find_joint(model, ("object_freejoint", "object/freejoint", "freejoint"))
    object_body_ids = _object_body_ids(model)

    rows = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"empty trace: {trace_path}")

    output.parent.mkdir(parents=True, exist_ok=True)
    raw_output = output.with_name(f"{output.stem}_raw_mp4v{output.suffix}")
    raw_output.unlink(missing_ok=True)
    writer = cv2.VideoWriter(
        str(raw_output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open mp4 writer: {output}")

    frame_count = 0
    try:
        for index, row in enumerate(rows):
            if index % stride != 0:
                continue

            root_state = row.get("robot_root_state")
            if not isinstance(root_state, list) or len(root_state) < 7:
                continue
            _set_freejoint(model, data, robot_joint, root_state)

            dof_names = row.get("robot_dof_names") or []
            dof_pos = row.get("robot_dof_pos") or []
            dof_vel = row.get("robot_dof_vel") or []
            for name, value in zip(dof_names, dof_pos):
                addr = _joint_qpos_addr(model, str(name))
                if addr is not None:
                    data.qpos[addr] = float(value)
            for name, value in zip(dof_names, dof_vel):
                addr = _joint_qvel_addr(model, str(name))
                if addr is not None:
                    data.qvel[addr] = float(value)

            obj_state = _object_state(row)
            if obj_state is not None and len(obj_state) >= 7:
                _set_freejoint(model, data, object_joint, obj_state)

            mujoco.mj_forward(model, data)

            root_pos = np.asarray(root_state[:3], dtype=np.float64)
            if object_body_ids:
                obj_pos = np.mean(data.xpos[object_body_ids], axis=0)
            elif obj_state is not None:
                obj_pos = np.asarray(obj_state[:3], dtype=np.float64)
            else:
                obj_pos = None
            _update_camera(cam, root_pos, obj_pos)

            renderer.update_scene(data, camera=cam)
            frame = renderer.render()
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            frame_count += 1
    finally:
        writer.release()
        renderer.close()

    if frame_count == 0:
        raise RuntimeError("no frames rendered")
    _encode_h264(raw_output, output)
    return frame_count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-xml", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--stride", type=int, default=7)
    args = parser.parse_args()

    frames = render_trace(
        model_xml=args.model_xml,
        trace_path=args.trace,
        output=args.output,
        width=args.width,
        height=args.height,
        fps=args.fps,
        stride=max(1, args.stride),
    )
    print(f"wrote {args.output} ({frames} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
