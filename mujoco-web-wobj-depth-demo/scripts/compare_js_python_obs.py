#!/usr/bin/env python3
"""Compare browser-built MuJoCo-WASM observations against a Python reference."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


DEMO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = DEMO_ROOT / "public/demo-assets/manifest.json"


def resolve_default_config_path(config_path: Path | None) -> Path:
    if config_path is not None:
        return config_path
    manifest = json.loads(DEFAULT_MANIFEST.read_text())
    default_clip_id = manifest.get("default_clip_id")
    clips = manifest.get("clips", [])
    for clip in clips:
        if clip.get("id") == default_clip_id:
            return DEMO_ROOT / "public/demo-assets" / clip["config_path"]
    if clips:
        return DEMO_ROOT / "public/demo-assets" / clips[0]["config_path"]
    raise RuntimeError(f"No clips found in {DEFAULT_MANIFEST}")


NODE_SNAPSHOT_SCRIPT = r"""
import { chromium } from 'playwright';

const url = process.argv[2];
const steps = Number(process.argv[3] || 0);
const timeoutMs = Number(process.argv[4] || 60000);

const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  await page.goto(url, { waitUntil: 'networkidle', timeout: timeoutMs });
  await page.waitForFunction(() => window.__boxDepthTrackApp?.ready === true, null, { timeout: timeoutMs });
  const snapshot = await page.evaluate(async ({ steps }) => {
    const app = window.__boxDepthTrackApp;
    app.reset();
    app.history.clear();
    app.lastPolicyAction.fill(0.0);

    if (steps > 0) {
      app.policyActive = true;
      app.paused = false;
      for (let i = 0; i < steps; i += 1) {
        app.syncTargetToMotionReference();
        await app.stepPolicy();
      }
    }

    app.syncTargetToMotionReference();
    const obs = Array.from(app.buildObs());
    app.captureDepthObservation();
    const termBuffers = {};
    for (const [name, values] of Object.entries(app.buildTermBuffers())) {
      termBuffers[name] = Array.from(values);
    }
    const history = {};
    for (const [key, values] of app.history.entries()) {
      history[key] = values.map((item) => Array.from(item));
    }

    return {
      status: document.querySelector('#status')?.textContent || '',
      motionTimestep: app.motionTimestep,
      policyActive: app.policyActive,
      qpos: Array.from(app.data.qpos),
      qvel: Array.from(app.data.qvel),
      rootJoint: {
        qposAdr: app.rootJoint.qposAdr,
        qvelAdr: app.rootJoint.qvelAdr
      },
      jointBindings: app.jointBindings.map((binding) => ({
        index: binding.index,
        name: binding.name,
        qposAdr: binding.qposAdr,
        qvelAdr: binding.qvelAdr
      })),
      target: {
        x: app.target.x,
        y: app.target.y,
        yaw: app.target.yaw
      },
      lastPolicyAction: Array.from(app.lastPolicyAction),
      obs,
      termBuffers,
      history,
      depthObservation: Array.from(app.depthObservation),
      rawDepth: Array.from(app.rawDepth),
      rawCameraDepth: Array.from(app.rawCameraDepth),
      visibleMeshes: app.meshes.filter((mesh) => mesh?.visible).length,
      depthRaycastMeshes: app.depthRaycastMeshes.length
    };
  }, { steps });
  await page.close();
  console.log(JSON.stringify(snapshot));
} finally {
  await browser.close();
}
"""


def wrap_angle(value: float) -> float:
    angle = float(value)
    while angle > np.pi:
        angle -= 2.0 * np.pi
    while angle < -np.pi:
        angle += 2.0 * np.pi
    return angle


def yaw_from_quat_wxyz(q: np.ndarray) -> float:
    w, x, y, z = [float(v) for v in q]
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def yaw_quat_wxyz(yaw: float) -> np.ndarray:
    return np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)], dtype=np.float32)


def quat_apply_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(item) for item in q]
    tx = 2.0 * (y * float(v[2]) - z * float(v[1]))
    ty = 2.0 * (z * float(v[0]) - x * float(v[2]))
    tz = 2.0 * (x * float(v[1]) - y * float(v[0]))
    return np.array(
        [
            float(v[0]) + w * tx + (y * tz - z * ty),
            float(v[1]) + w * ty + (z * tx - x * tz),
            float(v[2]) + w * tz + (x * ty - y * tx),
        ],
        dtype=np.float32,
    )


def run_browser_snapshot(url: str, steps: int, timeout_ms: int) -> dict[str, Any]:
    result = subprocess.run(
        ["node", "--input-type=module", "-", url, str(steps), str(timeout_ms)],
        input=NODE_SNAPSHOT_SCRIPT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=DEMO_ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Browser snapshot failed. Make sure `npm run dev` is serving the demo.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return json.loads(result.stdout)


def build_python_terms(config: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, np.ndarray]:
    qpos = np.asarray(snapshot["qpos"], dtype=np.float32)
    qvel = np.asarray(snapshot["qvel"], dtype=np.float32)
    root = snapshot["rootJoint"]
    root_qpos_adr = int(root["qposAdr"])
    root_qvel_adr = int(root["qvelAdr"])
    root_pos = qpos[root_qpos_adr : root_qpos_adr + 3]
    root_quat = qpos[root_qpos_adr + 3 : root_qpos_adr + 7]
    root_yaw = yaw_from_quat_wxyz(root_quat)
    target = snapshot["target"]
    rel_world = np.array(
        [float(target["x"]) - float(root_pos[0]), float(target["y"]) - float(root_pos[1]), 0.0],
        dtype=np.float32,
    )
    rel_body = quat_apply_wxyz(yaw_quat_wxyz(-root_yaw), rel_world)
    sparse = np.array(
        [rel_body[0], rel_body[1], wrap_angle(float(target["yaw"]) - root_yaw)],
        dtype=np.float32,
    )

    dof_count = len(config["dof_names"])
    dof_pos = np.zeros(dof_count, dtype=np.float32)
    dof_vel = np.zeros(dof_count, dtype=np.float32)
    defaults = np.asarray(config["default_dof_angles"], dtype=np.float32)
    for binding in snapshot["jointBindings"]:
        idx = int(binding["index"])
        dof_pos[idx] = qpos[int(binding["qposAdr"])] - defaults[idx]
        dof_vel[idx] = qvel[int(binding["qvelAdr"])]

    return {
        "actions": np.asarray(snapshot["lastPolicyAction"], dtype=np.float32),
        "base_ang_vel": qvel[root_qvel_adr + 3 : root_qvel_adr + 6].astype(np.float32, copy=False),
        "dof_pos": dof_pos,
        "dof_vel": dof_vel,
        "sparse_target_root_trajectory_command": sparse,
    }


def obs_from_js_history(config: dict[str, Any], snapshot: dict[str, Any]) -> np.ndarray:
    history = {
        key: [np.asarray(item, dtype=np.float32) for item in values]
        for key, values in snapshot["history"].items()
    }
    chunks: list[np.ndarray] = []
    for group in config["observation"]["actor_groups"]:
        group_name = group["name"]
        history_len = int(group.get("history_length", 1))
        for term in group["terms"]:
            dim = int(term["dim"])
            key = f"{group_name}:{term['name']}"
            values = list(history.get(key, []))[-history_len:]
            missing = max(0, history_len - len(values))
            if missing:
                chunks.extend(np.zeros(dim, dtype=np.float32) for _ in range(missing))
            chunks.extend(item.astype(np.float32, copy=False).reshape(-1) for item in values)
    return np.concatenate(chunks).astype(np.float32, copy=False)


def resample_cropped_depth(
    raw: np.ndarray,
    raw_width: int,
    raw_height: int,
    out_width: int,
    out_height: int,
    perception: dict[str, Any],
) -> np.ndarray:
    crop_top = min(int(perception.get("camera_warp_crop_top", 0) or 0), max(0, raw_height - 1))
    crop_bottom = min(
        int(perception.get("camera_warp_crop_bottom", 0) or 0),
        max(0, raw_height - crop_top - 1),
    )
    crop_left = min(int(perception.get("camera_warp_crop_left", 0) or 0), max(0, raw_width - 1))
    crop_right = min(
        int(perception.get("camera_warp_crop_right", 0) or 0),
        max(0, raw_width - crop_left - 1),
    )
    crop_height = max(1, raw_height - crop_top - crop_bottom)
    crop_width = max(1, raw_width - crop_left - crop_right)
    raw = raw.reshape(raw_height, raw_width)
    out = np.zeros(out_width * out_height, dtype=np.float32)
    cursor = 0
    for y in range(out_height):
        source_y = np.clip((y + 0.5) * crop_height / out_height - 0.5, 0.0, crop_height - 1.0)
        y0 = int(np.floor(source_y))
        y1 = min(crop_height - 1, y0 + 1)
        wy = float(source_y - y0)
        for x in range(out_width):
            source_x = np.clip((x + 0.5) * crop_width / out_width - 0.5, 0.0, crop_width - 1.0)
            x0 = int(np.floor(source_x))
            x1 = min(crop_width - 1, x0 + 1)
            wx = float(source_x - x0)
            top = raw[crop_top + y0, crop_left + x0] * (1.0 - wx) + raw[crop_top + y0, crop_left + x1] * wx
            bottom = raw[crop_top + y1, crop_left + x0] * (1.0 - wx) + raw[crop_top + y1, crop_left + x1] * wx
            out[cursor] = top * (1.0 - wy) + bottom * wy
            cursor += 1
    return out


def depth_from_raw_camera(config: dict[str, Any], snapshot: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    perception = config["perception"]
    raw_width = int(perception.get("camera_width") or perception["observation_width"])
    raw_height = int(perception.get("camera_height") or perception["observation_height"])
    out_width = int(perception["observation_width"])
    out_height = int(perception["observation_height"])
    raw = np.asarray(snapshot["rawCameraDepth"], dtype=np.float32)
    depth = resample_cropped_depth(raw, raw_width, raw_height, out_width, out_height, perception)
    near = float(perception.get("camera_near", 0.0))
    max_distance = float(perception.get("max_distance", perception.get("camera_far", 3.0)))
    min_valid = float(perception.get("camera_warp_min_valid_depth", 0.0))
    depth = np.clip(depth, near, max_distance).astype(np.float32, copy=False)
    depth[depth < min_valid] = max_distance
    if perception.get("camera_warp_normalize", False):
        denom = max(1.0e-6, max_distance - near)
        obs = np.clip((depth - near) / denom - 0.5, -0.5, 0.5).astype(np.float32, copy=False)
    else:
        obs = depth.copy()
    return depth, obs


def compare_arrays(name: str, python_values: np.ndarray, js_values: np.ndarray, atol: float, rtol: float) -> bool:
    python_values = np.asarray(python_values, dtype=np.float32).reshape(-1)
    js_values = np.asarray(js_values, dtype=np.float32).reshape(-1)
    if python_values.shape != js_values.shape:
        print(f"FAIL {name}: shape python={python_values.shape} js={js_values.shape}")
        return False
    diff = np.abs(python_values - js_values)
    allowed = atol + rtol * np.abs(js_values)
    ok = bool(np.all(diff <= allowed))
    idx = int(np.argmax(diff)) if diff.size else 0
    status = "OK  " if ok else "FAIL"
    print(
        f"{status} {name:<48} max_abs={float(diff[idx]) if diff.size else 0.0:.8g} "
        f"idx={idx} py={float(python_values[idx]) if diff.size else 0.0:.8g} "
        f"js={float(js_values[idx]) if diff.size else 0.0:.8g}"
    )
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:4173/", help="running Vite demo URL")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--snapshot", type=Path, help="compare an existing browser snapshot JSON")
    parser.add_argument("--keep-snapshot", type=Path, help="write the browser snapshot JSON to this path")
    parser.add_argument("--steps", type=int, default=0, help="manual policy steps before taking the browser snapshot")
    parser.add_argument("--timeout-ms", type=int, default=60000)
    parser.add_argument("--atol", type=float, default=2.0e-6)
    parser.add_argument("--rtol", type=float, default=2.0e-6)
    args = parser.parse_args()

    config_path = resolve_default_config_path(args.config)
    config = json.loads(config_path.read_text())
    print(f"config: {config_path}")
    if args.snapshot:
        snapshot = json.loads(args.snapshot.read_text())
    else:
        snapshot = run_browser_snapshot(args.url, args.steps, args.timeout_ms)
        if args.keep_snapshot:
            args.keep_snapshot.write_text(json.dumps(snapshot, indent=2))

    print(
        "snapshot:",
        f"status={snapshot['status']}",
        f"motion_timestep={snapshot['motionTimestep']}",
        f"visible_meshes={snapshot['visibleMeshes']}",
        f"depth_raycast_meshes={snapshot['depthRaycastMeshes']}",
    )

    ok = True
    python_terms = build_python_terms(config, snapshot)
    for name, values in python_terms.items():
        ok &= compare_arrays(f"term.{name}", values, np.asarray(snapshot["termBuffers"][name]), args.atol, args.rtol)

    ok &= compare_arrays(
        "actor_obs.from_js_history",
        obs_from_js_history(config, snapshot),
        np.asarray(snapshot["obs"], dtype=np.float32),
        args.atol,
        args.rtol,
    )

    python_raw_depth, python_depth_obs = depth_from_raw_camera(config, snapshot)
    ok &= compare_arrays(
        "depth.raw_cropped_clamped",
        python_raw_depth,
        np.asarray(snapshot["rawDepth"], dtype=np.float32),
        args.atol,
        args.rtol,
    )
    ok &= compare_arrays(
        "depth.observation",
        python_depth_obs,
        np.asarray(snapshot["depthObservation"], dtype=np.float32),
        args.atol,
        args.rtol,
    )

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
