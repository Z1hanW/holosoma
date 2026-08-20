#!/usr/bin/env python3
"""Run a dynamics-linearization canary on a real MuJoCo model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from holosoma_retargeting.trajectory_optimization import (
    MujocoDynamicsLinearizer,
    MujocoNominalTrajectory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frames < 2:
        raise ValueError("--frames must be at least 2")
    model = mujoco.MjModel.from_xml_path(str(args.model.expanduser().resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    qpos = np.empty((args.frames, model.nq), dtype=np.float64)
    qvel = np.empty((args.frames, model.nv), dtype=np.float64)
    controls = np.zeros((args.frames - 1, model.nu), dtype=np.float64)
    activations = (
        np.empty((args.frames, model.na), dtype=np.float64) if model.na else None
    )
    for frame in range(args.frames):
        qpos[frame] = data.qpos
        qvel[frame] = data.qvel
        if activations is not None:
            activations[frame] = data.act
        if frame < args.frames - 1:
            data.ctrl[:] = controls[frame]
            mujoco.mj_step(model, data)

    nominal = MujocoNominalTrajectory(
        qpos=qpos,
        qvel=qvel,
        controls=controls,
        activations=activations,
    )
    linearizer = MujocoDynamicsLinearizer(model)
    started = time.perf_counter()
    result = linearizer.linearize(nominal)
    elapsed = time.perf_counter() - started
    report = {
        "model": str(args.model.expanduser().resolve()),
        "frames": args.frames,
        "nq": model.nq,
        "nv": model.nv,
        "na": model.na,
        "nu": model.nu,
        "tangent_state_dimension": linearizer.state_dimension,
        "transition_shape": list(result.dynamics.transition.shape),
        "control_shape": list(result.dynamics.control.shape),
        "max_rollout_defect": float(np.max(result.defect_norms)),
        "linearization_time_s": elapsed,
        "finite": bool(
            np.isfinite(result.dynamics.transition).all()
            and np.isfinite(result.dynamics.control).all()
            and np.isfinite(result.dynamics.offset).all()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["finite"] or report["max_rollout_defect"] > 1e-8:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
