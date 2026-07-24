from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from holosoma.utils.motion_transition_source import (
    canonical_motion_transition_source,
    resolve_motion_transition_source_for_motion_path,
)


MAX_VISUAL_MOTION_TRANSITION_STEPS = 4096
_SUPPORTED_SIMULATOR_TYPES = {"isaacgym", "isaacsim", "mujoco"}


@dataclass(frozen=True)
class VisualMotionTransitionPlan:
    """Effective viewer timeline derived from the unfiltered motion source."""

    source_semantics: str
    prepend_steps: int
    append_steps: int


def configured_simulator_type(simulator_cfg: Any) -> str:
    """Resolve and cross-check the exact simulator backend in an experiment config."""

    target = getattr(simulator_cfg, "_target_", None)
    nested_cfg = getattr(simulator_cfg, "config", None)
    configured_name = getattr(nested_cfg, "name", None)
    if not isinstance(target, str) or not target.strip():
        raise ValueError("simulator._target_ must be a non-empty string.")
    if not isinstance(configured_name, str) or not configured_name.strip():
        raise ValueError("simulator.config.name must be a non-empty string.")
    target_name = target.rsplit(".", 1)[-1].lower()
    configured_name = configured_name.lower()
    if target_name != configured_name or target_name not in _SUPPORTED_SIMULATOR_TYPES:
        raise ValueError(
            "Simulator target/config name mismatch or unsupported backend: "
            f"target={target!r}, config.name={configured_name!r}."
        )
    return target_name


def configured_control_dt_s(simulator_cfg: Any) -> float:
    """Return the same control timestep used by ``BaseTask`` construction."""

    nested_cfg = getattr(simulator_cfg, "config", None)
    sim_cfg = getattr(nested_cfg, "sim", None)
    fps = getattr(sim_cfg, "fps", None)
    decimation = getattr(sim_cfg, "control_decimation", None)
    if (
        isinstance(fps, bool)
        or not isinstance(fps, Real)
        or not math.isfinite(float(fps))
        or float(fps) <= 0.0
        or isinstance(decimation, bool)
        or not isinstance(decimation, Integral)
        or int(decimation) <= 0
    ):
        raise ValueError(
            "simulator.config.sim must provide finite positive fps and positive integer control_decimation."
        )
    return int(decimation) / float(fps)


def list_motion_source_clips(motion_path: Path) -> list[str]:
    """List clip ids before a viewer pins one clip for display.

    A selected clip from a global bank must retain global-bank transition
    semantics. Counting after ``MotionLoader`` applies ``motion_clip_name``
    would incorrectly turn that clip into a standalone single-clip source.
    """

    motion_path = motion_path.expanduser()
    if motion_path.is_dir():
        files = sorted(
            path
            for path in motion_path.iterdir()
            if path.is_file() and path.suffix.lower() == ".npz"
        )
        if not files:
            raise FileNotFoundError(f"No motion clips found in directory: {motion_path}")
        return [path.stem for path in files]

    if motion_path.suffix.lower() in {".h5", ".hdf5"}:
        try:
            import h5py  # type: ignore[import-not-found]
        except ImportError:
            # MotionLoader will surface the missing optional dependency when it
            # loads the file. Preserve the conservative standalone fallback for
            # legacy HDF5 files with no multi-clip group.
            return [motion_path.stem]
        with h5py.File(motion_path, "r") as h5f:
            clips = h5f.get("clips")
            if clips is None or "clip_ids" not in clips:
                return [motion_path.stem]
            clip_ids: list[str] = []
            for item in clips["clip_ids"]:
                clip_ids.append(
                    item.decode("utf-8")
                    if isinstance(item, (bytes, bytearray))
                    else str(item)
                )
            if not clip_ids:
                raise ValueError(f"HDF5 motion bank has an empty /clips/clip_ids dataset: {motion_path}")
            return clip_ids

    if motion_path.is_file():
        return [motion_path.stem]
    raise FileNotFoundError(f"Motion file not found: {motion_path}")


def resolve_visual_motion_transition_plan(
    motion_cfg: Any,
    *,
    fps: float,
    control_dt_s: float,
    source_clip_count: int,
    simulator_type: str,
    motion_transition_source: dict[str, Any] | None = None,
) -> VisualMotionTransitionPlan:
    """Resolve requested viewer splices using training's source semantics.

    Viewers have no authenticated checkpoint artifact, so this deliberately
    mirrors the live command's source classification: an original global bank
    may display its requested prepend, but never fabricates the requested
    append that training skipped. A standalone source keeps both static
    splices.
    """

    if (
        isinstance(source_clip_count, bool)
        or not isinstance(source_clip_count, Integral)
        or int(source_clip_count) <= 0
    ):
        raise ValueError("source_clip_count must be a positive integer.")
    if isinstance(fps, bool) or not isinstance(fps, Real) or not math.isfinite(float(fps)) or float(fps) <= 0.0:
        raise ValueError("fps must be a finite positive real number.")
    if (
        isinstance(control_dt_s, bool)
        or not isinstance(control_dt_s, Real)
        or not math.isfinite(float(control_dt_s))
        or float(control_dt_s) <= 0.0
    ):
        raise ValueError("control_dt_s must be a finite positive real number.")
    control_fps = 1.0 / float(control_dt_s)
    if not math.isclose(float(fps), control_fps, rel_tol=1.0e-6, abs_tol=1.0e-6):
        raise ValueError(
            "Motion FPS must match the configured control frequency before visual transition splicing: "
            f"motion.fps={fps}, control_fps={control_fps}."
        )
    if simulator_type not in _SUPPORTED_SIMULATOR_TYPES:
        raise ValueError(
            f"simulator_type must be one of {sorted(_SUPPORTED_SIMULATOR_TYPES)}, got {simulator_type!r}."
        )

    def requested_steps(phase_name: str) -> int:
        enabled = getattr(motion_cfg, f"enable_default_pose_{phase_name}", None)
        duration = getattr(motion_cfg, f"default_pose_{phase_name}_duration_s", None)
        if type(enabled) is not bool:
            raise ValueError(f"enable_default_pose_{phase_name} must be boolean.")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, Real)
            or not math.isfinite(float(duration))
            or float(duration) < 0.0
        ):
            raise ValueError(
                f"default_pose_{phase_name}_duration_s must be finite and non-negative."
            )
        if not enabled:
            return 0
        # Match MotionCommand exactly. It uses round(duration / env.dt), and
        # BaseTask constructs env.dt from this serialized control timestep.
        steps = round(float(duration) / float(control_dt_s))
        if steps <= 1:
            return 0
        if steps > MAX_VISUAL_MOTION_TRANSITION_STEPS:
            raise ValueError(
                f"Default-pose {phase_name} requires {steps} frames, exceeding the safe maximum "
                f"{MAX_VISUAL_MOTION_TRANSITION_STEPS}."
            )
        return int(steps)

    prepend_steps = requested_steps("prepend")
    append_steps = requested_steps("append")
    if motion_transition_source is not None:
        source_semantics = canonical_motion_transition_source(
            motion_transition_source,
            active_clip_count=int(source_clip_count),
            role="visual motion_transition_source",
        )["source_semantics"]
    else:
        source_semantics = (
            "global_multi_clip_runtime"
            if int(source_clip_count) > 1
            else "single_clip_static"
        )

    if source_semantics == "global_multi_clip_runtime":
        return VisualMotionTransitionPlan(
            source_semantics="global_multi_clip_runtime",
            # The live global-bank implementation is an IsaacSim-only runtime
            # blend. MuJoCo and IsaacGym explicitly disable it.
            prepend_steps=prepend_steps if simulator_type == "isaacsim" else 0,
            append_steps=0,
        )
    return VisualMotionTransitionPlan(
        source_semantics="single_clip_static",
        prepend_steps=prepend_steps,
        append_steps=append_steps,
    )
