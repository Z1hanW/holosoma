"""Camera geometry helpers for pinhole projection."""

from __future__ import annotations

import math
from typing import Iterable

from holosoma.utils.safe_torch_import import torch


def resolve_camera_intrinsics(
    width: int,
    height: int,
    *,
    vfov_deg: float | None = None,
    hfov_deg: float | None = None,
    fx: float | None = None,
    fy: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
) -> tuple[float, float, float, float, float, float]:
    """Resolve pinhole intrinsics from FOV or focal lengths."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive.")

    if (fx is None) != (fy is None):
        raise ValueError("fx and fy must be provided together.")

    if fx is None:
        if vfov_deg is None and hfov_deg is None:
            raise ValueError("vfov_deg or hfov_deg must be provided when fx/fy are missing.")
        if vfov_deg is None:
            hfov_rad = math.radians(float(hfov_deg))
            vfov_rad = 2.0 * math.atan(math.tan(hfov_rad / 2.0) * height / width)
            vfov_deg = math.degrees(vfov_rad)
        if hfov_deg is None:
            vfov_rad = math.radians(float(vfov_deg))
            hfov_rad = 2.0 * math.atan(math.tan(vfov_rad / 2.0) * width / height)
            hfov_deg = math.degrees(hfov_rad)

        fx = width / (2.0 * math.tan(math.radians(float(hfov_deg)) / 2.0))
        fy = height / (2.0 * math.tan(math.radians(float(vfov_deg)) / 2.0))
    else:
        vfov_deg = math.degrees(2.0 * math.atan(height / (2.0 * float(fy))))
        hfov_deg = math.degrees(2.0 * math.atan(width / (2.0 * float(fx))))

    if cx is None:
        cx = width / 2.0
    if cy is None:
        cy = height / 2.0

    return float(fx), float(fy), float(cx), float(cy), float(vfov_deg), float(hfov_deg)


def build_camera_parameters(
    extrinsics: torch.Tensor | Iterable[float],
    *,
    width: int,
    height: int,
    vfov_deg: float | None = None,
    hfov_deg: float | None = None,
    fx: float | None = None,
    fy: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
    fps: float = 30.0,
    near: float = 0.1,
    far: float = 10.0,
    distortion: Iterable[float] | None = None,
) -> dict[str, torch.Tensor | float | int]:
    """Build camera parameter dictionary from batched extrinsics."""
    extrinsics_t = torch.as_tensor(extrinsics)
    if extrinsics_t.ndim == 2:
        extrinsics_t = extrinsics_t.unsqueeze(0)
    if extrinsics_t.ndim != 3 or extrinsics_t.shape[-2:] != (4, 4):
        raise ValueError("extrinsics must have shape (4, 4) or (N, 4, 4).")

    fx_val, fy_val, cx_val, cy_val, vfov_val, hfov_val = resolve_camera_intrinsics(
        width,
        height,
        vfov_deg=vfov_deg,
        hfov_deg=hfov_deg,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )

    k = torch.tensor(
        [[fx_val, 0.0, cx_val], [0.0, fy_val, cy_val], [0.0, 0.0, 1.0]],
        device=extrinsics_t.device,
        dtype=extrinsics_t.dtype,
    )
    k = k.unsqueeze(0).repeat(extrinsics_t.shape[0], 1, 1)

    if distortion is None:
        distortion = (0.0, 0.0, 0.0, 0.0, 0.0)
    distortion_t = torch.tensor(distortion, device=extrinsics_t.device, dtype=extrinsics_t.dtype)
    if distortion_t.numel() != 5:
        raise ValueError("distortion must have 5 coefficients (k1, k2, p1, p2, k3).")
    distortion_t = distortion_t.view(1, 5).repeat(extrinsics_t.shape[0], 1)

    return {
        "extrinsics": extrinsics_t,
        "K": k,
        "fx": fx_val,
        "fy": fy_val,
        "cx": cx_val,
        "cy": cy_val,
        "width": width,
        "height": height,
        "vfov_deg": vfov_val,
        "hfov_deg": hfov_val,
        "fps": float(fps),
        "near": float(near),
        "far": float(far),
        "distortion": distortion_t,
    }
