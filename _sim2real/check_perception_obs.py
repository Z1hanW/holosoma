#!/usr/bin/env python3
"""Check that the policy can receive real perception_obs over shared memory."""

from __future__ import annotations

import argparse
import time
from multiprocessing import resource_tracker
from multiprocessing import shared_memory

import numpy as np


def _print_stats(prefix: str, obs: np.ndarray, *, expected_dim: int, count: int) -> bool:
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    ok = obs.size == int(expected_dim)
    finite = bool(np.isfinite(obs).all())
    if obs.size:
        obs_min = float(np.nanmin(obs))
        obs_max = float(np.nanmax(obs))
        obs_mean = float(np.nanmean(obs))
        obs_std = float(np.nanstd(obs))
    else:
        obs_min = obs_max = obs_mean = obs_std = float("nan")
    print(
        "[OK]" if ok and finite else "[BAD]",
        prefix,
        f"msg={count}",
        f"dim={obs.size}",
        f"expected={expected_dim}",
        f"finite={finite}",
        f"min={obs_min:.4f}",
        f"max={obs_max:.4f}",
        f"mean={obs_mean:.4f}",
        f"std={obs_std:.4f}",
    )
    return ok and finite


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shm-name", default="depth_img_shm")
    parser.add_argument("--expected-dim", type=int, default=5046)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    deadline = time.monotonic() + max(args.timeout, 0.0)
    shm = None
    try:
        while time.monotonic() <= deadline:
            try:
                shm = shared_memory.SharedMemory(name=args.shm_name, create=False)
                try:
                    resource_tracker.unregister(shm._name, "shared_memory")
                except Exception:
                    pass
                break
            except FileNotFoundError:
                time.sleep(min(max(args.poll_interval, 0.01), 0.25))

        if shm is None:
            print(f"[ERROR] Shared memory '{args.shm_name}' not found within {args.timeout:.1f}s.")
            return 1

        expected_bytes = int(args.expected_dim) * np.dtype(np.float32).itemsize
        if len(shm.buf) < expected_bytes:
            print(
                f"[ERROR] Shared memory '{args.shm_name}' is too small: "
                f"{len(shm.buf)} bytes < {expected_bytes} bytes."
            )
            return 1

        arr = np.ndarray((int(args.expected_dim),), dtype=np.float32, buffer=shm.buf)
        count = 0
        while time.monotonic() <= deadline:
            count += 1
            ok = _print_stats(
                f"shm={args.shm_name}",
                arr.copy(),
                expected_dim=args.expected_dim,
                count=count,
            )
            if args.once:
                return 0 if ok else 1
            time.sleep(max(args.poll_interval, 0.01))
    finally:
        if shm is not None:
            shm.close()

    print(f"[ERROR] No valid shared-memory perception_obs received from '{args.shm_name}' within {args.timeout:.1f}s.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
