#!/usr/bin/env python3
"""Check that the policy can receive real perception_obs over ZMQ."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import zmq


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5658)
    parser.add_argument("--key", default="perception_obs")
    parser.add_argument("--expected-dim", type=int, default=5046)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.setsockopt(zmq.RCVTIMEO, 100)
    socket.connect(f"tcp://localhost:{args.port}")

    deadline = time.monotonic() + max(args.timeout, 0.0)
    count = 0
    try:
        while time.monotonic() <= deadline:
            try:
                payload = json.loads(socket.recv_string())
            except zmq.Again:
                continue
            except json.JSONDecodeError as exc:
                print(f"[WARN] Ignoring non-JSON payload: {exc}")
                continue

            if args.key not in payload:
                print(f"[WARN] Payload missing key '{args.key}': keys={sorted(payload.keys())}")
                continue

            obs = np.asarray(payload[args.key], dtype=np.float32).reshape(-1)
            count += 1
            ok = obs.size == args.expected_dim
            finite = bool(np.isfinite(obs).all())
            sim_time = payload.get("sim_time_ms", payload.get("time_ms", "n/a"))
            if obs.size:
                obs_min = float(np.nanmin(obs))
                obs_max = float(np.nanmax(obs))
                obs_mean = float(np.nanmean(obs))
                obs_std = float(np.nanstd(obs))
            else:
                obs_min = obs_max = obs_mean = obs_std = float("nan")
            print(
                "[OK]" if ok and finite else "[BAD]",
                f"msg={count}",
                f"dim={obs.size}",
                f"expected={args.expected_dim}",
                f"finite={finite}",
                f"min={obs_min:.4f}",
                f"max={obs_max:.4f}",
                f"mean={obs_mean:.4f}",
                f"std={obs_std:.4f}",
                f"time_ms={sim_time}",
            )
            if args.once:
                return 0 if ok and finite else 1
    finally:
        socket.close(0)
        context.term()

    print(f"[ERROR] No valid '{args.key}' payload received on tcp://localhost:{args.port} within {args.timeout:.1f}s.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
