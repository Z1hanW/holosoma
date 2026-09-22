"""Publish the D435i's IR stereo pair over the network.

Runs on the machine the camera is plugged into. Streams left/right infrared
frames plus the stereo calibration so that a learned stereo depth model (e.g.
Fast-FoundationStereo via image_server.py --camera-type remote_stereo) can run
on another host. Standalone: needs only pyrealsense2, numpy and pyzmq.

The wire format is documented in holosoma/sensors/remote_stereo.py and must be
kept in sync with it. At 848x480 the pair is ~0.8 MB/frame, ~195 Mbit/s at
30 Hz - fine on gigabit, not on a 100 Mb/s link.

Usage (on the camera host):

    python stereo_relay_pub.py --bind 'tcp://*:5602' --emitter off
"""

from __future__ import annotations

import argparse
import signal
import struct
import sys
import time

import numpy as np
import zmq

HEADER_FMT = "<4sQdIIII"
MAGIC = b"HSS1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bind", default="tcp://*:5602")
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--emitter",
        choices=["on", "off"],
        default="on",
        help="IR projector. Its speckle pattern helps the D435i's own matcher; a learned model may not want it.",
    )
    parser.add_argument(
        "--with-depth",
        action="store_true",
        help="also stream the D435i hardware depth (doubles bandwidth; only for side-by-side comparison)",
    )
    parser.add_argument("--stats-interval", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.infrared, 1, args.width, args.height, rs.format.y8, args.fps)
    config.enable_stream(rs.stream.infrared, 2, args.width, args.height, rs.format.y8, args.fps)
    if args.with_depth:
        config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    profile = pipeline.start(config)

    sensor = profile.get_device().first_depth_sensor()
    if sensor.supports(rs.option.emitter_enabled):
        sensor.set_option(rs.option.emitter_enabled, 1.0 if args.emitter == "on" else 0.0)
    print(f"[stereo-relay] {args.width}x{args.height}@{args.fps}  emitter={args.emitter}  depth={'yes' if args.with_depth else 'no'}")

    # Same convention as holosoma.sensors.realsense: left is the reference frame,
    # the baseline is the right eye's translation (sign flipped to positive).
    lp = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
    rp = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()

    def k_of(i):
        return np.array([[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1]], dtype=np.float32)

    intr = np.stack([k_of(lp.get_intrinsics()), k_of(rp.get_intrinsics())])
    e = rp.get_extrinsics_to(lp)
    t = np.array(e.translation, dtype=np.float32)
    t[0] *= -1.0
    right_ext = np.eye(4, dtype=np.float32)
    right_ext[:3, :3] = np.array(e.rotation, dtype=np.float32).reshape(3, 3)
    right_ext[:3, 3] = t
    extr = np.stack([np.eye(4, dtype=np.float32), right_ext])
    calib_bytes = np.concatenate([intr.ravel(), extr.ravel()]).astype(np.float32).tobytes()
    print(f"[stereo-relay] fx={intr[0,0,0]:.2f}  baseline={abs(t[0]):.6f} m")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 1)
    sock.bind(args.bind)
    print(f"[stereo-relay] publishing on {args.bind}")

    stop = False

    def on_term(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, on_term)

    seq = sent = 0
    last = time.monotonic()
    timeouts = 0
    try:
        while not stop:
            try:
                frames = pipeline.wait_for_frames(5000)
            except RuntimeError as exc:
                # Usually another process has grabbed the camera, or the stream
                # stalled after a reconnect. Say so and keep trying for a while
                # instead of dying and leaving the laptop with a frozen frame.
                timeouts += 1
                print(f"[stereo-relay] no frames for 5 s ({timeouts}/6): {exc}", file=sys.stderr)
                if timeouts >= 6:
                    print("[stereo-relay] giving up; check for another RealSense client on this host", file=sys.stderr)
                    raise
                continue
            timeouts = 0
            l = frames.get_infrared_frame(1)
            r = frames.get_infrared_frame(2)
            if not l or not r:
                continue
            left = np.asanyarray(l.get_data())
            right = np.asanyarray(r.get_data())
            depth_bytes = b""
            has_depth = 0
            if args.with_depth:
                d = frames.get_depth_frame()
                if d:
                    depth_bytes = np.asanyarray(d.get_data()).astype(np.uint16).tobytes()
                    has_depth = 1
            header = struct.pack(HEADER_FMT, MAGIC, seq, time.time(), args.height, args.width, 1, has_depth)
            sock.send(header + calib_bytes + left.tobytes() + right.tobytes() + depth_bytes, copy=False)
            seq += 1
            sent += 1
            now = time.monotonic()
            if args.stats_interval > 0 and now - last >= args.stats_interval:
                print(f"[stereo-relay] seq={seq} rate={sent / (now - last):.1f} Hz  left mean={left.mean():.1f}")
                sent = 0
                last = now
    except KeyboardInterrupt:
        pass
    finally:
        print("[stereo-relay] stopping")
        sock.close(linger=0)
        ctx.term()
        pipeline.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
