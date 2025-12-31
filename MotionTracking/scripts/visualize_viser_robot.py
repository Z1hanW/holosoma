#!/usr/bin/env python3
import argparse
import time
from pathlib import Path
import numpy as np
import yaml

from motion_tracking.utils.viser_visualizer import ViserHelper
from motion_tracking.utils.robot_viser import RobotMjcfViser


def main():
    ap = argparse.ArgumentParser(description="Visualize recorded robot motion with Viser.")
    # Make record_dir optional with a sensible default
    ap.add_argument(
        "record_dir",
        type=str,
        nargs="?",
        default="output/recordings/000",
        help="Directory like output/recordings/000",
    )
    ap.add_argument("--env_idx", type=int, default=0, help="Env index file suffix to load (default 0)")
    ap.add_argument("--port", type=int, default=8080, help="Viser server port")
    ap.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    ap.add_argument("--scene_obj", type=str, default="motion_tracking/data/assets/urdf/1104/pkr_grn_11/ours/mesh.obj", help="Optional path to a scene OBJ file to visualize")
    args = ap.parse_args()

    rec = Path(args.record_dir)
    if not rec.exists():
        raise FileNotFoundError(f"Record dir not found: {rec}")

    # Load robot info
    info_path = rec / "robot_vis_info.yaml"
    if info_path.exists():
        info = yaml.safe_load(info_path.read_text())
        body_names = info.get("body_names", [])
        dt = float(info.get("dt", 1.0 / 60.0)) / max(args.speed, 1e-6)
        asset_xml_rel = info.get("asset_xml", None)
    else:
        print("[Visualizer] robot_vis_info.yaml not found; proceeding with defaults.")
        body_names = []
        dt = 1.0 / 60.0 / max(args.speed, 1e-6)
        asset_xml_rel = None

    # Resolve MJCF path
    if asset_xml_rel is not None and (rec / asset_xml_rel).exists():
        mjcf_path = str(rec / asset_xml_rel)
    else:
        # Best-effort search in dir
        cand = list(rec.glob("*.xml"))
        mjcf_path = str(cand[0]) if cand else None
        if mjcf_path is None:
            raise FileNotFoundError("No MJCF xml found in record dir.")

    # Load rigid bodies trajectory
    npz_path = rec / f"rigid_bodies_{args.env_idx}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Rigid body file not found: {npz_path}")
    data = np.load(npz_path)
    pos = data["pos"]  # (T, B, 3)
    rot = data["rot"]  # (T, B, 4) xyzw
    T, B, _ = pos.shape

    viser = ViserHelper(port=args.port)
    if not viser.ok():
        print("[Visualizer] Viser not available; exiting.")
        return

    robot = RobotMjcfViser(viser, mjcf_path, body_names if body_names else None)

    # Optionally load and visualize an external scene OBJ
    if args.scene_obj is not None:
        obj_path = Path(args.scene_obj)
        if not obj_path.exists():
            print(f"[Visualizer] Scene OBJ not found: {obj_path}")
        else:
            def load_obj_file(filepath: Path):
                vertices = []
                faces = []
                with filepath.open('r') as f:
                    for line in f:
                        if line.startswith('v '):
                            parts = line.strip().split()
                            try:
                                v = [float(parts[1]), float(parts[2]), float(parts[3])]
                                vertices.append(v)
                            except Exception:
                                pass
                        elif line.startswith('f '):
                            parts = line.strip().split()[1:]
                            idxs = []
                            for p in parts:
                                try:
                                    # supports v, v/t, v//n, v/t/n
                                    vidx = int(p.split('/')[0]) - 1
                                    idxs.append(vidx)
                                except Exception:
                                    pass
                            # triangulate polygon (fan)
                            if len(idxs) >= 3:
                                for i in range(1, len(idxs) - 1):
                                    faces.append([idxs[0], idxs[i], idxs[i + 1]])
                return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)

            try:
                verts, tris = load_obj_file(obj_path)
                if verts.size > 0 and tris.size > 0:
                    viser.add_mesh_simple("/scene_obj", verts, tris, color=(0.6, 0.7, 0.9), side="double")
                    print(f"[Visualizer] Loaded scene OBJ: {obj_path}")
                else:
                    print(f"[Visualizer] Scene OBJ has no geometry: {obj_path}")
            except Exception as e:
                print(f"[Visualizer] Failed to load scene OBJ '{obj_path}': {e}")

    # Simple camera setup
    root0 = pos[0, 0]
    cam = root0 + np.array([0.0, -2.0, 1.5], dtype=np.float32)
    look = root0 + np.array([0.0, 0.0, 0.4], dtype=np.float32)
    viser.set_camera(cam, look)

    # Playback loop
    print("[Visualizer] Starting looped playback. Press Ctrl-C to exit.")
    try:
        while True:
            for t in range(T):
                robot.update(pos[t], rot[t])
                time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[Visualizer] Stopped by user.")


if __name__ == "__main__":
    main()
