from pathlib import Path
import argparse
import numpy as np

# ----------------------------
# Inputs (CRISP output)
# ----------------------------
CRISP_BIG_PATH = Path("/home/ANT.AMAZON.COM/zzzihanw/FAR/CRISP-Real2Sim/results/output/post_scene/far_robot/gv")
JOINT_FILE = CRISP_BIG_PATH / 'hmr' / "far_robot.npz"
Z_UP_MESH_FILE = CRISP_BIG_PATH / "saved_obj" / "hmr_after_shared_0000.obj"

# ----------------------------
# Output (Holosoma)
# ----------------------------
HOLOSOMA_ROOT = Path("/home/ANT.AMAZON.COM/zzzihanw/FAR/holosoma")  # change if needed


def load_obj_min_z(obj_path: Path) -> float:
    """Parse an .obj and return (max_z - min_z) over all vertices."""
    min_z = None
    max_z = None
    with obj_path.open("r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    z = float(parts[3])
                    min_z = z if min_z is None else min(min_z, z)
                    max_z = z if max_z is None else max(max_z, z)

    if min_z is None or max_z is None:
        raise RuntimeError(f"No vertices found in: {obj_path}")

    return float(max_z - min_z)


def pick_joints_array(npz: np.lib.npyio.NpzFile) -> np.ndarray:
    """Try common keys for joints and normalize to (T, J, 3)."""
    keys = list(npz.keys())
    print(keys)
    
    candidates = [
        "joints", "J", "j3d", "joints3d", "joints_world", "global_joint_positions",
        "pred_joints", "post_scene", "joint_pos",
    ]

    arr = None
    chosen_key = None

    for k in candidates:
        if k in npz:
            arr = npz[k]
            chosen_key = k
            break

    if arr is None:
        raise KeyError(
            f"Could not find joints array in {JOINT_FILE}.\n"
            f"Available keys: {keys}\n"
            f"Add your key name to `candidates` in pick_joints_array()."
        )

    arr = np.asarray(arr)

    # (T, J, 3)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr

    # (T, 3, J)
    if arr.ndim == 3 and arr.shape[1] == 3:
        return np.transpose(arr, (0, 2, 1))

    # (T, J*3)
    if arr.ndim == 2 and arr.shape[-1] % 3 == 0:
        J = arr.shape[-1] // 3
        return arr.reshape(arr.shape[0], J, 3)

    raise ValueError(f"Unsupported joints shape {arr.shape} from key '{chosen_key}'.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seq_name", type=str, help="Sequence name under holosoma/demo_data/")
    parser.add_argument("--holosoma_root", type=str, default=str(HOLOSOMA_ROOT), help="Path to holosoma repo root")
    parser.add_argument("--crisp_big_path", type=str, default=str(CRISP_BIG_PATH), help="Path to CRISP gv output folder")
    parser.add_argument("--joint_file", type=str, default=str(JOINT_FILE), help="Path to source joints npz")
    parser.add_argument("--mesh_file", type=str, default=str(Z_UP_MESH_FILE), help="Path to source z-up obj")
    args = parser.parse_args()

    holosoma_root = Path(args.holosoma_root)
    crisp_big_path = Path(args.crisp_big_path)
    joint_file = Path(args.joint_file)
    mesh_file = Path(args.mesh_file)

    if not holosoma_root.exists():
        raise FileNotFoundError(holosoma_root)
    if not crisp_big_path.exists():
        raise FileNotFoundError(crisp_big_path)
    if not joint_file.exists():
        raise FileNotFoundError(joint_file)
    if not mesh_file.exists():
        raise FileNotFoundError(mesh_file)

    out_dir = holosoma_root / "src/holosoma_retargeting/demo_data" / args.seq_name
    out_file = out_dir / "far_robot.npz"
    out_dir.mkdir(parents=True, exist_ok=True)

    # height offset so that mesh min-z becomes 0
    height = load_obj_min_z(mesh_file)

    data = np.load(joint_file, allow_pickle=True)
    joints = pick_joints_array(data)[:, :22, :]#.transpose(1, 0, 2)  # (T, J, 3)
    print(joints.shape)

    np.savez_compressed(
        out_file,
        global_joint_positions=joints.astype(np.float32),
        height=np.float32(height),
        seq_name=str(args.seq_name),
        src_joint_file=str(joint_file),
        src_mesh_file=str(mesh_file),
    )

    print(f"Saved: {out_file}")
    print(f"joints shape: {joints.shape} | height (=-min_z): {height:.6f}")


if __name__ == "__main__":
    main()
