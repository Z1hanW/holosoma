from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import tyro

# Ensure local packages are importable when running from source.
SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

REPO_ROOT = Path(__file__).resolve().parents[3]
VISER_SRC = REPO_ROOT / "viser" / "src"
if VISER_SRC.exists() and str(VISER_SRC) not in sys.path:
    sys.path.insert(0, str(VISER_SRC))

import viser  # type: ignore[import-not-found]  # noqa: E402
from viser.extras import ViserUrdf  # type: ignore[import-not-found]  # noqa: E402

from holosoma.config_types.env import get_tyro_env_config  # noqa: E402
from holosoma.config_types.experiment import ExperimentConfig  # noqa: E402
from holosoma.config_values.experiment import AnnotatedExperimentConfig  # noqa: E402
from holosoma.utils.eval_utils import init_sim_imports  # noqa: E402
from holosoma.utils.helpers import get_class  # noqa: E402
from holosoma.utils.module_utils import get_holosoma_root  # noqa: E402
from holosoma.utils.path import resolve_data_file_path  # noqa: E402
from holosoma.utils.sim_utils import close_simulation_app  # noqa: E402
from holosoma.utils.tyro_utils import TYRO_CONIFG  # noqa: E402


def _resolve_data_path(path: str) -> str:
    if path.startswith("@holosoma/"):
        return str(Path(get_holosoma_root()) / path[len("@holosoma/") :])
    return resolve_data_file_path(path)


def _resolve_robot_urdf_path(cfg: ExperimentConfig) -> str:
    asset_root = _resolve_data_path(cfg.robot.asset.asset_root)
    urdf_path = os.path.join(asset_root, cfg.robot.asset.urdf_file)
    return _resolve_data_path(urdf_path)


def _depth_to_rgb(depth: np.ndarray, near: float, far: float) -> np.ndarray:
    depth = np.nan_to_num(depth, nan=far, posinf=far, neginf=near)
    depth = np.clip(depth, near, far)
    denom = max(far - near, 1.0e-6)
    norm = (depth - near) / denom
    inv = 1.0 - norm
    img = (inv * 255.0).astype(np.uint8)
    return np.repeat(img[..., None], 3, axis=2)


def replay_perception(cfg: ExperimentConfig) -> None:
    simulation_app = init_sim_imports(cfg)

    import torch
    from holosoma.utils.common import seeding

    seeding(42, torch_deterministic=False)

    env_target = cfg.env_class
    tyro_env_config = get_tyro_env_config(cfg)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = get_class(env_target)(tyro_env_config, device=device)

    perception = env.perception_manager
    if perception is None or not perception.enabled or perception.cfg.output_mode != "camera_depth":
        raise RuntimeError("Perception camera_depth is required for viser_perception.")

    server = viser.ViserServer(port=int(os.environ.get("HOLOSOMA_VISER_PORT", "6060")))
    robot_root = server.scene.add_frame("/robot", show_axes=False)
    robot_urdf_path = _resolve_robot_urdf_path(cfg)
    vr = ViserUrdf(server, urdf_or_path=Path(robot_urdf_path), root_node_name="/robot")

    camera_frame = server.scene.add_frame(
        "/robot/d435i",
        show_axes=True,
        axes_length=0.12,
        axes_radius=0.006,
        origin_color=(0, 200, 255),
    )

    viser_joint_names = list(vr.get_actuated_joint_names())
    name_to_robot_idx = {name: idx for idx, name in enumerate(cfg.robot.dof_names)}
    missing = [name for name in viser_joint_names if name not in name_to_robot_idx]
    if missing:
        raise ValueError(f"Viser URDF joints missing in robot config: {missing}")

    env.simulator.sim.step()
    env.step_visualize_motion(None)  # type: ignore[attr-defined]
    perception.update()

    depth_map = perception.get_camera_depth_map()[0].detach().cpu().numpy()
    depth_img = _depth_to_rgb(depth_map, perception.cfg.camera_near, perception.cfg.max_distance)
    depth_handle = server.gui.add_image(depth_img, label="D435i Depth")

    try:
        done = False
        while not done:
            env.simulator.sim.step()
            done = env.step_visualize_motion(None)  # type: ignore[attr-defined]

            perception.update()
            depth_map = perception.get_camera_depth_map()[0].detach().cpu().numpy()
            depth_handle.image = _depth_to_rgb(depth_map, perception.cfg.camera_near, perception.cfg.max_distance)

            root_state = env.simulator.robot_root_states[0]
            root_pos = root_state[:3].detach().cpu().numpy()
            root_quat_xyzw = root_state[3:7].detach().cpu().numpy()
            root_quat_wxyz = root_quat_xyzw[[3, 0, 1, 2]]
            robot_root.position = root_pos
            robot_root.wxyz = root_quat_wxyz

            dof_pos = env.simulator.dof_pos[0].detach().cpu().numpy()
            viser_joints = np.array([dof_pos[name_to_robot_idx[name]] for name in viser_joint_names], dtype=np.float32)
            vr.update_cfg(viser_joints)

            cam_pos, cam_quat = perception.get_camera_pose()
            cam_pos_np = cam_pos[0].detach().cpu().numpy()
            cam_quat_xyzw = cam_quat[0].detach().cpu().numpy()
            cam_quat_wxyz = cam_quat_xyzw[[3, 0, 1, 2]]
            camera_frame.position = cam_pos_np
            camera_frame.wxyz = cam_quat_wxyz
    finally:
        close_simulation_app(simulation_app)

    print("Open the viewer URL printed above. Close the process (Ctrl+C) to exit.")
    while True:
        time.sleep(1.0)


def main() -> None:
    tyro_cfg = tyro.cli(AnnotatedExperimentConfig, config=TYRO_CONIFG)
    replay_perception(tyro_cfg)


if __name__ == "__main__":
    main()
