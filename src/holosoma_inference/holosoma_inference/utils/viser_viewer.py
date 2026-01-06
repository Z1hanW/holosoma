from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from loguru import logger

from holosoma_inference.config.config_types.robot import RobotConfig
from holosoma_inference.config.config_types.viser import ViserConfig


def _add_viser_to_path() -> None:
    current = Path(__file__).resolve()
    for _ in range(8):
        candidate = current / "viser" / "src"
        if candidate.exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return
        current = current.parent


_add_viser_to_path()

try:
    import viser  # type: ignore[import-not-found]
    from viser.extras import ViserUrdf  # type: ignore[import-not-found]
except Exception as exc:  # pragma: no cover - import guard
    raise ImportError("Viser is required for visualization but could not be imported.") from exc


def _load_onnx_metadata(model_path: str) -> dict:
    import onnx

    onnx_model = onnx.load(model_path)
    metadata: dict[str, object] = {}
    for prop in onnx_model.metadata_props:
        try:
            metadata[prop.key] = json.loads(prop.value)
        except json.JSONDecodeError:
            metadata[prop.key] = prop.value
    return metadata


def _resolve_urdf_path(cfg: ViserConfig, model_path: str) -> Path:
    if cfg.urdf_path:
        urdf_path = Path(cfg.urdf_path)
        if not urdf_path.exists():
            raise FileNotFoundError(f"Viser URDF path does not exist: {urdf_path}")
        return urdf_path

    metadata = _load_onnx_metadata(model_path)
    urdf_path_value = metadata.get("robot_urdf_path")
    if isinstance(urdf_path_value, str):
        candidate = Path(urdf_path_value)
        if candidate.exists():
            return candidate

    urdf_text = metadata.get("robot_urdf")
    if isinstance(urdf_text, str):
        tmp_dir = Path(tempfile.mkdtemp(prefix="holosoma_viser_urdf_"))
        urdf_path = tmp_dir / "robot.urdf"
        urdf_path.write_text(urdf_text, encoding="utf-8")
        logger.warning("Viser using URDF text from ONNX metadata; mesh paths may be unresolved.")
        return urdf_path

    raise ValueError(
        "URDF not found in ONNX metadata. Provide a path via --viser.urdf-path."
    )


class ViserInferenceViewer:
    def __init__(self, robot_config: RobotConfig, cfg: ViserConfig, model_path: str) -> None:
        self._robot_config = robot_config
        self._num_dofs = len(robot_config.dof_names)

        urdf_path = _resolve_urdf_path(cfg, model_path)

        self.server = viser.ViserServer(port=cfg.port)
        self.robot_root = self.server.scene.add_frame("/robot", show_axes=False)
        self.robot = ViserUrdf(self.server, urdf_or_path=urdf_path, root_node_name="/robot")
        self.robot.show_visual = cfg.show_meshes

        viser_joint_names = list(self.robot.get_actuated_joint_names())
        name_to_robot_idx = {name: idx for idx, name in enumerate(robot_config.dof_names)}
        missing = [name for name in viser_joint_names if name not in name_to_robot_idx]
        if missing:
            raise ValueError(f"Viser joints not found in robot config: {missing}")
        self.joint_order = [name_to_robot_idx[name] for name in viser_joint_names]

        if cfg.add_grid:
            self.server.scene.add_grid(
                "/grid",
                width=cfg.grid_size,
                height=cfg.grid_size,
                position=(0.0, 0.0, 0.0),
            )

        with self.server.gui.add_folder("Display"):
            show_meshes_cb = self.server.gui.add_checkbox("Show meshes", initial_value=cfg.show_meshes)

        @show_meshes_cb.on_update
        def _(_evt) -> None:
            self.robot.show_visual = bool(show_meshes_cb.value)

        logger.info("Viser server running on port %s", cfg.port)

    def update(self, robot_state_data) -> None:
        if robot_state_data is None:
            return
        state = np.asarray(robot_state_data)
        if state.ndim == 1:
            state = state.reshape(1, -1)

        expected = 7 + self._num_dofs
        if state.shape[1] < expected:
            raise ValueError(
                f"Robot state has {state.shape[1]} entries, expected at least {expected}."
            )

        root_pos = state[0, :3]
        root_quat_wxyz = state[0, 3:7]
        dof_pos = state[0, 7 : 7 + self._num_dofs]

        self.robot_root.position = root_pos
        self.robot_root.wxyz = root_quat_wxyz
        self.robot.update_cfg(dof_pos[self.joint_order])
