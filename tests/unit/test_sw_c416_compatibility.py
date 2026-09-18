"""SW's user-verified deployment is a Git-pinned compatibility contract."""

import ast
import hashlib
from pathlib import Path
import subprocess
from types import SimpleNamespace

import cv2
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
BASELINE = "c416c75ac5bd09c3449336ba3de4b466874ff728"
IMAGE_SERVER = "src/holosoma/holosoma/sensors/image_server.py"
WBT = "src/holosoma_inference/holosoma_inference/policies/wbt.py"


def historical(path):
    return subprocess.check_output(["git", "-C", str(ROOT), "show", f"{BASELINE}:{path}"])


def method(source, name):
    return next(node for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.FunctionDef) and node.name == name)


@pytest.mark.parametrize("path", [
    "src/holosoma/holosoma/config_types/image_server.py",
    "src/holosoma/holosoma/config_values/image_server.py",
    "src/holosoma_inference/holosoma_inference/config/config_values/camera.py",
    "src/holosoma_inference/holosoma_inference/config/config_values/observation.py",
    "src/holosoma_inference/holosoma_inference/config/config_values/robot.py",
    "src/holosoma_inference/holosoma_inference/policies/base.py",
    "src/holosoma_inference/holosoma_inference/sdk/unitree/unitree_interface.py",
])
def test_configs_and_control_chain_are_byte_identical_to_c416(path):
    assert (ROOT / path).read_bytes() == historical(path)


@pytest.mark.parametrize("path,name", [
    (IMAGE_SERVER, "_resize_clip_expand_transpose"),
    (WBT, "_get_sparse_target_root_trajectory_command"),
    (WBT, "_get_sparse_target_root_trajectory_command_contact_aware"),
    (WBT, "_update_sparse_root_joystick_command"),
    (WBT, "_handle_sparse_root_keyboard_command"),
    (WBT, "_handle_drop_button_joystick_command"),
    (WBT, "_handle_drop_button_keyboard_command"),
    ("src/holosoma/holosoma/sensors/realsense.py", "_compute_latency"),
])
def test_command_and_depth_operations_are_identical_to_c416(path, name):
    assert ast.dump(method((ROOT / path).read_text(), name)) == ast.dump(method(historical(path), name))


def test_real_launchers_default_to_sw_without_additional_capture_load():
    for name in ("real_drop.sh", "real_depth.sh"):
        script = (ROOT / name).read_text()
        assert '${HOLOSOMA_DEPLOYMENT_AUDIT:-0}' in script
        assert "unset HOLOSOMA_DEPLOYMENT_AUDIT_DIR" in script
    script = (ROOT / "real_drop.sh").read_text()
    assert '${HOLOSOMA_REAL_MODEL_PATH:-_ckps/swl41n4x_model_15500.onnx}' in script
    assert '--task.model-path "$checkpoint"' in script


@pytest.mark.parametrize("enabled", [False, True])
def test_extra_sensor_metadata_is_opt_in(enabled):
    from holosoma.sensors.realsense import RealSenseCamera, RealSenseCameraConfig

    frame_number_reads = []

    def frame_number():
        frame_number_reads.append(1)
        return 7

    frame = SimpleNamespace(get_timestamp=lambda: 1000., get_frame_timestamp_domain=lambda: "global",
                            get_frame_number=frame_number,
                            get_data=lambda: np.array([[0, 1000], [2000, 3000]], dtype=np.uint16))
    camera = RealSenseCamera.__new__(RealSenseCamera)
    camera.rs = SimpleNamespace(timestamp_domain=SimpleNamespace(global_time="global"))
    camera.config = RealSenseCameraConfig(enable_rgb=False)
    camera.pipeline = SimpleNamespace(wait_for_frames=lambda: SimpleNamespace(get_depth_frame=lambda: frame))
    camera.align = None
    camera.depth_scale = 0.001
    camera._audit_capture_timestamps = enabled
    values = camera.capture()
    np.testing.assert_allclose(values["depth"], [[0, 1], [2, 3]], rtol=0, atol=3e-7)
    assert len(frame_number_reads) == int(enabled)
    assert hasattr(camera, "last_capture_metadata") == enabled


def test_real_sw_actions_match_c416_for_native_depth_probes():
    import onnxruntime as ort
    from holosoma.config_values.image_server import real_d435i

    path = "_ckps/swl41n4x_model_15500.onnx"
    model = (ROOT / path).read_bytes()
    assert model == historical(path)
    assert hashlib.sha256(model).hexdigest() == "e70823f20d30ce6c25752e9494aaf9be4e136940ef8af59cf439ec9107b564b0"
    assert real_d435i.latency_frame == (3, 3) and real_d435i.buffer_len == 4

    preprocessors = []
    for source in (historical(IMAGE_SERVER), (ROOT / IMAGE_SERVER).read_text()):
        node = method(source, "_resize_clip_expand_transpose")
        namespace = {"np": np, "cv2": cv2}
        exec(compile(ast.Module(body=[node], type_ignores=[]), IMAGE_SERVER, "exec"), namespace)
        preprocessors.append(namespace[node.name])
    server = SimpleNamespace(cfg=real_d435i)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(model, sess_options=options, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(416)
    for index in range(32):
        frame = rng.uniform(0.05, 4.0, (480, 848)).astype(np.float32)
        if index == 0:
            frame[:] = 0
        elif index == 1:
            frame[:] = 1.0
        elif index == 2:
            frame[:, :424], frame[:, 424:] = 0.3, 3.0
        else:
            frame.flat[::7] = 0.0
        old, new = [fn(server, frame.copy()).reshape(1, -1) for fn in preprocessors]
        np.testing.assert_array_equal(old, new)
        actor = rng.normal(0, 0.1, (1, 94)).astype(np.float32)
        first = session.run(["action"], {"actor_obs": actor, "perception_obs": old})[0]
        second = session.run(["action"], {"actor_obs": actor, "perception_obs": new})[0]
        np.testing.assert_array_equal(first, second)
