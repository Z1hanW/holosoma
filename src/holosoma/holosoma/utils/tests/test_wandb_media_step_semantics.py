from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from holosoma.envs.base_task.base_task import BaseTask
from holosoma.simulator.shared.video_recorder import VideoRecorderInterface
from holosoma.utils import video_utils


def _wandb_stub() -> SimpleNamespace:
    return SimpleNamespace(
        run=object(),
        Image=MagicMock(side_effect=lambda frame, caption=None: (frame, caption)),
        Video=MagicMock(side_effect=lambda path, format=None: (path, format)),
        log=MagicMock(),
    )


def test_startup_depth_and_frame0_buffer_media_without_advancing_step() -> None:
    wandb = _wandb_stub()
    task = object.__new__(BaseTask)
    task._depth_log_episode_id = 17
    task._depth_log_startup_done = False
    task._depth_log_is_main_process = True
    task._depth_log_record_env_id = 0
    task.perception_manager = SimpleNamespace(
        enabled=True,
        cfg=SimpleNamespace(output_mode="camera_depth"),
    )
    task.simulator = SimpleNamespace(logger_cfg=SimpleNamespace(type="wandb"))
    task._resolve_depth_obs_source = MagicMock()
    task._extract_policy_depth_frame = MagicMock(return_value=np.zeros((2, 2), dtype=np.float32))
    task._depth_to_rgb = MagicMock(return_value=np.zeros((2, 2, 3), dtype=np.uint8))

    with patch.dict("sys.modules", {"wandb": wandb}):
        task._log_startup_depth_if_needed()
        task._log_depth_frame0(np.zeros((2, 2, 3), dtype=np.uint8))

    assert [call.kwargs for call in wandb.log.call_args_list] == [
        {"commit": False},
        {"commit": False},
    ]
    assert list(wandb.log.call_args_list[0].args[0]) == ["Depth/startup"]
    assert list(wandb.log.call_args_list[1].args[0]) == ["Depth/frame0"]
    assert task._depth_log_startup_done is True


def test_create_video_exposes_non_committing_wandb_media_upload(tmp_path) -> None:
    wandb = _wandb_stub()
    writer = MagicMock()
    completed = SimpleNamespace(returncode=0, stderr="")

    with (
        patch.object(video_utils, "wandb", wandb),
        patch.object(video_utils, "_is_wandb_available", return_value=True),
        patch.object(video_utils.cv2, "VideoWriter_fourcc", return_value=0),
        patch.object(video_utils.cv2, "VideoWriter", return_value=writer),
        patch.object(video_utils.cv2, "cvtColor", side_effect=lambda frame, _code: frame),
        patch.object(video_utils.subprocess, "run", return_value=completed),
    ):
        video_utils.create_video(
            video_frames=np.zeros((1, 2, 2, 3), dtype=np.uint8),
            fps=10,
            save_dir=tmp_path,
            output_format="h264",
            wandb_logging=True,
            wandb_commit=False,
            wandb_key="Depth rollout",
        )

    wandb.log.assert_called_once()
    assert list(wandb.log.call_args.args[0]) == ["Depth rollout"]
    assert wandb.log.call_args.kwargs == {"commit": False}


def test_simulator_training_video_forwards_buffered_media_contract(tmp_path) -> None:
    recorder = SimpleNamespace(
        video_frames=[np.zeros((2, 2, 3), dtype=np.uint8)],
        simulator=SimpleNamespace(
            simulator_config=SimpleNamespace(
                sim=SimpleNamespace(fps=50, control_decimation=1),
            ),
        ),
        config=SimpleNamespace(
            playback_rate=1.0,
            output_format="mp4",
            upload_to_wandb=True,
        ),
        _current_episode=3,
        _get_save_directory=MagicMock(return_value=tmp_path),
        _clear_frame_buffer=MagicMock(),
    )

    with patch("holosoma.simulator.shared.video_recorder.create_video") as create_video:
        VideoRecorderInterface._encode_and_save_video(recorder)

    assert create_video.call_args.kwargs["wandb_commit"] is False
    recorder._clear_frame_buffer.assert_called_once_with()
