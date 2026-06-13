from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.tensorboard import SummaryWriter

from holosoma.agents.modules.logging_utils import LoggingHelper, collect_reward_wandb_metadata
from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg


@pytest.fixture
def mock_writer():
    """Fixture providing a mock SummaryWriter."""
    return MagicMock(spec=SummaryWriter)


@pytest.fixture
def mock_wandb():
    """Fixture providing mocked wandb module."""
    with patch("holosoma.agents.modules.logging_utils.wandb") as mock_wandb:
        yield mock_wandb


@pytest.fixture
def logging_helper(mock_writer):
    """Fixture providing a LoggingHelper instance with default parameters."""
    return LoggingHelper(
        writer=mock_writer,
        log_dir="/tmp/test_logs",
        num_envs=2,
        num_steps_per_env=10,
        num_learning_iterations=100,
        device="cpu",
    )


@pytest.fixture
def prefixed_logging_helper(mock_writer):
    """Fixture providing a LoggingHelper instance with a prefix."""
    return LoggingHelper(
        writer=mock_writer,
        log_dir="/tmp/test_logs",
        num_envs=2,
        num_steps_per_env=10,
        num_learning_iterations=100,
        device="cpu",
        prefix="test_prefix/",
    )


def test_prefix_in_logging(prefixed_logging_helper, mock_writer, mock_wandb):
    """Test that the prefix is properly added to all logged metrics."""
    # Add episode info
    prefixed_logging_helper.ep_infos = [{"test_metric": torch.tensor([1.0], device=prefixed_logging_helper.device)}]

    # Call post_epoch_logging with some test data
    prefixed_logging_helper.post_epoch_logging(
        it=0,
        loss_dict={"test_loss": 0.5},
        extra_log_dicts={"test_section": {"test_metric": 1.0}},
    )

    # Check that the prefix was added to all logged metrics
    expected_calls = [
        "test_prefix/Loss/test_loss",
        "test_prefix/Perf/total_fps",
        "test_prefix/Perf/collection_time",
        "test_prefix/Perf/learning_time",
        "test_prefix/Train/num_samples",
        "test_prefix/test_section/test_metric",
        "test_prefix/Episode/test_metric",
    ]

    # Verify all expected calls were made to writer.add_scalar
    actual_calls = [call[0][0] for call in mock_writer.add_scalar.call_args_list]
    for expected in expected_calls:
        assert expected in actual_calls


def test_no_prefix_logging(logging_helper, mock_writer, mock_wandb):
    """Test that logging works correctly without a prefix."""
    # Add episode info
    logging_helper.ep_infos = [{"test_metric": torch.tensor([1.0], device=logging_helper.device)}]

    # Call post_epoch_logging with some test data
    logging_helper.post_epoch_logging(
        it=0,
        loss_dict={"test_loss": 0.5},
        extra_log_dicts={"test_section": {"test_metric": 1.0}},
    )

    # Check that metrics were logged without prefix
    expected_calls = [
        "Loss/test_loss",
        "Perf/total_fps",
        "Perf/collection_time",
        "Perf/learning_time",
        "Train/num_samples",
        "test_section/test_metric",
        "Episode/test_metric",
    ]

    # Verify all expected calls were made to writer.add_scalar
    actual_calls = [call[0][0] for call in mock_writer.add_scalar.call_args_list]
    for expected in expected_calls:
        assert expected in actual_calls


def test_episode_stats_update(logging_helper):
    """Test that episode statistics are properly updated."""
    # Create test data
    rewards = torch.tensor([1.0, 2.0], device=logging_helper.device)
    dones = torch.tensor([1.0, 0.0], device=logging_helper.device)
    infos = {
        "episode": {
            "test_metric": torch.tensor([1.0], device=logging_helper.device),
        },
        "raw_episode": {
            "raw_test_metric": torch.tensor([2.0], device=logging_helper.device),
        },
        "to_log": {
            "env_metric": torch.tensor([2.0], device=logging_helper.device),
        },
    }

    # Update episode stats
    logging_helper.update_episode_stats(rewards, dones, infos)

    # Verify reward buffer was updated
    assert len(logging_helper.rewbuffer) == 1
    assert logging_helper.rewbuffer[0] == 1.0  # First environment's reward

    # Verify length buffer was updated
    assert len(logging_helper.lenbuffer) == 1
    assert logging_helper.lenbuffer[0] == 1.0  # First environment's length

    # Verify episode info was stored
    assert len(logging_helper.ep_infos) == 1
    assert logging_helper.ep_infos[0]["test_metric"].item() == 1.0

    # Verify raw episode info was stored
    assert len(logging_helper.raw_ep_infos) == 1
    assert logging_helper.raw_ep_infos[0]["raw_test_metric"].item() == 2.0


def test_wandb_logging(prefixed_logging_helper, mock_wandb):
    """Test that metrics are properly logged to wandb when available."""
    # Add some episode info to avoid empty list error
    prefixed_logging_helper.ep_infos = [{"test_metric": torch.tensor([1.0], device=prefixed_logging_helper.device)}]
    prefixed_logging_helper.raw_ep_infos = [
        {"raw_test_metric": torch.tensor([2.0], device=prefixed_logging_helper.device)}
    ]
    mock_wandb.run = MagicMock()

    # Call post_epoch_logging with some test data
    prefixed_logging_helper.post_epoch_logging(
        it=0,
        loss_dict={"test_loss": 0.5},
        extra_log_dicts={"test_section": {"test_metric": 1.0}},
    )

    # Verify wandb.log was called with the correct data
    mock_wandb.log.assert_called_once()
    logged_data = mock_wandb.log.call_args[0][0]
    assert "test_prefix/Loss/test_loss" in logged_data
    assert "test_prefix/test_section/test_metric" in logged_data
    assert "test_prefix/Episode/test_metric" in logged_data
    assert "test_prefix/RawEpisode/raw_test_metric" in logged_data
    assert logged_data["global_step"] == 0


def test_reward_group_aliases_are_logged(logging_helper, mock_writer, mock_wandb):
    """Distill reward terms should also appear under grouped W&B Reward panels."""
    logging_helper.ep_infos = [
        {
            "rew_motion_global_ref_position_error_exp": torch.tensor([0.5], device=logging_helper.device),
            "rew_object_global_ref_position_error_exp": torch.tensor([1.0], device=logging_helper.device),
            "rew_offline_contact_guidance": torch.tensor([0.25], device=logging_helper.device),
            "rew_action_rate_l2": torch.tensor([-0.1], device=logging_helper.device),
            "rew_custom_success_bonus": torch.tensor([2.0], device=logging_helper.device),
        }
    ]
    logging_helper.rewbuffer.extend([3.65])
    logging_helper.lenbuffer.extend([42.0])
    mock_wandb.run = MagicMock()

    logging_helper.post_epoch_logging(it=7, loss_dict={}, extra_log_dicts={})

    actual_calls = [call[0][0] for call in mock_writer.add_scalar.call_args_list]
    expected_keys = {
        "Reward/Track/motion_global_ref_position_error_exp",
        "Reward/Object/object_global_ref_position_error_exp",
        "Reward/Contact/offline_contact_guidance",
        "Reward/Regularize/action_rate_l2",
        "Reward/Rest/custom_success_bonus",
        "Reward/Track",
        "Reward/Object",
        "Reward/Contact",
        "Reward/Regularize",
        "Reward/Rest",
        "Reward/total_episode_terms",
        "Reward/mean",
        "Episode Length/mean",
    }
    for expected_key in expected_keys:
        assert expected_key in actual_calls

    logged_data = mock_wandb.log.call_args[0][0]
    assert logged_data["Reward/Track/motion_global_ref_position_error_exp"] == 0.5
    assert logged_data["Reward/Object/object_global_ref_position_error_exp"] == 1.0
    assert logged_data["Reward/Contact/offline_contact_guidance"] == 0.25
    assert logged_data["Reward/Regularize/action_rate_l2"] == pytest.approx(-0.1)
    assert logged_data["Reward/Rest/custom_success_bonus"] == 2.0
    assert logged_data["Reward/total_episode_terms"] == pytest.approx(3.65)
    assert logged_data["Reward/mean"] == pytest.approx(3.65)
    assert logged_data["Episode Length/mean"] == pytest.approx(42.0)


def test_collect_reward_wandb_metadata_groups_weights_and_sigmas():
    reward_cfg = RewardManagerCfg(
        terms={
            "motion_global_ref_position_error_exp": RewardTermCfg(
                func="unused",
                params={"sigma": 0.3},
                weight=0.5,
            ),
            "object_global_ref_position_error_exp": RewardTermCfg(
                func="unused",
                params={"sigma": 0.3},
                weight=1.0,
            ),
            "offline_contact_guidance": RewardTermCfg(
                func="unused",
                params={"position_sigma": 0.08, "force_threshold": 1.4},
                weight=4.0,
            ),
            "action_rate_l2": RewardTermCfg(func="unused", weight=-0.1),
            "custom_success_bonus": RewardTermCfg(func="unused", weight=20.0),
            "custom_zero_reward": RewardTermCfg(func="unused", weight=0.0),
        }
    )

    config_metadata, summary_metadata = collect_reward_wandb_metadata(reward_cfg)
    spec = config_metadata["reward_group_spec"]

    assert spec["Track"]["motion_global_ref_position_error_exp"]["weight"] == 0.5
    assert spec["Track"]["motion_global_ref_position_error_exp"]["sigma"] == 0.3
    assert spec["Object"]["object_global_ref_position_error_exp"]["weight"] == 1.0
    assert spec["Contact"]["offline_contact_guidance"]["force_threshold"] == 1.4
    assert spec["Regularize"]["action_rate_l2"]["weight"] == -0.1
    assert spec["Rest"]["custom_success_bonus"]["weight"] == 20.0
    assert "custom_zero_reward" not in spec["Rest"]
    assert summary_metadata["RewardSpec/Track/motion_global_ref_position_error_exp/weight"] == 0.5
    assert (
        summary_metadata["RewardSpec/Contact/offline_contact_guidance/force_threshold"]
        == 1.4
    )


def test_wandb_hidden_metrics_are_defined(logging_helper, mock_wandb):
    """Housekeeping scalar metrics should be hidden from W&B auto-plots."""
    mock_wandb.run = MagicMock()
    logging_helper.post_epoch_logging(
        it=3,
        loss_dict={"teacher_bc_mask_fraction": 0.25},
        extra_log_dicts={
            "Train": {
                "teacher_action_mix_ratio": 0.8,
            },
            "Eval": {
                "fixed_bc_mu_mse": 0.5,
                "fixed_bc_num_samples": 4096.0,
            },
        },
    )

    define_calls = {
        call.args[0]: call.kwargs
        for call in mock_wandb.define_metric.call_args_list
    }
    assert define_calls["global_step"]["hidden"] is True
    assert define_calls["Loss/teacher_bc_mask_fraction"]["hidden"] is True
    assert define_calls["Train/teacher_action_mix_ratio"]["hidden"] is True
    assert define_calls["Eval/fixed_bc_num_samples"]["hidden"] is True
    assert define_calls["Eval/fixed_bc_mu_mse"]["summary"] == "min"


def test_save_checkpoint_artifact(prefixed_logging_helper, mock_wandb, tmp_path):
    """Test that checkpoints are properly saved and logged to wandb."""
    # Create a temporary directory for the test
    log_dir = tmp_path / "test_logs"
    log_dir.mkdir()
    prefixed_logging_helper.log_dir = str(log_dir)

    # Create test state dict
    state_dict = {"test_param": torch.tensor([1.0])}
    checkpoint_path = log_dir / "checkpoint.pt"

    # Save checkpoint
    prefixed_logging_helper.save_checkpoint_artifact(state_dict, str(checkpoint_path))

    # Verify wandb.save was called with correct path
    mock_wandb.save.assert_called_once()
    assert mock_wandb.save.call_args[0][0] == str(checkpoint_path)
    assert mock_wandb.save.call_args[1]["base_path"] == str(log_dir)
