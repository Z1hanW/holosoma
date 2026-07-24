import pathlib
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from holosoma.agents.modules.logging_utils import LoggingHelper, collect_reward_wandb_metadata
from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg


def _gloo_logging_worker(rank: int, world_size: int, init_file: str, output_dir: str) -> None:
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        helper = LoggingHelper(
            writer=MagicMock(spec=SummaryWriter),
            log_dir=output_dir,
            num_envs=1,
            num_steps_per_env=1,
            num_learning_iterations=1,
            device="cpu",
            is_main_process=rank == 0,
            num_gpus=world_size,
        )
        helper.ep_infos = [{"success": torch.tensor([float(1 + 2 * rank)])}]
        helper.episode_env_tensors.add({"metric": torch.tensor([float(2 + 4 * rank)])})
        helper._completed_rewards_since_sync.append(float(10 + 20 * rank))
        helper._completed_lengths_since_sync.append(float(5 + 4 * rank))
        helper.rewbuffer.append(float(10 + 20 * rank))
        helper.lenbuffer.append(float(5 + 4 * rank))
        helper.rewweightbuffer.append(1.0)
        helper.lenweightbuffer.append(1.0)
        helper.collection_time = float(1 + rank)
        helper.learn_time = float(2 - 0.5 * rank)
        loss_weight = 1.5 if rank == 0 else 0.5
        merged_loss = helper.synchronize_distributed_metrics(
            {"Value": float(2 + 4 * rank)},
            loss_weight=loss_weight,
            process_group=torch.distributed.group.WORLD,
        )
        result = {
            "merged_loss": merged_loss,
            "weight_sum": helper.distributed_loss_weight_sum,
            "episode_info_count": len(helper.ep_infos),
            "reward_mean": helper._mean_reward() if rank == 0 else None,
            "length_mean": helper._mean_episode_length() if rank == 0 else None,
            "env_mean": (float(helper.episode_env_tensors.mean()["metric"].item()) if rank == 0 else None),
            "collection_time": helper.collection_time,
            "learn_time": helper.learn_time,
        }
        torch.save(result, f"{output_dir}/rank_{rank}.pt")
    finally:
        torch.distributed.destroy_process_group()


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


def test_console_output_keeps_learning_rate_floor_visible(logging_helper):
    logging_helper.collection_time = 1.0
    logging_helper.learn_time = 1.0

    output = logging_helper._create_console_output(
        it=700,
        loss_dict={"actor_learning_rate": 1.0e-5, "critic_learning_rate": 1.0e-3},
        env_log_dict={},
        extra_log_dicts={},
        ep_string="",
        width=80,
        pad=35,
        iteration_time=2.0,
        fps=10,
    )

    assert "actor_learning_rate: 1.000e-05" in output
    assert "critic_learning_rate: 1.000e-03" in output
    assert "actor_learning_rate: 0.0000" not in output


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


def test_episode_stats_reuses_reset_ids_and_keeps_later_nonempty_episode_metrics(
    logging_helper,
    monkeypatch,
):
    def forbid_nonzero(*_args, **_kwargs):
        raise AssertionError("published reset_env_ids must avoid a second nonzero")

    monkeypatch.setattr(torch.Tensor, "nonzero", forbid_nonzero)
    empty_infos = {
        "episode": {},
        "raw_episode": {},
        "reset_env_ids": torch.empty(0, dtype=torch.long, device=logging_helper.device),
        "to_log": {},
    }
    logging_helper.update_episode_stats(
        torch.tensor([1.0, 2.0], device=logging_helper.device),
        torch.zeros(2, device=logging_helper.device),
        empty_infos,
    )

    assert logging_helper.ep_infos == []
    assert logging_helper.raw_ep_infos == []
    assert list(logging_helper.rewbuffer) == []

    reset_infos = {
        "episode": {"success": torch.tensor([1.0], device=logging_helper.device)},
        "raw_episode": {"raw_success": torch.tensor([2.0], device=logging_helper.device)},
        "reset_env_ids": torch.tensor([1], dtype=torch.long, device=logging_helper.device),
        "to_log": {},
    }
    logging_helper.update_episode_stats(
        torch.tensor([3.0, 4.0], device=logging_helper.device),
        torch.tensor([0.0, 1.0], device=logging_helper.device),
        reset_infos,
    )

    assert len(logging_helper.ep_infos) == 1
    assert logging_helper.ep_infos[0]["success"].item() == 1.0
    assert len(logging_helper.raw_ep_infos) == 1
    assert logging_helper.raw_ep_infos[0]["raw_success"].item() == 2.0
    assert list(logging_helper.rewbuffer) == [6.0]
    assert list(logging_helper.lenbuffer) == [2.0]


def _historical_tensor_dict_summary(items):
    summary = {}
    for item in items:
        for key, raw_value in item.items():
            value = raw_value.detach() if isinstance(raw_value, torch.Tensor) else torch.as_tensor(raw_value)
            value = value.to(dtype=torch.float64).reshape(-1)
            if value.numel() == 0:
                continue
            old_sum, old_count = summary.get(str(key), (0.0, 0))
            summary[str(key)] = (
                old_sum + float(value.sum().item()),
                old_count + int(value.numel()),
            )
    return summary


def _historical_env_tensor_summary(helper):
    summary = {}
    for key, meter in helper.episode_env_tensors.data.items():
        for raw_value in meter.tensors:
            value = raw_value.detach().to(dtype=torch.float64).reshape(-1)
            if value.numel() == 0:
                continue
            old_sum, old_count = summary.get(str(key), (0.0, 0))
            summary[str(key)] = (
                old_sum + float(value.sum().item()),
                old_count + int(value.numel()),
            )
    return summary


def test_tensor_metric_summaries_bulk_copy_preserves_exact_order_schema_and_counts(
    logging_helper,
):
    large = float(2**53)
    logging_helper.ep_infos = [
        {
            "cancellation": torch.tensor([large], dtype=torch.float64),
            "shared": torch.tensor([0.1, 0.2], dtype=torch.float32),
            "empty": torch.empty(0),
        },
        {
            "late": 3.25,
            "cancellation": torch.tensor([1.0], dtype=torch.float64),
        },
        {
            "cancellation": torch.tensor([-large], dtype=torch.float64),
            "shared": torch.tensor([0.3], dtype=torch.float64),
        },
    ]
    logging_helper.raw_ep_infos = [
        {"raw": torch.tensor([1.5, -0.25], dtype=torch.float32)},
        {"raw": 2, "raw_late": torch.tensor(7.0)},
    ]
    logging_helper.episode_env_tensors.add(
        {
            "env_first": torch.tensor([1.0, 2.0]),
            "env_empty": torch.empty(0),
        }
    )
    logging_helper.episode_env_tensors.add(
        {
            "env_first": torch.tensor([4.0]),
            "env_late": torch.tensor([5.0, 6.0]),
        }
    )

    expected = {
        "episode": _historical_tensor_dict_summary(logging_helper.ep_infos),
        "raw_episode": _historical_tensor_dict_summary(logging_helper.raw_ep_infos),
        "env": _historical_env_tensor_summary(logging_helper),
    }
    original_cpu = torch.Tensor.cpu
    cpu_calls = []

    def counted_cpu(tensor, *args, **kwargs):
        cpu_calls.append((tensor.device, tuple(tensor.shape), tensor.dtype))
        return original_cpu(tensor, *args, **kwargs)

    with (
        patch.object(torch.Tensor, "cpu", counted_cpu),
        patch.object(
            torch.Tensor,
            "item",
            side_effect=AssertionError("summary hot path must not call Tensor.item()"),
        ),
    ):
        actual = logging_helper._summarize_iteration_tensors()

    assert actual == expected
    assert [*actual] == ["episode", "raw_episode", "env"]
    assert [*actual["episode"]] == ["cancellation", "shared", "late"]
    assert [*actual["raw_episode"]] == ["raw", "raw_late"]
    assert [*actual["env"]] == ["env_first", "env_late"]
    assert actual["episode"]["cancellation"] == (0.0, 3)
    assert actual["episode"]["shared"][1] == 3
    assert actual["raw_episode"]["raw"][1] == 3
    assert actual["env"]["env_first"][1] == 3
    assert "empty" not in actual["episode"]
    assert "env_empty" not in actual["env"]
    assert cpu_calls == [(torch.device("cpu"), (12,), torch.float64)]

    # Keep the private category helpers behaviorally identical for callers that
    # summarize only one category rather than a complete iteration payload.
    assert logging_helper._summarize_tensor_dicts(logging_helper.ep_infos) == expected["episode"]
    assert logging_helper._summarize_env_tensors() == expected["env"]


def test_distributed_metric_sync_uses_global_counts_and_one_shot_episode_deltas(logging_helper):
    logging_helper.ep_infos = [{"success": torch.tensor([1.0, 0.0])}]
    logging_helper.raw_ep_infos = [{"raw": torch.tensor([2.0])}]
    logging_helper.episode_env_tensors.add({"contact_mass": torch.tensor([0.2, 0.4])})
    logging_helper.rewbuffer.extend([1.0])
    logging_helper.lenbuffer.extend([5.0])
    logging_helper.rewweightbuffer.extend([1.0])
    logging_helper.lenweightbuffer.extend([1.0])
    logging_helper._completed_rewards_since_sync.extend([1.0])
    logging_helper._completed_lengths_since_sync.extend([5.0])
    logging_helper.collection_time = 2.0
    logging_helper.learn_time = 1.0

    remote_payload = {
        "loss_dict": {"Value": 3.0},
        "loss_weight": 0.5,
        "episode": {"success": (2.0, 2)},
        "raw_episode": {"raw": (8.0, 2)},
        "env": {"contact_mass": (1.8, 2)},
        "completed_reward_sum": 7.0,
        "completed_length_sum": 9.0,
        "completed_episode_count": 1,
        "collection_time": 3.0,
        "learn_time": 0.5,
    }

    def fake_gather(gathered, payload, group):
        gathered[0] = payload
        gathered[1] = remote_payload

    with (
        patch("torch.distributed.is_available", return_value=True),
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=2),
        patch("torch.distributed.get_rank", return_value=0),
        patch("torch.distributed.get_backend", return_value="gloo"),
        patch("torch.distributed.all_gather_object", side_effect=fake_gather),
    ):
        merged_loss = logging_helper.synchronize_distributed_metrics({"Value": 1.0}, loss_weight=1.5)

    assert merged_loss["Value"] == pytest.approx((1.0 * 1.5 + 3.0 * 0.5) / 2.0)
    assert logging_helper.ep_infos[0]["success"].item() == pytest.approx((1.0 * 1.5 + 2.0 * 0.5) / 4.0)
    assert logging_helper.raw_ep_infos[0]["raw"].item() == pytest.approx((2.0 * 1.5 + 8.0 * 0.5) / 2.5)
    assert logging_helper.episode_env_tensors.mean()["contact_mass"].item() == pytest.approx(
        (0.6 * 1.5 + 1.8 * 0.5) / 4.0
    )
    # Rank-local public buffers are no longer concatenated in rank order; the
    # exact weighted global batch drives the rolling distributed metric.
    assert list(logging_helper.rewbuffer) == [1.0]
    assert list(logging_helper.lenbuffer) == [5.0]
    assert logging_helper._mean_reward() == pytest.approx(2.5)
    assert logging_helper._mean_episode_length() == pytest.approx(6.0)
    assert logging_helper.distributed_loss_weight_sum == pytest.approx(2.0)
    assert logging_helper.distributed_effective_episode_count == pytest.approx(2.0)
    assert logging_helper.collection_time == pytest.approx(3.0)
    assert logging_helper.learn_time == pytest.approx(0.5)
    assert logging_helper._completed_rewards_since_sync == []
    assert logging_helper._completed_lengths_since_sync == []


def test_distributed_metric_sync_refuses_nccl_object_collective(logging_helper):
    with (
        patch("torch.distributed.is_available", return_value=True),
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=2),
        patch("torch.distributed.get_backend", return_value="nccl"),
        patch("torch.distributed.all_gather_object") as gather,
    ):
        with pytest.raises(RuntimeError, match="requires a Gloo process group"):
            logging_helper.synchronize_distributed_metrics({"Value": 1.0})
    gather.assert_not_called()


def test_distributed_metric_sync_rejects_weight_sum_that_changes_gradient_scale(logging_helper):
    def fake_gather(gathered, payload, group):
        gathered[0] = payload
        gathered[1] = dict(payload, loss_weight=0.25)

    with (
        patch("torch.distributed.is_available", return_value=True),
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=2),
        patch("torch.distributed.get_rank", return_value=0),
        patch("torch.distributed.get_backend", return_value="gloo"),
        patch("torch.distributed.all_gather_object", side_effect=fake_gather),
    ):
        with pytest.raises(RuntimeError, match="must sum to world_size"):
            logging_helper.synchronize_distributed_metrics({"Value": 1.0}, loss_weight=1.0)


def test_distributed_episode_window_is_not_biased_by_rank_append_order(logging_helper):
    payloads = [
        {
            "loss_weight": 1.0,
            "completed_reward_sum": 100.0,
            "completed_length_sum": 1_000.0,
            "completed_episode_count": 100,
        },
        {
            "loss_weight": 1.0,
            "completed_reward_sum": 900.0,
            "completed_length_sum": 3_000.0,
            "completed_episode_count": 100,
        },
    ]

    logging_helper._append_distributed_episode_batch(payloads)

    assert logging_helper.distributed_effective_episode_count == pytest.approx(100.0)
    assert logging_helper._mean_reward() == pytest.approx(5.0)
    assert logging_helper._mean_episode_length() == pytest.approx(20.0)

    before = list(logging_helper._distributed_episode_batches)
    logging_helper._append_distributed_episode_batch(
        [
            {
                "loss_weight": 1.0,
                "completed_reward_sum": 0.0,
                "completed_length_sum": 0.0,
                "completed_episode_count": 0,
            }
        ]
    )
    assert list(logging_helper._distributed_episode_batches) == before


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed is unavailable")
def test_distributed_metric_sync_completes_on_real_two_rank_gloo(tmp_path):
    init_file = tmp_path / "gloo_init"
    mp.start_processes(
        _gloo_logging_worker,
        args=(2, str(init_file), str(tmp_path)),
        nprocs=2,
        join=True,
        start_method="fork",
    )

    rank0 = torch.load(tmp_path / "rank_0.pt", weights_only=True)
    rank1 = torch.load(tmp_path / "rank_1.pt", weights_only=True)
    assert rank0["merged_loss"]["Value"] == pytest.approx(3.0)
    assert rank1["merged_loss"]["Value"] == pytest.approx(3.0)
    assert rank0["weight_sum"] == pytest.approx(2.0)
    assert rank1["weight_sum"] == pytest.approx(2.0)
    assert rank0["reward_mean"] == pytest.approx(15.0)
    assert rank0["length_mean"] == pytest.approx(6.0)
    assert rank0["env_mean"] == pytest.approx(3.0)
    assert rank0["episode_info_count"] == 1
    assert rank1["episode_info_count"] == 0
    # Rank 1 has the slowest total time (2.0 + 1.5 seconds).
    assert rank0["collection_time"] == pytest.approx(2.0)
    assert rank0["learn_time"] == pytest.approx(1.5)


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
    assert summary_metadata["RewardSpec/Contact/offline_contact_guidance/force_threshold"] == 1.4


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
                "fixed_bc_final_mu_mse": 0.5,
                "fixed_bc_num_samples": 4096.0,
                "fixed_bc_terminal_observation": 1.0,
                "fixed_bc_scheduled_evaluation": 0.0,
                "fixed_bc_guard_applied": 0.0,
                "fixed_bc_guard_reference_min_mu_mse": 0.08,
                "fixed_bc_guard_effective_threshold_mu_mse": 0.16,
                "fixed_bc_guard_consecutive_exceedances": 1.0,
                "fixed_bc_guard_last_mu_mse": 0.17,
            },
        },
    )

    define_calls = {call.args[0]: call.kwargs for call in mock_wandb.define_metric.call_args_list}
    assert define_calls["global_step"]["hidden"] is True
    assert define_calls["Loss/teacher_bc_mask_fraction"]["hidden"] is True
    assert define_calls["Train/teacher_action_mix_ratio"]["hidden"] is True
    assert define_calls["Eval/fixed_bc_num_samples"]["hidden"] is True
    assert define_calls["Eval/fixed_bc_terminal_observation"]["hidden"] is True
    assert define_calls["Eval/fixed_bc_scheduled_evaluation"]["hidden"] is True
    assert define_calls["Eval/fixed_bc_guard_applied"]["hidden"] is True
    assert define_calls["Eval/fixed_bc_mu_mse"]["summary"] == "min"
    assert define_calls["Eval/fixed_bc_final_mu_mse"]["summary"] == "last"
    assert define_calls["Eval/fixed_bc_guard_reference_min_mu_mse"]["summary"] == "min"
    assert define_calls["Eval/fixed_bc_guard_effective_threshold_mu_mse"]["summary"] == "last"
    assert define_calls["Eval/fixed_bc_guard_consecutive_exceedances"]["summary"] == "last"
    assert define_calls["Eval/fixed_bc_guard_last_mu_mse"]["summary"] == "last"


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
    assert torch.equal(torch.load(checkpoint_path, weights_only=False)["test_param"], state_dict["test_param"])
    assert not list(log_dir.glob(".checkpoint.pt.*.tmp"))


def test_failed_checkpoint_serialization_preserves_published_file(
    prefixed_logging_helper,
    mock_wandb,
    tmp_path,
):
    """A partial temporary write must never replace the last good checkpoint."""
    log_dir = tmp_path / "test_logs"
    log_dir.mkdir()
    prefixed_logging_helper.log_dir = str(log_dir)
    checkpoint_path = log_dir / "checkpoint.pt"
    checkpoint_path.write_bytes(b"previous-valid-checkpoint")

    def fail_after_partial_write(_state_dict, temp_path):
        pathlib.Path(temp_path).write_bytes(b"partial")
        raise OSError("injected serialization failure")

    with patch(
        "holosoma.agents.modules.logging_utils.torch.save",
        side_effect=fail_after_partial_write,
    ), pytest.raises(OSError, match="injected serialization failure"):
        prefixed_logging_helper.save_checkpoint_artifact({"value": 1}, str(checkpoint_path))

    assert checkpoint_path.read_bytes() == b"previous-valid-checkpoint"
    assert not list(log_dir.glob(".checkpoint.pt.*.tmp"))
    mock_wandb.save.assert_not_called()


def test_checkpoint_path_uses_real_directory_boundary(prefixed_logging_helper, tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    prefixed_logging_helper.log_dir = str(log_dir)

    with pytest.raises(ValueError, match="not in the logging directory"):
        prefixed_logging_helper.save_checkpoint_artifact(
            {"value": 1},
            str(tmp_path / "logs-other" / "checkpoint.pt"),
        )
