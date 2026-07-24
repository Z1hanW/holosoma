from __future__ import annotations

import base64
import hashlib
import importlib
import json
import sys
import warnings
from types import ModuleType
from types import SimpleNamespace

import pytest
import torch
import tyro

import holosoma.config_values.perception as perception_values
import holosoma.utils.sim_utils as sim_utils
from holosoma.config_types.randomization import RandomizationManagerCfg, RandomizationTermCfg
from holosoma.config_types.run_sim import RunSimConfig
from holosoma.managers.perception.manager import PerceptionManager
from holosoma.managers.randomization import RandomizationManager
from holosoma.managers.randomization.terms.locomotion import randomize_camera_raycast
from holosoma.utils.sim_utils import DirectSimulation
from holosoma.utils.tyro_utils import TYRO_CONIFG


class _PerceptionRecorder:
    def __init__(self, events: list[str] | None = None) -> None:
        self.events = [] if events is None else events
        self.update_calls: list[torch.Tensor | None] = []
        self.reset_calls: list[torch.Tensor] = []

    def update(self, env_ids: torch.Tensor | None = None) -> None:
        self.events.append("perception.update")
        self.update_calls.append(None if env_ids is None else env_ids.detach().cpu().clone())

    def reset(self, env_ids: torch.Tensor) -> None:
        self.events.append("perception.reset")
        self.reset_calls.append(env_ids.detach().cpu().clone())

    @staticmethod
    def get_obs() -> torch.Tensor:
        return torch.tensor([[1.0, 2.0]], dtype=torch.float32)

    @staticmethod
    def uses_legacy_full_reset_refresh() -> bool:
        return False


def _scheduled_direct_simulation() -> tuple[DirectSimulation, SimpleNamespace, _PerceptionRecorder]:
    direct = object.__new__(DirectSimulation)
    simulator = SimpleNamespace(completed_physics_steps=0, refresh_calls=0)

    def refresh() -> None:
        simulator.refresh_calls += 1

    simulator.refresh_sim_tensors = refresh
    manager = _PerceptionRecorder()
    direct.simulator = simulator
    direct._perception_manager = manager
    direct._perception_producer_steps = 40
    direct._perception_last_update_completed_steps = 0
    direct._perception_publish_pending = True
    return direct, simulator, manager


def test_direct_perception_advances_only_on_completed_training_ticks() -> None:
    direct, simulator, manager = _scheduled_direct_simulation()

    assert direct._get_split_sim_perception_obs() == [1.0, 2.0]
    # The already-computed reset frame is retransmitted without another RNG
    # draw until physics advances, covering a late ZMQ subscriber.
    assert direct._get_split_sim_perception_obs() == [1.0, 2.0]

    simulator.completed_physics_steps = 39
    assert direct._get_split_sim_perception_obs() is None
    simulator.completed_physics_steps = 40
    assert direct._get_split_sim_perception_obs() == [1.0, 2.0]
    assert direct._get_split_sim_perception_obs() is None

    assert simulator.refresh_calls == 1
    assert manager.update_calls == [None]
    assert direct._perception_last_update_completed_steps == 40


def test_direct_perception_fails_if_a_producer_tick_is_skipped() -> None:
    direct, simulator, _ = _scheduled_direct_simulation()
    direct._perception_publish_pending = False
    simulator.completed_physics_steps = 80

    with pytest.raises(RuntimeError, match="skipped an authenticated producer tick"):
        direct._get_split_sim_perception_obs()


def test_direct_simulation_step_failure_propagates_to_process_status(monkeypatch) -> None:
    direct = object.__new__(DirectSimulation)
    direct.config = SimpleNamespace(
        simulator=SimpleNamespace(config=SimpleNamespace(sim=SimpleNamespace(fps=500))),
        training=SimpleNamespace(headless=True),
        viewer_dt=0.02,
    )
    direct.simulator = SimpleNamespace(
        simulate_at_each_physics_step=lambda: (_ for _ in ()).throw(
            RuntimeError("authenticated producer failure")
        )
    )
    direct._perception_manager = None
    direct._calculate_viewer_steps = lambda: 10

    monkeypatch.setattr(sim_utils, "get_simulator_type", lambda: sim_utils.SimulatorType.MUJOCO)
    monkeypatch.setattr(
        sim_utils,
        "RateLimiter",
        lambda _frequency: SimpleNamespace(sleep=lambda: None),
    )

    with pytest.raises(RuntimeError, match="authenticated producer failure"):
        direct.run()


def test_direct_episode_reset_matches_training_manager_order() -> None:
    events: list[str] = []
    direct = object.__new__(DirectSimulation)
    direct.device = "cpu"
    direct._perception_env_proxy = SimpleNamespace(num_envs=1)
    direct._perception_manager = _PerceptionRecorder(events)
    direct._perception_randomization_manager = SimpleNamespace(
        reset=lambda env_ids: events.append("randomization.reset")
    )
    direct.simulator = SimpleNamespace(completed_physics_steps=0)
    direct.simulator.refresh_sim_tensors = lambda: events.append("simulator.refresh")
    direct._perception_last_update_completed_steps = 123
    direct._perception_publish_pending = False

    direct._reset_split_sim_perception()

    assert events == [
        "perception.reset",
        "randomization.reset",
        "simulator.refresh",
        "perception.update",
    ]
    assert direct._perception_last_update_completed_steps == 0
    assert direct._perception_publish_pending is True


def test_direct_initialization_replays_training_control_tick_and_sensor_phase() -> None:
    events: list[str] = []
    direct = object.__new__(DirectSimulation)
    direct.device = "cpu"
    direct._perception_env_proxy = SimpleNamespace(num_envs=1)
    direct._perception_manager = _PerceptionRecorder(events)
    direct._perception_randomization_manager = SimpleNamespace(
        reset=lambda env_ids: events.append("randomization.reset")
    )
    direct._perception_producer_steps = 2
    simulator = SimpleNamespace(completed_physics_steps=0, bridge=None)

    def refresh() -> None:
        events.append("simulator.refresh")

    def simulate() -> None:
        events.append("simulator.step")
        simulator.completed_physics_steps += 1

    simulator.refresh_sim_tensors = refresh
    simulator.simulate_at_each_physics_step = simulate
    direct.simulator = simulator
    direct._perception_last_update_completed_steps = None
    direct._perception_publish_pending = False

    direct._reset_split_sim_perception(initialization_warmup=True)

    assert events == [
        "perception.reset",
        "randomization.reset",
        "simulator.refresh",
        "simulator.step",
        "simulator.refresh",
        "simulator.step",
        "simulator.refresh",
        "perception.update",
        "simulator.refresh",
        "perception.update",
    ]
    assert direct._perception_manager.update_calls[0] is None
    assert torch.equal(direct._perception_manager.update_calls[1], torch.tensor([0]))
    assert direct._perception_last_update_completed_steps == 2
    assert direct._perception_publish_pending is True


def test_initialization_warmup_isolates_bridge_and_realigns_control_phase() -> None:
    events: list[str] = []
    direct = object.__new__(DirectSimulation)
    direct._perception_producer_steps = 3
    simulator = SimpleNamespace(completed_physics_steps=0)
    bridge = SimpleNamespace(_zmq_lowcmd_substep=2)

    def reset_control_phase() -> None:
        bridge._zmq_lowcmd_substep = 0
        events.append("bridge.reset_control_phase")

    def apply_initialization_hold() -> None:
        assert simulator.bridge is None
        bridge._zmq_lowcmd_substep = 0
        events.append("bridge.nominal_hold")

    def refresh() -> None:
        events.append("simulator.refresh")

    def simulate() -> None:
        assert simulator.bridge is None
        simulator.completed_physics_steps += 1
        events.append("simulator.step")

    bridge.reset_control_phase = reset_control_phase
    bridge.apply_initialization_default_pose_hold = apply_initialization_hold
    simulator.bridge = bridge
    simulator.refresh_sim_tensors = refresh
    simulator.simulate_at_each_physics_step = simulate
    direct.simulator = simulator

    direct._advance_initialization_perception_warmup()

    assert simulator.bridge is bridge
    assert bridge._zmq_lowcmd_substep == 0
    assert events == [
        "bridge.reset_control_phase",
        "simulator.refresh",
        "bridge.nominal_hold",
        "simulator.step",
        "simulator.refresh",
        "bridge.nominal_hold",
        "simulator.step",
        "simulator.refresh",
        "bridge.nominal_hold",
        "simulator.step",
        "bridge.reset_control_phase",
    ]


def test_direct_runtime_rejects_legacy_vectorized_reset_contract() -> None:
    manager = object.__new__(PerceptionManager)
    manager._reset_refresh_semantics = "legacy_full_v1"

    with pytest.raises(ValueError, match="targeted_v2"):
        manager.authenticate_observation_contract(
            {
                "version": 2,
                "producer_lifecycle": {
                    "reset_refresh_semantics": "legacy_full_v1"
                },
            },
            declared_sha256="0" * 64,
        )


def test_mujoco_physical_reset_realigns_bridge_before_perception(monkeypatch) -> None:
    mujoco_package = importlib.import_module("mujoco")
    viewer_module = ModuleType("mujoco.viewer")
    monkeypatch.setitem(sys.modules, "mujoco.viewer", viewer_module)
    monkeypatch.setattr(mujoco_package, "viewer", viewer_module, raising=False)
    monkeypatch.setitem(sys.modules, "glfw", ModuleType("glfw"))
    mujoco_module = importlib.import_module(
        "holosoma.simulator.mujoco.mujoco"
    )
    MuJoCo = mujoco_module.MuJoCo
    events: list[str] = []
    simulator = object.__new__(MuJoCo)
    simulator.completed_physics_steps = 17
    simulator.root_model = object()
    simulator.root_data = object()
    simulator.virtual_gantry = None
    simulator.bridge = SimpleNamespace(
        reset_episode_state=lambda: events.append("bridge.reset")
    )
    simulator._set_robot_initial_state = lambda: events.append("state.reset")
    simulator._zero_commands = lambda: events.append("commands.zero")
    simulator._has_registered_dynamic_objects = lambda: False

    def reset_perception() -> None:
        assert simulator.completed_physics_steps == 0
        events.append("perception.reset")

    simulator._reset_split_sim_perception_provider = reset_perception
    monkeypatch.setattr(
        mujoco_module,
        "mujoco",
        SimpleNamespace(
            mj_resetData=lambda model, data: events.append("physics.reset"),
            mj_forward=lambda model, data: events.append("physics.forward"),
        ),
    )

    simulator.reset()

    assert events == [
        "bridge.reset",
        "physics.reset",
        "state.reset",
        "commands.zero",
        "physics.forward",
        "perception.reset",
    ]


def test_bridge_reset_request_holds_old_episode_and_resets_phase() -> None:
    from holosoma.simulator.shared.simulator_bridge import SimulatorBridge

    bridge = object.__new__(SimulatorBridge)
    bridge.simulator = SimpleNamespace(_pending_reset=True)
    bridge._use_zmq_lowcmd = True
    bridge._latest_lowcmd_payload = {"seq": 1}
    bridge._pending_lowcmd_payload = {"seq": 2}
    bridge._last_applied_lowcmd_seq = 1
    bridge._received_external_active_command = True
    bridge._logged_first_command_summary = True
    bridge._logged_active_command_summaries = 3
    bridge._logged_default_pose_hold = True
    bridge._logged_initial_pose_hold = True
    bridge._logged_rejected_lowcmd_generation = True
    bridge._initial_hold_q = object()
    bridge._zmq_lowcmd_substep = 7
    bridge._zmq_lowcmd_hold_physics = False
    bridge._last_zmq_torque_preview = [1.0]
    bridge._episode_generation = 4
    invalidations: list[str] = []
    bridge.perception_obs_shm_pub = SimpleNamespace(
        invalidate=lambda: invalidations.append("perception.invalidate")
    )

    assert bridge.should_hold_physics() is True
    bridge.reset_episode_state()

    assert bridge._zmq_lowcmd_substep == 0
    assert bridge._latest_lowcmd_payload is None
    assert bridge._pending_lowcmd_payload is None
    assert bridge._received_external_active_command is False
    assert bridge._episode_generation == 5
    assert invalidations == ["perception.invalidate"]


def test_bridge_rejects_queued_lowcmd_from_previous_episode_generation() -> None:
    from holosoma.simulator.shared.simulator_bridge import SimulatorBridge

    bridge = object.__new__(SimulatorBridge)
    bridge.simulator = SimpleNamespace(num_dof=2)
    bridge.bridge_config = SimpleNamespace(log_first_command_summary=False)
    bridge._use_zmq_lowcmd = True
    bridge._episode_generation = 7
    bridge._latest_lowcmd_payload = None
    bridge._pending_lowcmd_payload = None
    bridge._last_applied_lowcmd_seq = None
    bridge._received_external_active_command = False
    bridge._reset_perception_on_first_lowcmd = False
    bridge._logged_first_command_summary = False
    bridge._logged_active_command_summaries = 0
    bridge._logged_rejected_lowcmd_generation = False
    bridge._zmq_lowcmd_lockstep_control_boundary = False
    bridge._zmq_lowcmd_latch_control_boundary = False

    stale = {
        "action": "lowcmd",
        "seq": 10,
        "episode_generation": 6,
        "kp": [1.0, 1.0],
    }
    bridge._queue_lowcmd_payload(stale, lowcmd_boundary=True)
    assert bridge._latest_lowcmd_payload is None

    missing_generation = {"action": "lowcmd", "seq": 11, "kp": [1.0, 1.0]}
    bridge._queue_lowcmd_payload(missing_generation, lowcmd_boundary=True)
    assert bridge._latest_lowcmd_payload is None

    current = {
        "action": "lowcmd",
        "seq": 12,
        "episode_generation": 7,
        "kp": [1.0, 1.0],
    }
    bridge._queue_lowcmd_payload(current, lowcmd_boundary=True)
    assert bridge._latest_lowcmd_payload is current
    assert bridge._received_external_active_command is True


def test_bridge_publishes_same_episode_identity_on_both_perception_transports() -> None:
    from holosoma.simulator.shared.simulator_bridge import SimulatorBridge

    class Recorder:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple, dict]] = []

        def publish(self, *args, **kwargs) -> None:
            self.calls.append((args, kwargs))

    zmq_pub = Recorder()
    shm_pub = Recorder()
    contract_sha256 = "12" * 32
    bridge = object.__new__(SimulatorBridge)
    bridge.simulator = SimpleNamespace(
        time=lambda: 0.0,
        _split_sim_perception_provider=lambda: [1.0, 2.0],
        _split_sim_perception_contract_provider=lambda: contract_sha256,
    )
    bridge.perception_obs_pub = zmq_pub
    bridge.perception_obs_shm_pub = shm_pub
    bridge._logged_perception_obs_publish = False
    bridge._episode_generation = 987654

    bridge._publish_perception_obs()

    assert len(zmq_pub.calls) == 1
    assert zmq_pub.calls[0][0][0]["episode_generation"] == 987654
    assert zmq_pub.calls[0][0][0]["sim_time_ms"] == 0
    assert len(shm_pub.calls) == 1
    assert shm_pub.calls[0][1]["episode_generation"] == 987654
    assert shm_pub.calls[0][1]["sim_time_ms"] == 0


def test_bridge_episode_identity_has_a_random_per_process_base(monkeypatch) -> None:
    from holosoma.config_types.simulator import BridgeConfig
    from holosoma.simulator.shared import simulator_bridge as bridge_module

    monkeypatch.setattr(bridge_module.secrets, "randbits", lambda bits: 123456789)
    bridge = bridge_module.SimulatorBridge(
        SimpleNamespace(),
        BridgeConfig(enabled=False, interface="lo"),
    )

    assert bridge._episode_generation == 123456789


def test_real_randomization_manager_samples_all_direct_camera_state() -> None:
    manager_cfg = RandomizationManagerCfg(
        reset_terms={
            "camera": RandomizationTermCfg(
                func=(
                    "holosoma.managers.randomization.terms.locomotion:"
                    "randomize_camera_raycast"
                ),
                params={
                    "enabled": True,
                    "translation_range": {
                        "x": [-0.02, 0.02],
                        "y": [-0.01, 0.01],
                        "z": [-0.03, 0.03],
                    },
                    "rotation_range_deg": {
                        "roll": [-2.0, 2.0],
                        "pitch": [-3.0, 3.0],
                        "yaw": [-4.0, 4.0],
                    },
                    "noise_std_mult_range": [0.01, 0.05],
                    "noise_drop_prob_range": [0.005, 0.025],
                },
            )
        }
    )
    env = SimpleNamespace(
        device="cpu",
        num_envs=8,
        logger=None,
        perception_manager=SimpleNamespace(
            enabled=True,
            cfg=SimpleNamespace(output_mode="camera_depth", camera_source="far_tracking_warp"),
        ),
    )
    randomization = RandomizationManager(manager_cfg, env, "cpu")
    env.randomization_manager = randomization
    env_ids = torch.arange(env.num_envs)

    torch.manual_seed(7)
    randomization.reset(env_ids)

    assert env._perception_camera_offset_pos.shape == (8, 3)
    assert env._perception_camera_offset_rpy.shape == (8, 3)
    assert env._perception_camera_offset_quat.shape == (8, 4)
    assert torch.allclose(
        torch.linalg.vector_norm(env._perception_camera_offset_quat, dim=-1),
        torch.ones(8),
        atol=1.0e-6,
    )
    assert bool((env._perception_camera_noise_std_mult >= 0.01).all())
    assert bool((env._perception_camera_noise_std_mult <= 0.05).all())
    assert bool((env._perception_camera_noise_drop_prob >= 0.005).all())
    assert bool((env._perception_camera_noise_drop_prob <= 0.025).all())

    first_sample = (
        env._perception_camera_offset_pos.clone(),
        env._perception_camera_offset_rpy.clone(),
        env._perception_camera_noise_std_mult.clone(),
        env._perception_camera_noise_drop_prob.clone(),
    )
    randomization.reset(env_ids)
    second_sample = (
        env._perception_camera_offset_pos,
        env._perception_camera_offset_rpy,
        env._perception_camera_noise_std_mult,
        env._perception_camera_noise_drop_prob,
    )
    assert all(not torch.equal(left, right) for left, right in zip(first_sample, second_sample, strict=True))


def test_camera_randomization_seed_authenticates_distribution_not_batch_sample_path() -> None:
    def sample(num_envs: int):
        env = SimpleNamespace(
            device="cpu",
            num_envs=num_envs,
            perception_manager=SimpleNamespace(
                enabled=True,
                cfg=SimpleNamespace(output_mode="camera_depth", camera_source="far_tracking_warp"),
            ),
        )
        torch.manual_seed(42)
        randomize_camera_raycast(
            env,
            torch.arange(num_envs),
            translation_range={axis: [-0.02, 0.02] for axis in ("x", "y", "z")},
            rotation_range_deg={axis: [-3.0, 3.0] for axis in ("roll", "pitch", "yaw")},
            noise_std_mult_range=[0.01, 0.05],
            noise_drop_prob_range=[0.005, 0.025],
        )
        return (
            env._perception_camera_offset_pos[0].clone(),
            env._perception_camera_offset_rpy[0].clone(),
            env._perception_camera_noise_std_mult[0].clone(),
            env._perception_camera_noise_drop_prob[0].clone(),
        )

    one_env = sample(1)
    training_batch = sample(2048)

    # The first translation is a prefix of the same torch.rand call, but all
    # later component calls begin at different global-RNG offsets.  This locks
    # in the honest distribution-only contract for direct one-env evaluation.
    assert torch.equal(one_env[0], training_batch[0])
    assert all(
        not torch.equal(one_env[index], training_batch[index])
        for index in (1, 2, 3)
    )


def test_run_sim_cli_carries_structured_direct_camera_contract() -> None:
    envelope_b64 = base64.b64encode(
        b'{"contract":{"version":2},"sha256":"00"}'
    ).decode("ascii")
    parsed = tyro.cli(
        RunSimConfig,
        args=[
            "--perception-producer-tick-dt",
            "0.02",
            "--perception-allow-mujoco-noise",
            "True",
            "--perception-contract-envelope-b64",
            envelope_b64,
            "--perception-randomization.enabled",
            "True",
            "--perception-randomization.translation-range",
            '{"x":[-0.02,0.02],"y":[-0.01,0.01],"z":[-0.03,0.03]}',
            "--perception-randomization.rotation-range-deg",
            '{"roll":[-2,2],"pitch":[-3,3],"yaw":[-4,4]}',
            "--perception-randomization.noise-std-mult-range",
            "[0.0,0.05]",
            "--perception-randomization.noise-drop-prob-range",
            "[0.0,0.025]",
        ],
        config=TYRO_CONIFG,
    )

    assert parsed.perception_producer_tick_dt == pytest.approx(0.02)
    assert parsed.perception_allow_mujoco_noise is True
    assert parsed.perception_contract_envelope_b64 == envelope_b64
    assert parsed.perception_randomization.enabled is True
    assert parsed.perception_randomization.translation_range["z"] == [-0.03, 0.03]
    assert parsed.perception_randomization.rotation_range_deg["pitch"] == [-3.0, 3.0]
    assert parsed.perception_randomization.noise_std_mult_range == [0.0, 0.05]
    assert parsed.perception_randomization.noise_drop_prob_range == [0.0, 0.025]


@pytest.mark.parametrize(
    "subcommand",
    [
        "perception:camera_depth_d435i_mujoco",
        "perception:camera-depth-d435i-mujoco",
        "perception:camera_depth_d435i_mujoco_render_848x480",
        "perception:camera-depth-d435i-mujoco-render-848x480",
    ],
)
def test_run_sim_mujoco_perception_aliases_are_equivalent_and_warning_free(
    subcommand: str,
) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message=r"Duplicate subcommand name detected:.*",
            category=UserWarning,
        )
        parsed = tyro.cli(
            RunSimConfig,
            args=[
                "simulator:mujoco-split-debug",
                "robot:g1-29dof-w-object",
                "terrain:terrain-locomotion-plane",
                subcommand,
            ],
            config=TYRO_CONIFG,
        )

    assert (
        parsed.perception
        == perception_values.camera_depth_d435i_mujoco_render_848x480
    )


def test_direct_contract_envelope_is_strict_and_digest_authenticated() -> None:
    contract = {"version": 2, "producer_tick_dt": 0.02}
    canonical_contract = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(canonical_contract).hexdigest()
    encoded = base64.b64encode(
        json.dumps(
            {"contract": contract, "sha256": digest},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).decode("ascii")

    decoded_contract, decoded_digest = DirectSimulation._decode_perception_contract_envelope(
        encoded
    )

    assert decoded_contract == contract
    assert decoded_digest == digest

    duplicate = base64.b64encode(
        f'{{"contract":{{}},"contract":{{}},"sha256":"{digest}"}}'.encode()
    ).decode("ascii")
    with pytest.raises(ValueError, match="strict unique-key JSON"):
        DirectSimulation._decode_perception_contract_envelope(duplicate)

    mismatched = base64.b64encode(
        json.dumps(
            {"contract": contract, "sha256": "0" * 64},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).decode("ascii")
    with pytest.raises(ValueError, match="SHA-256"):
        DirectSimulation._decode_perception_contract_envelope(mismatched)


def test_reference_batch_hole_normalization_is_not_live_batch_normalization() -> None:
    class _FixedGenerator:
        def __init__(self, frame: torch.Tensor) -> None:
            self.frame = frame
            self.frame_idx = 0

        def generate_frame(
            self,
            *,
            frame_index: int | None = None,
            env_ids: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if frame_index is None:
                self.frame_idx += 1
            if env_ids is None:
                return self.frame.clone()
            return self.frame[env_ids].clone()

    reference_fields = torch.stack(
        (
            torch.linspace(-0.25, 0.25, 16).view(4, 4),
            torch.full((4, 4), -10.0),
            torch.full((4, 4), 10.0),
        )
    )
    depth = torch.ones((1, 4, 4))

    reference = object.__new__(PerceptionManager)
    reference._camera_warp_hole_prob = 0.2
    reference._camera_warp_hole_generator = _FixedGenerator(reference_fields)
    reference._camera_warp_hole_frame_stats = None
    reference_result = reference._apply_warp_hole_noise(depth.clone(), max_depth=10.0)

    live_only = object.__new__(PerceptionManager)
    live_only._camera_warp_hole_prob = 0.2
    live_only._camera_warp_hole_generator = _FixedGenerator(reference_fields[:1])
    live_only._camera_warp_hole_frame_stats = None
    live_result = live_only._apply_warp_hole_noise(depth.clone(), max_depth=10.0)

    assert not torch.equal(reference_result, live_result)
