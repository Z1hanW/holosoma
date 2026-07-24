from __future__ import annotations

import base64
import copy
import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import tyro

from holosoma.config_types.run_sim import RunSimConfig
from holosoma.config_types.terrain import MeshType
from holosoma.config_values import perception as perception_values
from holosoma.config_values import robot as robot_values
from holosoma.eval_agent import (
    Sim2SimConfig,
    _build_sim2sim_commands,
    _launch_sim2sim,
    _perception_runtime_values,
    _preflight_perception_onnx_artifact,
    _validate_direct_perception_distribution_source,
    _validate_eval_policy_contract,
)
from holosoma.utils.tyro_utils import TYRO_CONIFG


def _config(*, with_perception: bool):
    if with_perception:
        actor_groups = [
            "actor_obs_root_contact_aware",
            "actor_obs_drop_button",
            "actor_obs_proprio_with_actions_no_linvel",
        ]
        perception_input_name = "perception_obs"
        perception = replace(
            perception_values.camera_depth_d435i,
            object_geometry_mode="mesh",
            camera_pitch_deg=10.0,
            camera_warp_hole_reference_batch_size=2048,
        )
        env_class = "holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager"
    else:
        actor_groups = ["actor_obs"]
        perception_input_name = ""
        perception = perception_values.none
        env_class = "holosoma.envs.locomotion.locomotion_manager.LocomotionManager"

    actor = SimpleNamespace(
        input_dim=actor_groups,
        layer_config=SimpleNamespace(perception_input_name=perception_input_name),
    )
    return SimpleNamespace(
        env_class=env_class,
        command=None,
        robot=robot_values.g1_29dof,
        algo=SimpleNamespace(config=SimpleNamespace(module_dict=SimpleNamespace(actor=actor))),
        simulator=SimpleNamespace(
            _target_="holosoma.simulator.isaacgym.isaacgym.IsaacGym",
            config=SimpleNamespace(
                name="isaacgym",
                sim=SimpleNamespace(fps=500, control_decimation=10),
            ),
        ),
        training=SimpleNamespace(num_envs=2048, seed=42),
        randomization=SimpleNamespace(reset_terms={}),
        terrain=SimpleNamespace(
            terrain_term=SimpleNamespace(mesh_type=MeshType.PLANE, obj_file_path=None)
        ),
        perception=perception,
    )


def _option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def _set_simulator_backend(config, backend: str) -> None:
    targets = {
        "isaacgym": "holosoma.simulator.isaacgym.isaacgym.IsaacGym",
        "isaacsim": "holosoma.simulator.isaacsim.isaacsim.IsaacSim",
        "mujoco": "holosoma.simulator.mujoco.mujoco.MuJoCo",
    }
    config.simulator._target_ = targets[backend]
    config.simulator.config.name = backend


def _effective_noise_schema(config, *, enabled: bool) -> dict[str, object]:
    if not enabled:
        return {
            "edge_noise": False,
            "enable_holes": False,
            "hole_prob": 0.0,
            "additive_noise_std": 0.0,
            "depth_offset_std": 0.0,
            "apply_sensor_noise": False,
        }
    perception = config.perception
    return {
        "edge_noise": perception.camera_warp_edge_noise,
        "enable_holes": perception.camera_warp_enable_holes,
        "hole_prob": perception.camera_warp_hole_prob,
        "additive_noise_std": perception.camera_warp_additive_noise_std,
        "depth_offset_std": perception.camera_warp_depth_offset_std,
        "apply_sensor_noise": perception.camera_apply_sensor_noise,
    }


def _perception_contract(config, *, effective_noise: bool = True) -> dict[str, object]:
    holes_active = bool(
        effective_noise
        and config.perception.camera_warp_enable_holes
        and config.perception.camera_warp_hole_prob > 0.0
    )
    hole_generator_schema = None
    if holes_active:
        hole_generator_schema = {
            "shape": [64, 96],
            "resolutions": [[2, 2], [4, 4], [8, 8], [16, 16], [32, 32]],
            "periods": [32, 16, 8, 4, 2],
            "factors": [1.0],
            "normalization_scope": "reference_batch",
            "reference_batch_size": config.perception.camera_warp_hole_reference_batch_size,
            "seed_semantics": "rank_local_v2",
            "effective_seed": config.training.seed,
            "gradient_seed_mixer": "sha256_u63_be_v1",
            "octave_profile": "legacy_single_octave_v1",
        }
    return {
        "version": 2,
        "camera_source": "far_tracking_warp",
        "producer_tick_dt": (
            config.simulator.config.sim.control_decimation
            / config.simulator.config.sim.fps
        ),
        "producer_lifecycle": {
            "reset_refresh_semantics": "targeted_v2",
            "ordinary_manager_update_calls_per_control_tick": 1,
            "initialization_control_ticks_before_first_reset_output": 1,
            "initialization_ordinary_manager_update_calls_before_first_reset_output": 1,
            "reset_output_republished_until_physics_advances": True,
            "reset_output_scope": "reset_env_subset",
            "hole_clock_advances_on_reset_refresh": False,
            "camera_frequency_phase_advances_on_reset_refresh": False,
            "camera_producer_reset_refresh_consumes_process_global_rng": True,
            "future_noise_sample_path_peer_reset_coupled": True,
            "batch_size_invariant_sample_path": False,
            "stochastic_equivalence": "distribution_only",
            "seed_replay_scope": "same_execution_trace_only",
        },
        "camera_reset_randomization": None,
        "camera_setup_randomization": None,
        "camera_ray_correction_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "effective_observation_schema": _effective_noise_schema(
            config,
            enabled=effective_noise,
        ),
        "hole_generator_schema": hole_generator_schema,
        "training_geometry_support": {
            "version": 1,
            "camera_source": "far_tracking_warp",
            "training_rank_count": 1,
            "robot_mesh_bindings": [],
            "object_mesh_support": [],
        },
    }


def _contract_envelope(
    config,
    *,
    effective_noise: bool = True,
    contract: dict[str, object] | None = None,
) -> str:
    contract = contract or _perception_contract(config, effective_noise=effective_noise)
    digest = hashlib.sha256(
        json.dumps(
            contract,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    payload = json.dumps(
        {"contract": contract, "sha256": digest},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return base64.b64encode(payload).decode("ascii")


def _build_authenticated_sim2sim_commands(
    config,
    sim2sim: Sim2SimConfig | None = None,
    model_path: str = "/tmp/model.onnx",
    *,
    effective_noise: bool = True,
    contract: dict[str, object] | None = None,
):
    return _build_sim2sim_commands(
        config,
        sim2sim or Sim2SimConfig(),
        model_path,
        perception_contract_envelope_b64=_contract_envelope(
            config,
            effective_noise=effective_noise,
            contract=contract,
        ),
    )


def _write_authenticated_perception_onnx(
    path: Path,
    *,
    checkpoint_sha256: str,
    config,
    contract_override: dict[str, object] | None = None,
) -> str:
    onnx = pytest.importorskip("onnx")
    model = onnx.helper.make_model(onnx.helper.make_graph([], "metadata-only", [], []))
    contract = contract_override or _perception_contract(config)
    digest = hashlib.sha256(
        json.dumps(
            contract,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    for key, value in {
        "source_checkpoint_sha256": checkpoint_sha256,
        "perception_observation_contract": contract,
        "perception_observation_contract_sha256": digest,
    }.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = json.dumps(value, sort_keys=True, allow_nan=False)
    onnx.save(model, path)
    return digest


def _with_object_motion(config, *, motion_source, object_source):
    config.robot = replace(
        robot_values.g1_29dof,
        object=replace(
            robot_values.g1_29dof.object,
            enabled=True,
            object_urdf_path=str(object_source),
        ),
    )
    config.command = SimpleNamespace(
        setup_terms={
            "motion_command": SimpleNamespace(
                func="holosoma.managers.command.terms.wbt:MotionCommand",
                params={"motion_config": SimpleNamespace(motion_file=str(motion_source))},
            )
        }
    )
    return config


def _object_bank(tmp_path):
    bank = tmp_path / "motion_bank"
    bank.mkdir()
    source = tmp_path / "motion_source"
    source.mkdir()
    source_motion = source / "clip_a.npz"
    source_motion.touch()
    motion = bank / "clip_a.npz"
    motion.symlink_to(source_motion)
    urdf_dir = bank / "urdfs"
    urdf_dir.mkdir()
    urdf = urdf_dir / "clip_a.urdf"
    urdf.write_text("<robot name='object'/>", encoding="utf-8")
    object_map = bank / "_clip_object_urdf_map.json"
    object_map.write_text(
        json.dumps({"clips": {"clip_a": {"object_urdf_path": "urdfs/clip_a.urdf"}}}),
        encoding="utf-8",
    )
    return bank, motion, urdf, object_map


def test_non_perception_sim2sim_command_does_not_enable_perception() -> None:
    run_sim, run_policy = _build_sim2sim_commands(
        _config(with_perception=False), Sim2SimConfig(), "/tmp/model.onnx"
    )

    assert not any(token.startswith("perception:") for token in run_sim)
    assert "--simulator.config.bridge.publish-perception-obs" not in run_sim
    assert "--simulator.config.bridge.publish-perception-obs-shm" not in run_sim
    assert "--task.use-split-perception-obs" not in run_policy
    assert "--device" not in run_sim


def test_perception_onnx_is_authenticated_before_run_sim_launch(tmp_path) -> None:
    config = _config(with_perception=True)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"authenticated checkpoint bytes")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    onnx_path = tmp_path / "model.onnx"
    expected_contract_sha256 = _write_authenticated_perception_onnx(
        onnx_path,
        checkpoint_sha256=checkpoint_sha256,
        config=config,
    )

    assert _preflight_perception_onnx_artifact(
        str(onnx_path),
        checkpoint_sha256=checkpoint_sha256,
        config=config,
    ) == expected_contract_sha256

    with pytest.raises(ValueError, match="source checkpoint does not match"):
        _preflight_perception_onnx_artifact(
            str(onnx_path),
            checkpoint_sha256="00" * 32,
            config=config,
        )


def test_perception_onnx_rejects_self_consistent_wrong_rank_zero_hole_seed(tmp_path) -> None:
    config = _config(with_perception=True)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"authenticated checkpoint bytes")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    contract = _perception_contract(config)
    contract["hole_generator_schema"]["effective_seed"] = config.training.seed + 1
    onnx_path = tmp_path / "wrong-seed.onnx"
    _write_authenticated_perception_onnx(
        onnx_path,
        checkpoint_sha256=checkpoint_sha256,
        config=config,
        contract_override=contract,
    )

    with pytest.raises(ValueError, match="rank-zero training seed"):
        _preflight_perception_onnx_artifact(
            str(onnx_path),
            checkpoint_sha256=checkpoint_sha256,
            config=config,
        )


def test_depth_policy_builds_matching_zmq_producer_and_checkpoint_preprocessing() -> None:
    config = _config(with_perception=True)
    run_sim, run_policy = _build_authenticated_sim2sim_commands(config)

    assert any(token.startswith("perception:camera-depth-d435i") for token in run_sim)
    assert _option_value(run_sim, "--simulator.config.bridge.publish-perception-obs") == "True"
    assert _option_value(run_sim, "--simulator.config.bridge.publish-perception-obs-shm") == "False"
    assert _option_value(run_sim, "--simulator.config.bridge.perception-obs-port") == "5558"
    assert _option_value(run_policy, "--task.perception-obs-port") == "5558"
    assert "--task.use-split-perception-obs" in run_policy
    assert "--task.no-use-split-perception-obs-shm" in run_policy
    assert _option_value(run_sim, "--device") == "cuda:0"
    assert float(_option_value(run_sim, "--perception-producer-tick-dt")) == pytest.approx(0.02)
    assert _option_value(run_sim, "--perception-allow-mujoco-noise") == "True"
    assert _option_value(run_sim, "--training.seed") == "42"
    assert _option_value(run_sim, "--perception-randomization.enabled") == "False"
    assert _option_value(
        run_sim,
        "--perception.camera-warp-hole-reference-batch-size",
    ) == "2048"

    parsed = tyro.cli(RunSimConfig, args=run_sim[2:], config=TYRO_CONIFG)
    assert _perception_runtime_values(parsed.perception) == _perception_runtime_values(config.perception)


@pytest.mark.parametrize(
    ("saved_backend", "effective_noise", "expected_permission"),
    [
        ("isaacgym", True, True),
        ("isaacsim", True, True),
        ("mujoco", False, False),
        ("mujoco", True, True),
    ],
)
def test_direct_mujoco_noise_permission_reconstructs_authenticated_training_effect(
    saved_backend: str,
    effective_noise: bool,
    expected_permission: bool,
) -> None:
    config = _config(with_perception=True)
    _set_simulator_backend(config, saved_backend)

    run_sim, _ = _build_authenticated_sim2sim_commands(
        config,
        effective_noise=effective_noise,
    )

    assert _option_value(run_sim, "--perception-allow-mujoco-noise") == str(
        expected_permission
    )


def test_perception_command_builder_requires_authenticated_contract_envelope() -> None:
    config = _config(with_perception=True)

    with pytest.raises(ValueError, match="authenticated ONNX effective observation-contract"):
        _build_sim2sim_commands(config, Sim2SimConfig(), "/tmp/model.onnx")


@pytest.mark.parametrize(
    ("saved_backend", "effective_schema", "error"),
    [
        (
            "isaacgym",
            {
                "edge_noise": False,
                "enable_holes": False,
                "hole_prob": 0.0,
                "additive_noise_std": 0.0,
                "depth_offset_std": 0.0,
                "apply_sensor_noise": False,
            },
            "does not match the saved Isaac training configuration",
        ),
        (
            "mujoco",
            {
                "edge_noise": True,
                "enable_holes": False,
                "hole_prob": 0.0,
                "additive_noise_std": 0.0,
                "depth_offset_std": 0.0,
                "apply_sensor_noise": False,
            },
            "matches neither the saved MuJoCo configuration nor MuJoCo's canonical effective-off",
        ),
    ],
)
def test_direct_mujoco_noise_permission_rejects_authenticated_ambiguous_effect(
    saved_backend: str,
    effective_schema: dict[str, object],
    error: str,
) -> None:
    config = _config(with_perception=True)
    _set_simulator_backend(config, saved_backend)
    contract = _perception_contract(config)
    contract["effective_observation_schema"] = effective_schema

    with pytest.raises(ValueError, match=error):
        _build_authenticated_sim2sim_commands(config, contract=contract)


@pytest.mark.parametrize("failure_mode", ["missing_schema", "bad_digest", "noncanonical_json"])
def test_perception_command_builder_parses_contract_envelope_strictly(
    failure_mode: str,
) -> None:
    config = _config(with_perception=True)
    contract = _perception_contract(config)
    if failure_mode == "missing_schema":
        del contract["effective_observation_schema"]
        envelope = _contract_envelope(config, contract=contract)
        error = "missing effective_observation_schema"
    else:
        envelope_payload = json.loads(base64.b64decode(_contract_envelope(config)))
        if failure_mode == "bad_digest":
            envelope_payload["sha256"] = "00" * 32
            payload = json.dumps(
                envelope_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
            error = "SHA-256 does not match"
        else:
            payload = json.dumps(
                envelope_payload,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
            error = "not in its canonical encoding"
        envelope = base64.b64encode(payload).decode("ascii")

    with pytest.raises(ValueError, match=error):
        _build_sim2sim_commands(
            config,
            Sim2SimConfig(),
            "/tmp/model.onnx",
            perception_contract_envelope_b64=envelope,
        )


def test_depth_policy_matches_shared_memory_inference_transport() -> None:
    sim2sim = Sim2SimConfig(
        inference_config="inference:g1-29dof-wbt-object-distill-mujoco",
        run_policy_args="--task.perception-obs-shm-name scientific_depth",
    )
    config = _config(with_perception=True)
    run_sim, run_policy = _build_authenticated_sim2sim_commands(config, sim2sim)

    assert _option_value(run_sim, "--simulator.config.bridge.publish-perception-obs") == "False"
    assert _option_value(run_sim, "--simulator.config.bridge.publish-perception-obs-shm") == "True"
    assert _option_value(run_sim, "--simulator.config.bridge.perception-obs-shm-name") == "scientific_depth"
    assert "--task.use-split-perception-obs-shm" in run_policy
    assert _option_value(run_policy, "--task.perception-obs-shm-name") == "scientific_depth"


def test_depth_policy_propagates_policy_zmq_port_to_run_sim() -> None:
    sim2sim = Sim2SimConfig(run_policy_args="--task.perception-obs-port 6208")
    config = _config(with_perception=True)
    run_sim, run_policy = _build_authenticated_sim2sim_commands(config, sim2sim)

    assert _option_value(run_sim, "--simulator.config.bridge.perception-obs-port") == "6208"
    # The explicit policy override is last, so this checks the effective value.
    assert run_policy[-2:] == ["--task.perception-obs-port", "6208"]


def test_depth_policy_rejects_disabled_checkpoint_perception() -> None:
    config = _config(with_perception=True)
    config.perception = perception_values.none

    with pytest.raises(ValueError, match="checkpoint perception is disabled"):
        _build_sim2sim_commands(config, Sim2SimConfig(), "/tmp/model.onnx")


@pytest.mark.parametrize(
    "run_sim_args, error",
    [
        (
            "--simulator.config.bridge.publish-perception-obs False",
            "conflicts with the policy perception transport",
        ),
        (
            "--simulator.config.bridge.perception-obs-port 6209",
            "ports differ",
        ),
        ("--device cpu", "requires a CUDA"),
        ("--perception.camera-pitch-deg 0", "may not override perception"),
        ("--perception-randomization.enabled False", "may not override perception"),
        ("--perception-producer-tick-dt 0.01", "may not override perception"),
        ("--perception-allow-mujoco-noise False", "may not override perception"),
        ("--training.seed 7", "may not override perception"),
    ],
)
def test_depth_policy_rejects_conflicting_run_sim_overrides(run_sim_args: str, error: str) -> None:
    config = _config(with_perception=True)
    with pytest.raises(ValueError, match=error):
        _build_authenticated_sim2sim_commands(
            config,
            Sim2SimConfig(run_sim_args=run_sim_args),
        )


def test_depth_policy_rejects_disabled_policy_subscription() -> None:
    config = _config(with_perception=True)
    with pytest.raises(ValueError, match="split perception subscription is explicitly disabled"):
        _build_authenticated_sim2sim_commands(
            config,
            Sim2SimConfig(run_policy_args="--task.no-use-split-perception-obs"),
        )


@pytest.mark.parametrize(
    "apply_sensor_noise, params, component",
    [
        (
            True,
            {"noise_std_mult_range": [0.0, 0.05], "noise_drop_prob_range": [0.0, 0.025]},
            "multiplicative depth noise",
        ),
        (
            False,
            {"translation_range": {"x": [-0.025, 0.025]}},
            "camera translation",
        ),
    ],
)
def test_depth_policy_reconstructs_direct_camera_randomization(
    apply_sensor_noise: bool, params: dict, component: str
) -> None:
    config = _config(with_perception=True)
    config.perception = replace(config.perception, camera_apply_sensor_noise=apply_sensor_noise)
    if "translation_range" in params:
        params["translation_range"] = {
            "x": [-0.025, 0.025],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
        }
    config.randomization = SimpleNamespace(
        reset_terms={
            "randomize_camera_raycast": SimpleNamespace(
                func="holosoma.managers.randomization.terms.locomotion:randomize_camera_raycast",
                params={"enabled": True, **params},
            )
        }
    )

    run_sim, _ = _build_authenticated_sim2sim_commands(config)

    assert _option_value(run_sim, "--perception-randomization.enabled") == "True"
    if component == "multiplicative depth noise":
        assert json.loads(
            _option_value(run_sim, "--perception-randomization.noise-std-mult-range")
        ) == [0.0, 0.05]
        assert json.loads(
            _option_value(run_sim, "--perception-randomization.noise-drop-prob-range")
        ) == [0.0, 0.025]
    else:
        assert json.loads(
            _option_value(run_sim, "--perception-randomization.translation-range")
        )["x"] == [-0.025, 0.025]


def test_sensor_noise_toggle_without_declared_distribution_is_rejected() -> None:
    config = _config(with_perception=True)
    config.perception = replace(config.perception, camera_apply_sensor_noise=True)

    with pytest.raises(ValueError, match="no enabled canonical camera randomizer"):
        _build_authenticated_sim2sim_commands(config)


def test_disabled_sensor_noise_does_not_require_unused_noise_randomization() -> None:
    config = _config(with_perception=True)
    config.randomization = SimpleNamespace(
        reset_terms={
            "randomize_camera_raycast": SimpleNamespace(
                func="holosoma.managers.randomization.terms.locomotion:randomize_camera_raycast",
                params={
                    "enabled": True,
                    "noise_std_mult_range": [0.0, 0.05],
                    "noise_drop_prob_range": [0.0, 0.025],
                },
            )
        }
    )

    run_sim, _ = _build_authenticated_sim2sim_commands(config)
    assert any(token.startswith("perception:camera-depth-d435i") for token in run_sim)


def test_legacy_vectorized_reset_producer_cannot_be_relabelled_exact_direct() -> None:
    config = _config(with_perception=True)
    config.perception = replace(
        config.perception,
        reset_refresh_semantics="legacy_full_v1",
    )

    with pytest.raises(ValueError, match="cannot be represented by direct RunSim"):
        _build_authenticated_sim2sim_commands(config)


def test_exact_direct_perception_rejects_eval_seed_drift() -> None:
    saved = _config(with_perception=True)
    evaluated = copy.deepcopy(saved)
    evaluated.training.seed = 43

    with pytest.raises(
        ValueError,
        match="Distribution-initialization seed.*must come from the saved checkpoint config",
    ):
        _validate_direct_perception_distribution_source(saved, evaluated)


def test_exact_direct_perception_rejects_saved_simulator_backend_drift() -> None:
    saved = _config(with_perception=True)
    evaluated = copy.deepcopy(saved)
    _set_simulator_backend(evaluated, "mujoco")

    with pytest.raises(ValueError, match="changed the saved simulator backend"):
        _validate_direct_perception_distribution_source(saved, evaluated)


def test_perception_policy_rejects_control_timestep_drift_before_export() -> None:
    saved = _config(with_perception=True)
    evaluated = copy.deepcopy(saved)
    saved.algo._target_ = "holosoma.agents.ppo.ppo.PPO"
    evaluated.algo._target_ = saved.algo._target_
    evaluated.simulator.config.sim.fps = 1000

    with pytest.raises(ValueError, match="falsely re-authenticate student weights"):
        _validate_eval_policy_contract(saved, evaluated)


class _FakeProcess:
    def __init__(self, status: int | None) -> None:
        self.status = status
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.status

    def terminate(self) -> None:
        self.terminated = True
        self.status = -15

    def kill(self) -> None:
        self.killed = True
        self.status = -9

    def wait(self, timeout: float | None = None) -> int:
        assert timeout is None or timeout > 0.0
        return int(self.status or 0)


def test_sim2sim_launcher_propagates_run_sim_failure_and_stops_policy(monkeypatch) -> None:
    run_sim = _FakeProcess(status=7)
    run_policy = _FakeProcess(status=None)
    processes = iter((run_sim, run_policy))
    monkeypatch.setattr("holosoma.eval_agent.subprocess.Popen", lambda _cmd: next(processes))

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        _launch_sim2sim(["run_sim"], ["run_policy"])

    assert exc_info.value.returncode == 7
    assert run_policy.terminated is True


def test_sim2sim_launcher_rejects_clean_but_early_run_sim_exit(monkeypatch) -> None:
    run_sim = _FakeProcess(status=0)
    run_policy = _FakeProcess(status=None)
    processes = iter((run_sim, run_policy))
    monkeypatch.setattr("holosoma.eval_agent.subprocess.Popen", lambda _cmd: next(processes))

    with pytest.raises(RuntimeError, match="run_sim exited before run_policy completed"):
        _launch_sim2sim(["run_sim"], ["run_policy"])

    assert run_policy.terminated is True


def test_object_motion_bank_requires_explicit_concrete_pair(tmp_path) -> None:
    bank, _, _, object_map = _object_bank(tmp_path)
    config = _with_object_motion(
        _config(with_perception=True), motion_source=bank, object_source=object_map
    )

    with pytest.raises(ValueError, match="--motion-init.motion-file.*--task.motion-file"):
        _build_sim2sim_commands(config, Sim2SimConfig(), "/tmp/model.onnx")


def test_object_motion_bank_accepts_and_verifies_explicit_mapped_pair(tmp_path) -> None:
    bank, motion, urdf, object_map = _object_bank(tmp_path)
    config = _with_object_motion(
        _config(with_perception=True), motion_source=bank, object_source=object_map
    )
    sim2sim = Sim2SimConfig(
        run_sim_args=(
            f"--motion-init.motion-file {motion} "
            f"--robot.object.object-urdf-path {urdf}"
        ),
        run_policy_args=f"--task.motion-file {motion}",
    )

    run_sim, run_policy = _build_authenticated_sim2sim_commands(config, sim2sim)

    assert "robot:g1_29dof_w_object" in run_sim
    assert _option_value(run_sim, "--motion-init.enabled") == "True"
    assert Path(_option_value(run_sim, "--motion-init.motion-file")) == motion
    assert _option_value(run_sim, "--robot.object.enabled") == "True"
    assert Path(_option_value(run_sim, "--robot.object.object-urdf-path")) == urdf
    assert Path(_option_value(run_policy, "--task.motion-file")) == motion


def test_object_motion_bank_rejects_wrong_mapped_urdf(tmp_path) -> None:
    bank, motion, _, object_map = _object_bank(tmp_path)
    wrong_urdf = tmp_path / "wrong.urdf"
    wrong_urdf.write_text("<robot name='wrong'/>", encoding="utf-8")
    config = _with_object_motion(
        _config(with_perception=True), motion_source=bank, object_source=object_map
    )
    sim2sim = Sim2SimConfig(
        run_sim_args=(
            f"--motion-init.motion-file {motion} "
            f"--robot.object.object-urdf-path {wrong_urdf}"
        ),
        run_policy_args=f"--task.motion-file {motion}",
    )

    with pytest.raises(ValueError, match="does not match the checkpoint clip-object mapping"):
        _build_sim2sim_commands(config, sim2sim, "/tmp/model.onnx")


def test_object_motion_bank_rejects_different_sim_and_policy_motion(tmp_path) -> None:
    bank, motion, urdf, object_map = _object_bank(tmp_path)
    other_motion = bank / "clip_b.npz"
    other_motion.touch()
    config = _with_object_motion(
        _config(with_perception=True), motion_source=bank, object_source=object_map
    )
    sim2sim = Sim2SimConfig(
        run_sim_args=(
            f"--motion-init.motion-file {motion} "
            f"--robot.object.object-urdf-path {urdf}"
        ),
        run_policy_args=f"--task.motion-file {other_motion}",
    )

    with pytest.raises(ValueError, match="must use the same concrete motion file"):
        _build_sim2sim_commands(config, sim2sim, "/tmp/model.onnx")


def test_single_object_motion_contract_is_automatic(tmp_path) -> None:
    motion = tmp_path / "only_clip.npz"
    motion.touch()
    urdf = tmp_path / "only_object.urdf"
    urdf.write_text("<robot name='only'/>", encoding="utf-8")
    config = _with_object_motion(
        _config(with_perception=True), motion_source=motion, object_source=urdf
    )

    run_sim, run_policy = _build_authenticated_sim2sim_commands(config)

    assert "robot:g1_29dof_w_object" in run_sim
    assert Path(_option_value(run_sim, "--motion-init.motion-file")) == motion
    assert Path(_option_value(run_sim, "--robot.object.object-urdf-path")) == urdf
    assert Path(_option_value(run_policy, "--task.motion-file")) == motion


@pytest.mark.parametrize("disabled_option", ["--motion-init.enabled", "--robot.object.enabled"])
def test_object_motion_contract_rejects_disabled_sim_component(tmp_path, disabled_option: str) -> None:
    motion = tmp_path / "only_clip.npz"
    motion.touch()
    urdf = tmp_path / "only_object.urdf"
    urdf.write_text("<robot name='only'/>", encoding="utf-8")
    config = _with_object_motion(
        _config(with_perception=True), motion_source=motion, object_source=urdf
    )

    with pytest.raises(ValueError, match="cannot be disabled"):
        _build_sim2sim_commands(
            config,
            Sim2SimConfig(run_sim_args=f"{disabled_option} False"),
            "/tmp/model.onnx",
        )
