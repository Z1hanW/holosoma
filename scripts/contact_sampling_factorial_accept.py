#!/usr/bin/env python3
"""Independently validate each two-update factorial canary before formal launch."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml

import contact_sampling_factorial as exp
from holosoma.agents.modules.modules import FarTrackingDepthSmallEncoder
from holosoma.config_values.perception import WARP_SENSORS_G1_D435_MESH_FILE_MAP
from holosoma.utils.checkpoint_validation import load_verified_torch_checkpoint


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite(value):
    if isinstance(value, torch.Tensor):
        require(bool(torch.isfinite(value).all()), "Non-finite checkpoint tensor")
    elif isinstance(value, dict):
        for child in value.values():
            finite(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            finite(child)
    elif isinstance(value, float):
        require(math.isfinite(value), "Non-finite checkpoint scalar")


class IndependentActor(torch.nn.Module):
    def __init__(self, state):
        super().__init__()
        self.encoder = FarTrackingDepthSmallEncoder(58, 87, 32)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(126, 512), torch.nn.ELU(), torch.nn.Linear(512, 256),
            torch.nn.ELU(), torch.nn.Linear(256, 128), torch.nn.ELU(), torch.nn.Linear(128, 29))
        for module, prefix in ((self.encoder, "actor_module.perception_encoder."), (self.mlp, "actor_module.module.")):
            module.load_state_dict({k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}, strict=True)

    def forward(self, actor, depth):
        return self.mlp(torch.cat((actor, self.encoder(depth)), dim=-1))


def accept(root, arm, *, world_size=8, bank=None, append_duration_s=0.0):
    campaign = json.loads((root / "campaign.json").read_text())
    mix, contact, adaptive = exp.arm_flags(arm)
    artifacts = root / arm / "canary_artifacts"
    pair = json.loads((artifacts / "model_00002.pair.json").read_text())
    require(pair["semantics"] == "atomic_same_iteration_pt_onnx_policy_pair", "Wrong pair type")
    require(pair["completed_iteration"] == 1 and pair["next_iteration"] == 2, "Wrong pair iteration")
    for extension in ("pt", "onnx"):
        path = artifacts / f"model_00002.{extension}"
        require(exp.sha(path) == pair[extension]["sha256"], f"{extension} checksum mismatch")
        require(path.stat().st_size == pair[extension]["size_bytes"], f"{extension} size mismatch")
    require(pair["onnx"]["pytorch_vs_ort"] is True, "Native ONNX parity failed")
    checkpoint, _ = load_verified_torch_checkpoint(artifacts / "model_00002.pt", expected_sha256=pair["pt"]["sha256"], map_location="cpu")
    finite(checkpoint)
    require(checkpoint["iter"] == checkpoint["iteration"] == 1 and checkpoint["next_iter"] == 2, "PT iteration mismatch")
    for key in ("rng_state_by_rank", "env_state_by_rank"):
        require(sorted(map(int, checkpoint[key])) == list(range(world_size)), f"Incomplete {key}")
    cfg = yaml.safe_load((artifacts / "holosoma_config.yaml").read_text())
    require(cfg == checkpoint["experiment_config"], "Config mismatch")
    training, algo = cfg["training"], cfg["algo"]["config"]
    motion = cfg["command"]["setup_terms"]["motion_command"]["params"]["motion_config"]
    require(training["num_envs"] == 2048 and training["multigpu"] and training["export_onnx"], "Topology/export drift")
    require(training["checkpoint"] is None and training["policy_init_checkpoint"] == exp.INIT, "Initializer drift")
    require(algo["distill"]["enabled"] == mix, "RL/mix drift")
    require(not algo["distill"]["take_teacher_actions"], "Teacher controls environment")
    if mix:
        distill = algo["distill"]
        require(distill["policy_to_clone"] == exp.TEACHER and distill["teacher_obs_keys"] == ["actor_obs"], "Teacher binding drift")
        require(distill["ppo_start_coeff"] == .01 and distill["ppo_target_coeff"] == .9, "Mix weights drift")
        require(distill["ppo_schedule_step_epochs"] == 700 and distill["dagger_end_epoch"] == 6300, "Schedule drift")
    require(motion["motion_file"] == (exp.BANK if bank is None else bank), "Motion bank drift")
    require(motion.get("runtime_default_pose_append_duration_s", 0.0) == append_duration_s, "Runtime append drift")
    require(motion["contact_aware_button_window_mode"] == "peak_height", "Button drift")
    require(motion["contact_aware_sparse_root_command_mode"] == "precomputed_turn_then_forward", "Command drift")
    require(motion["zero_root_command_when_drop_active"], "Drop is not exclusive")
    require(motion["use_adaptive_timesteps_sampler"] == adaptive, "Within-clip sampler drift")
    require(motion["clip_weighting_strategy"] == "uniform_clip" and not motion["uniform_t1_window_sampling_enabled"], "Other sampling drift")
    require(motion["start_at_timestep_zero_prob_end_iter"] == 1 and motion["start_at_timestep_zero_prob_end"] == 1.0 / 39999, "Canary must preserve first two formal probabilities")
    reward = cfg["reward"]["terms"]["offline_contact_guidance"]
    require(reward["weight"] == float(contact), "Contact outer weight drift")
    require(reward["params"]["contact_weight"] == reward["params"]["wrist_weight"] == 1., "Contact inner weight drift")
    require(reward["params"]["contact_schedule_missing_mode"] == "inactive", "Contact fallback enabled")
    perception = cfg["perception"]
    require(perception["sensor_offset"] == [.01, .01, .44] and perception["camera_pitch_deg"] == 10., "Camera drift")
    require(perception["camera_mount_quat"] == [.00644801, .23350163, .00644801, .97231365], "Mount drift")
    require(perception["camera_warp_latency_frame"] == [3, 4], "Depth latency drift")
    require(perception["camera_mesh_file_map"] == WARP_SENSORS_G1_D435_MESH_FILE_MAP, "SW depth mesh drift")
    initializer = json.loads((artifacts / "initializer_preflight.json").read_text())
    require(initializer["accepted"] and initializer["checkpoint_sha256"] == exp.INIT_SHA, "Invalid init preflight")
    require(initializer["source_vs_target_actor_max_abs_error"] == 0 and initializer["depth_encoder_trainable"], "Initializer is not exact")
    verification = json.loads((artifacts / "git_verification.json").read_text())
    require(verification["accepted"] and verification["commit_sha"] == campaign["source"]["commit"], "Wrong Git source")
    require(verification["tracked_diff_clean"] and verification["untracked_clean"], "Dirty Git source")
    provenance = checkpoint["training_provenance"]
    require(provenance["teacher_enabled"] == mix, "Teacher provenance drift")
    require(provenance["factorial_experiment"]["definition_sha256"] == campaign["definitions"][arm], "Definition drift")
    logs = sorted(artifacts.glob("train_rank_*.log"))
    require(len(logs) == world_size, "Missing rank logs")
    fatal = re.compile(r"Traceback|RuntimeError|CUDA out of memory|ChildFailedError|Segmentation fault|non-finite|NCCL.*error", re.I)
    for log in [*logs, artifacts / "controller.log"]:
        content = log.read_text(errors="replace")
        require(not fatal.search(content), f"Fatal log: {log}")
    controller = (artifacts / "controller.log").read_text()
    require("HOLOSOMA_PROGRESS completed_iteration=2" in controller and "HOLOSOMA_RUN_COMPLETE target_iteration=2" in controller, "Incomplete training")
    graph = onnx.load(artifacts / "model_00002.onnx")
    onnx.checker.check_model(graph, full_check=True)
    metadata = {item.key: item.value for item in graph.metadata_props}
    require(len(metadata) == len(graph.metadata_props), "Duplicate metadata")
    require("precomputed_turn_then_forward_deployment_contract" in metadata, "Missing deployment adapter")
    session = ort.InferenceSession(str(artifacts / "model_00002.onnx"), providers=["CPUExecutionProvider"])
    require([(i.name, i.shape) for i in session.get_inputs()] == [("actor_obs", ["batch", 94]), ("perception_obs", ["batch", 5046])], "Input contract drift")
    actor = IndependentActor(checkpoint["actor_model_state_dict"]).eval()
    rng = np.random.default_rng(42)
    max_abs = 0.
    for batch in (1, 2, 3, 8):
        inputs = {"actor_obs": rng.normal(size=(batch, 94)).astype(np.float32), "perception_obs": rng.uniform(-.5, .5, (batch, 5046)).astype(np.float32)}
        with torch.inference_mode():
            expected = actor(torch.from_numpy(inputs["actor_obs"]), torch.from_numpy(inputs["perception_obs"])).numpy()
        actual = session.run(["action"], inputs)[0]
        require(np.allclose(expected, actual, rtol=1e-3, atol=2e-6), "Independent ONNX parity failed")
        max_abs = max(max_abs, float(np.max(np.abs(actual - expected))))
    report = {"accepted": True, "arm": arm, "commit": campaign["source"]["commit"],
              "definition_sha256": campaign["definitions"][arm], "git_verification": verification,
              "pair": pair, "independent_onnx_max_abs": max_abs, "independent_probe_rows": 14,
              "envs_per_gpu": 2048, "ranks": world_size, "canary_updates": 2,
              "runtime_default_pose_append_duration_s": append_duration_s}
    exp.save(root / arm / "canary_acceptance.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=exp.ARMS)
    args = parser.parse_args()
    torch.set_num_threads(2)
    for arm in args.arms or exp.ARMS:
        accept(args.root, arm)
