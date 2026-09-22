#!/usr/bin/env python3
"""Exact-Git, 64-rank CORL79+rollout137 mix-contact+adaptive experiment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import contact_sampling_factorial as recipe

ARM = "mix-contact+adaptive"
NAME = "mix-contact+adaptive-corl79+rollout137-append02-ws64"
WORLD_SIZE = 64


def training_args(campaign, mode, run_id=None):
    args = recipe.training_args(ARM, mode, campaign["persist"], run_id)
    bank, contact = campaign["dataset"]["bank"], campaign["dataset"]["contact_root"]
    changes = {
        "training.name": NAME, "logger.name": NAME, "training.num-envs": WORLD_SIZE * 2048,
        recipe.MOTION_PREFIX + "motion-file": bank,
        recipe.MOTION_PREFIX + "adaptive-sampling-contact-interval-root": contact,
        recipe.MOTION_PREFIX + "runtime-default-pose-append-duration-s": 0.2,
        "reward.terms.offline-contact-guidance.params.contact-export-root": contact,
        "robot.object.object-urdf-path": bank + "/_clip_object_urdf_map.json",
    }
    args = [a for a in args if not (a.startswith("--") and a[2:].split("=", 1)[0] in changes)]
    args += [f"--{k}={v}" for k, v in changes.items()]
    if "--training.export-onnx=True" not in args:
        raise ValueError("Native ONNX export must be enabled")
    return args


def definition(campaign):
    return {"cli": training_args({**campaign, "persist": "/canonical"}, "canary"),
            "teacher_sha256": recipe.TEACHER_SHA, "initializer_sha256": recipe.INIT_SHA,
            "world_size": WORLD_SIZE, "append_duration_s": .2,
            "transport": "8_local_nccl_groups_8_gloo_cpu_leaders"}


def environment(campaign, node_rank, mode, work, verification):
    base = {**campaign, "nodes": {ARM: {"port": campaign["port"]}}}
    env = recipe.worker_environment(base, ARM, mode, work, verification)
    bank, contact = campaign["dataset"]["bank"], campaign["dataset"]["contact_root"]
    env.update({
        "NNODES": "8", "NODE_RANK": str(node_rank), "MASTER_ADDR": campaign["nodes"][0]["ip"],
        "MASTER_PORT": str(campaign["port"]), "TORCH_DIST_TIMEOUT_SEC": "3600",
        "GLOO_SOCKET_IFNAME": campaign["nodes"][node_rank]["interface"],
        "NCCL_SOCKET_IFNAME": campaign["nodes"][node_rank]["interface"],
        "NCCL_SOCKET_FAMILY": "AF_INET", "NCCL_SOCKET_RETRY_CNT": "34", "NCCL_SOCKET_RETRY_SLEEP_MSEC": "100",
        "NCCL_SOCKET_NTHREADS": "2", "NCCL_NSOCKS_PERTHREAD": "4",
        "TORCH_NCCL_DUMP_ON_TIMEOUT": "1", "TORCH_NCCL_TRACE_BUFFER_SIZE": "65536",
        "TORCH_NCCL_PROPAGATE_ERROR": "1", "TORCH_NCCL_BLOCKING_WAIT": "0",
        "HOLOSOMA_GLOO_GRAD_REDUCE": "0", "HOLOSOMA_HIERARCHICAL_GRAD_REDUCE": "1",
        "HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER": "1",
        "HOLOSOMA_EXTERNAL_AS_WORLD_SIZE": "64", "HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_DIR": bank,
        "MOTION_DIR": bank, "OBJECT_SPEC_PATH": bank + "/_clip_object_urdf_map.json",
        "OBJECT_URDF": bank + "/_clip_object_urdf_map.json", "CONTACT_EXPORT_ROOT": contact,
    })
    # The offline union has multiple producers; only the online label teacher is ch2/40K.
    env.pop("HOLOSOMA_EXTERNAL_AS_MOTION_GENERATOR_TEACHER_SHA256", None)
    return env


def worker(campaign_path, node_rank, mode):
    campaign = json.loads(campaign_path.read_text())
    if len(campaign["nodes"]) != 8 or campaign["definition_sha256"] != recipe.json_sha(definition(campaign)):
        raise ValueError("Campaign topology/definition mismatch")
    if campaign["nodes"][node_rank]["ip"] not in recipe.run(["hostname", "-I"]).split():
        raise ValueError("Worker launched on the wrong host")
    formal = mode in {"formal", "formal-preflight"}
    launch_mode = "formal" if formal else "canary"
    root = Path(campaign["persist"])
    work = root / ARM / mode / f"node_{node_rank}"
    work.mkdir(parents=True, exist_ok=True)
    (work / "wandb").mkdir(exist_ok=True)
    verification = work / "git_verification.json"
    recipe.run([sys.executable, recipe.ROOT / "scripts/verify_formal_git_checkout.py", "--source-root", recipe.ROOT,
                "--remote-url", recipe.REMOTE, "--remote-ref", "main", "--commit", campaign["source"]["commit"],
                "--tree", campaign["source"]["tree"], "--output", verification])
    recipe.save(work / "gpu_preflight.json", {"rows": recipe.gpu_health()})
    for path, expected in ((recipe.INIT, recipe.INIT_SHA), (recipe.TEACHER, recipe.TEACHER_SHA),
                           (recipe.NCCL + "/libnccl.so.2", recipe.NCCL_SHA)):
        if recipe.sha(path) != expected:
            raise ValueError(f"Initializer/teacher/runtime hash mismatch: {path}")
    for item in campaign["assets"]:
        if recipe.sha(item["path"]) != item["sha256"]:
            raise ValueError(f"Asset changed: {item['path']}")
    env = environment(campaign, node_rank, launch_mode, work, verification)
    recipe.run([recipe.PYTHON, recipe.ROOT / "scripts/verify_python_runtime_overlay.py", "--site-packages", recipe.RUNTIME,
                "--manifest-sha256", recipe.RUNTIME_SHA, "--require-distribution-closure", "--require-current-runtime-binding"], env=env)
    run_id = None
    if formal:
        acceptance_path = root / ARM / "canary_acceptance.json"
        acceptance = json.loads(acceptance_path.read_text())
        contract = json.loads((root / ARM / "run_contract.json").read_text())
        if (acceptance.get("accepted") is not True or acceptance["commit"] != campaign["source"]["commit"]
                or acceptance["ranks"] != 64 or contract["campaign_sha256"] != recipe.sha(campaign_path)
                or contract["canary_acceptance_sha256"] != recipe.sha(acceptance_path)):
            raise ValueError("Missing exact-source/config/topology acceptance")
        if len(contract["node_verifications"]) != 8:
            raise ValueError("Missing per-node source verification")
        run_id = contract["run_id"]
    cli = training_args(campaign, launch_mode, run_id)
    if formal and contract["cli"] != cli:
        raise ValueError("Immutable CLI mismatch")
    recipe.save(work / "training_cli.json", cli)
    recipe.run([recipe.PYTHON, recipe.ROOT / "scripts/validate_train_cli.py", "--expected-motion-end-mode", "episodic", "--", *cli], env=env)
    dataset = campaign["dataset"]
    provenance = json.loads(recipe.run([
        recipe.PYTHON, recipe.ROOT / "scripts/compute_training_provenance.py", "--training-regime", "distillation",
        "--motion-dir", dataset["bank"], "--object-map", dataset["bank"] + "/_clip_object_urdf_map.json",
        "--contact-root", dataset["contact_root"], "--contact-sidecar-mode", "runtime-intervals",
        "--motion-shard-manifest", env["HOLOSOMA_MOTION_SHARD_MANIFEST"], "--policy-init-checkpoint", recipe.INIT,
        "--contact-interval-runtime-prepend-compensation", "true", "--source-root", recipe.ROOT,
        "--teacher-checkpoint", recipe.TEACHER, "--student-motion-end-mode", "episodic"], env=env, timeout=1200))
    provenance["factorial_experiment"] = {
        "arm": ARM, "definition_sha256": campaign["definition_sha256"], "campaign_sha256": recipe.sha(campaign_path),
        "source": campaign["source"], "node_git_verification": json.loads(verification.read_text()),
        "offline_producers": dataset["source_counts"], "online_teacher": "ch2ckwzw/model_40000.pt",
        "dataset": dataset, "depth_mesh_reference_git": "df701cff64bd698e130d9533ceb47a3825d87e14"}
    if formal:
        provenance["factorial_experiment"]["formal_contract_sha256"] = recipe.sha(root / ARM / "run_contract.json")
        provenance["factorial_experiment"]["all_node_git_verifications"] = contract["node_verifications"]
    env["HOLOSOMA_TRAINING_PROVENANCE"] = json.dumps(provenance, sort_keys=True, separators=(",", ":"))
    recipe.save(work / "provenance.json", provenance)
    recipe.run([recipe.PYTHON, recipe.ROOT / "scripts/box23k_policy_init_preflight.py", "--world-size", "64",
                "--output", work / "initializer_preflight.json", "--allow-distillation", "--", *cli], env=env, timeout=1200)
    print(f"[INFO] mix216_preflight_passed node={node_rank} mode={mode} ws=64 envs_per_gpu=2048", flush=True)
    if mode in {"preflight", "formal-preflight"}:
        return
    recipe.gpu_health()
    os.chdir(recipe.ROOT)
    os.execve(recipe.PYTHON, [recipe.PYTHON, "-m", "torch.distributed.run", "--nnodes=8", f"--node_rank={node_rank}",
        "--master_addr=" + env["MASTER_ADDR"], "--master_port=" + env["MASTER_PORT"], "--nproc_per_node=8", "--max_restarts=0",
        "src/holosoma/holosoma/train_agent_rank_visible.py", *cli], env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "canary", "formal-preflight", "formal"))
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--node-rank", type=int, choices=range(8), required=True)
    args = parser.parse_args()
    worker(args.campaign, args.node_rank, args.mode)
