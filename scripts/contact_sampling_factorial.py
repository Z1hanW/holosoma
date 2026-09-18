#!/usr/bin/env python3
"""Git-bound 2x2x2 PPO/DAgger, contact, and within-clip sampling experiment."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "https://github.com/Z1hanW/holosoma"
PYTHON = "/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python3.11"
RUNTIME_SHA = "dd7ca81fa848917c362b3a239893a7a26f4c89d42b4f85cb515d91622f1690bc"
RUNTIME = f"/data/holosoma_runs/.runtime/python/python-runtime-v2-{RUNTIME_SHA}/site-packages"
NCCL_SHA = "e4a7aee9c3eecf53fac780441d2f03b578ab8db8874b71f8e391bcec7adb2899"
NCCL = f"/home/ubuntu/FAR/holosoma_runs/.runtime/nccl/{NCCL_SHA}"
INIT_SHA = "e9de2954556f7f39c98cc5e90de2e28550dad4ba656c986280918c929af1256d"
TEACHER_SHA = "14e644323b8e6a7b769dbf641d9f625bee895742b7064368b55afd0710a0d665"
INIT = f"/data/holosoma_checkpoint_cache/zihanw22_boxer_d9m3z369-recovered/by-sha256/{INIT_SHA}.pt"
TEACHER = f"/data/holosoma_checkpoint_cache/zihanw22_carry-any_ch2ckwzw/by-sha256/{TEACHER_SHA}.pt"
BANK = "/data/holosoma_inputs/ch2ckwzw_model13000_rollout137_precomputed_turn_forward_v1/by-source/95e5a54bb1e429874af7a93bfb6bb902b5fc9bd7564eab481d3bce5cd01c0303"
CONTACT = "/data/holosoma_inputs/ch2ckwzw_model13000_rollout137_contact_sidecars_20260828/by-source/506050566c020febc0308048d819a411b1b5263cbfab9ef3c73cd8d0c7d3aab4"
REGIONS = ["left_wrist", "right_wrist", "left_elbow", "right_elbow", "left_wrist_roll", "right_wrist_roll", "left_wrist_pitch", "right_wrist_pitch", "torso"]
ARMS = tuple(f"{regime}-{contact}+{sampling}" for regime in ("rl", "mix")
             for contact in ("nocontact", "contact") for sampling in ("uniform", "adaptive"))
MOTION_PREFIX = "command.setup-terms.motion-command.params.motion-config."


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def run(args, *, env=None, timeout=600):
    p = subprocess.run(list(map(str, args)), text=True, capture_output=True, env=env, timeout=timeout)
    if p.returncode:
        raise RuntimeError(f"Command failed ({p.returncode}): {args[0:3]}\n{p.stdout[-6000:]}\n{p.stderr[-6000:]}")
    return p.stdout


def arm_flags(arm):
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    regime, tail = arm.split("-", 1)
    contact, sampling = tail.split("+")
    return regime == "mix", contact == "contact", sampling == "adaptive"


def training_args(arm, mode, persist, run_id=None):
    mix, contact, adaptive = arm_flags(arm)
    args = json.loads((ROOT / "scripts/contact_sampling_factorial_base_cli.json").read_text())
    overrides = {
        "training.name": arm,
        "training.num-envs": 8 * 2048,
        "training.policy-init-actor-contract-migration": "box_tracking_to_precomputed_peak_height_drop_exclusive_v1",
        "algo.config.distill.enabled": mix,
        "algo.config.num-learning-iterations": 40000 if mode == "formal" else 2,
        "algo.config.save-interval": 500 if mode == "formal" else 2,
        "reward.terms.offline-contact-guidance.weight": 1.0 if contact else 0.0,
        "reward.terms.offline-contact-guidance.params.contact-weight": 1.0,
        "reward.terms.offline-contact-guidance.params.wrist-weight": 1.0,
        "reward.terms.offline-contact-guidance.params.contact-export-root": CONTACT,
        "reward.terms.offline-contact-guidance.params.contact-schedule-missing-mode": "inactive",
        "reward.terms.offline-contact-guidance.params.contact-region-names": json.dumps(REGIONS),
        "reward.terms.offline-contact-guidance.params.wrist-region-names": '["left_wrist","right_wrist"]',
        "robot.object.object-urdf-path": BANK + "/_clip_object_urdf_map.json",
        "logger.name": arm,
        "logger.mode": "online" if mode == "formal" else "offline",
        "logger.base-dir": str(Path(persist) / arm / mode / "training_logs"),
    }
    motion = {
        "motion-file": BANK,
        "contact-aware-button-window-mode": "peak_height",
        "contact-aware-carry-window-mode": "peak_height",
        "use-adaptive-timesteps-sampler": adaptive,
        "adaptive-sampling-contact-interval-root": CONTACT,
        "contact-interval-runtime-prepend-compensation": True,
        "uniform-t1-window-sampling-enabled": False,
        "start-at-timestep-zero-prob-end-iter": 39999,
        "freeze-at-timestep-zero-prob-end-iter": 39999,
    }
    overrides.update({MOTION_PREFIX + k: v for k, v in motion.items()})
    if mix:
        distill = {
            "mode": "dagger", "policy-to-clone": TEACHER, "teacher-obs-keys": '["actor_obs"]',
            "strict-teacher-load": True, "teacher-action-mix-ratio": 0.0,
            "take-teacher-actions": False, "teacher-use-stochastic-actions": False,
            "dagger-match-std": True, "dagger-loss-coef": 1.0, "dagger-replay-enabled": False,
            "ppo-start-epoch": 0, "ppo-start-coeff": 0.01, "ppo-target-coeff": 0.9,
            "ppo-schedule-step-epochs": 700, "dagger-end-epoch": 6300,
            "ppo-start-noise-std": "None", "fixed-bc-eval-log-interval": 100,
            "fixed-bc-guard-enabled": False,
        }
        overrides.update({"algo.config.distill." + k: v for k, v in distill.items()})
    if mode == "formal":
        if not run_id or not re.fullmatch("[a-z0-9]{8}", run_id):
            raise ValueError("Formal training requires a fresh run ID.")
        overrides.update({"logger.id": run_id, "logger.resume": "never"})
    # Replace keys instead of relying on duplicate CLI option precedence.
    args = [arg for arg in args if not (arg.startswith("--") and arg[2:].split("=", 1)[0] in overrides)]
    args.extend(f"--{k}={v}" for k, v in overrides.items())
    return args


def definition(arm):
    args = training_args(arm, "canary", "/canonical")
    return {"arm": arm, "cli": args, "initializer_sha256": INIT_SHA,
            "offline_producer": "ch2ckwzw/model_13000.pt", "label_teacher_sha256": TEACHER_SHA if arm_flags(arm)[0] else None}


def gpu_health():
    rows = run(["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total", "--format=csv,noheader,nounits"]).strip().splitlines()
    apps = run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"]).strip()
    if len(rows) != 8 or apps:
        raise RuntimeError(f"Expected 8 idle GPUs: {rows}, apps={apps}")
    for row in rows:
        _, name, memory, utilization, ecc = [x.strip() for x in row.split(",")]
        if name != "NVIDIA L40S" or int(memory) > 32 or int(utilization) > 1 or ecc != "0":
            raise RuntimeError(f"GPU not idle/healthy: {row}")
    return rows


def worker_environment(campaign, arm, mode, work, verification):
    env = {k: v for k, v in os.environ.items() if k in {
        "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "WANDB_API_KEY", "SSH_AUTH_SOCK", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"}}
    scratch = Path("/dev/shm") / ("holosoma_factorial_" + arm + "_" + mode)
    for part in ("tmp", "xdg", "robot", "object", "mesh", "derived", "provenance"):
        (scratch / part).mkdir(parents=True, exist_ok=True)
    env.update({
        "PATH": str(Path(PYTHON).parent) + ":/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "PYTHONPATH": f"{RUNTIME}:{ROOT}/src/holosoma:{ROOT}/src/holosoma_inference:{ROOT}/src",
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
        "PYTHON_RUNTIME_SITEPACKAGES": RUNTIME, "PYTHON_RUNTIME_MANIFEST_SHA256": RUNTIME_SHA,
        "HOLOSOMA_PYTHON_RUNTIME_MANIFEST_SHA256": RUNTIME_SHA, "HOLOSOMA_REQUIRE_PYTHON_RUNTIME_OVERLAY": "1",
        "NCCL_LIB_DIR": NCCL, "NCCL_LIB_SHA256": NCCL_SHA, "LD_LIBRARY_PATH": NCCL, "LD_PRELOAD": NCCL + "/libnccl.so.2",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "OMP_NUM_THREADS": "1",
        "OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y", "HEADLESS": "1",
        "TMPDIR": str(scratch / "tmp"), "XDG_CACHE_HOME": str(scratch / "xdg"),
        "HOLOSOMA_RUNTIME_SCRATCH_ROOT": str(scratch),
        "HOLOSOMA_ROBOT_USD_CACHE_DIR": str(scratch / "robot"), "HOLOSOMA_OBJECT_USD_CACHE_DIR": str(scratch / "object"),
        "HOLOSOMA_PERCEPTION_MESH_CACHE_DIR": str(scratch / "mesh"), "HOLOSOMA_DATA_PROVENANCE_CACHE_ROOT": str(scratch / "provenance"),
        "HOLOSOMA_ISAACSIM_KIT_ARGS": f"--/UJITSO/datastore/localCachePath={scratch}/derived --/UJITSO/datastore/localDataStore/largeChunkDiskBudgetMB=1024",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7", "NPROC": "8", "NNODES": "1", "NODE_RANK": "0",
        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(campaign["nodes"][arm]["port"]),
        "TORCH_DIST_BACKEND": "gloo", "TORCH_DIST_TIMEOUT_SEC": "1800", "GLOO_SOCKET_IFNAME": "lo", "NCCL_SOCKET_IFNAME": "lo",
        "NCCL_IB_DISABLE": "1", "NCCL_DEBUG": "WARN", "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "TORCH_NCCL_ENABLE_MONITORING": "1", "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "300",
        "HOLOSOMA_GLOO_GRAD_REDUCE": "0", "HOLOSOMA_GLOO_BARRIER": "1", "HOLOSOMA_GLOO_SMALL_COLLECTIVES": "1",
        "HOLOSOMA_HIERARCHICAL_GRAD_REDUCE": "1", "HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER": "1",
        "HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC": "300", "HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES": "0",
        "HOLOSOMA_RANK_VISIBLE_DEVICES": "1", "HOLOSOMA_RANK_LOCAL_CPU_AFFINITY": "1",
        "HOLOSOMA_SYNC_BEFORE_GRAD_ALLREDUCE": "1", "HOLOSOMA_SYNC_AFTER_GRAD_ALLREDUCE": "0",
        "HOLOSOMA_SYNC_AFTER_OPTIMIZER_STEP": "0", "HOLOSOMA_CONTIGUOUS_MINIBATCHES": "1",
        "HOLOSOMA_MINIBATCH_THROUGHPUT_CANARY": "1" if mode == "canary" else "0",
        "HOLOSOMA_SOURCE_ROOT": str(ROOT), "HOLOSOMA_SOURCE_SNAPSHOT_ID": "src-" + campaign["source"]["manifest"],
        "HOLOSOMA_SOURCE_MANIFEST_SHA256": campaign["source"]["manifest"],
        "HOLOSOMA_GIT_REMOTE_URL": REMOTE, "HOLOSOMA_GIT_REMOTE_REF": "main",
        "HOLOSOMA_GIT_COMMIT_SHA": campaign["source"]["commit"], "HOLOSOMA_GIT_TREE_SHA": campaign["source"]["tree"],
        "HOLOSOMA_FORMAL_GIT_VERIFICATION_PATH": str(verification),
        "MOTION_DIR": BANK, "OBJECT_SPEC_PATH": BANK + "/_clip_object_urdf_map.json", "OBJECT_URDF": BANK + "/_clip_object_urdf_map.json",
        "HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_DIR": BANK, "HOLOSOMA_EXTERNAL_AS_WORLD_SIZE": "8",
        "HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_SOURCE_DIGEST": campaign["dataset"]["single_slot_source_digest"],
        "HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_VIEW_DIGEST": campaign["dataset"]["single_slot_view_digest"],
        "HOLOSOMA_EXTERNAL_AS_RANK_SHARD_SOURCE_DIGEST": campaign["dataset"]["shard_digest"],
        "HOLOSOMA_RANK_LOCAL_MOTION_ROOT": campaign["dataset"]["shard_root"],
        "HOLOSOMA_MOTION_SHARD_MANIFEST": campaign["dataset"]["shard_root"] + "/manifest.json",
        "HOLOSOMA_RANK_LOCAL_SHARDING_ENABLED": "1", "HOLOSOMA_REQUIRE_RANK_LOCAL_SHARD_PROVENANCE": "1",
        "HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK": "0", "HOLOSOMA_OBJECT_SPAWN_MODE": "single_slot_multi_urdf",
        "HOLOSOMA_REQUIRE_SINGLE_SLOT_OBJECTS": "1", "HOLOSOMA_REQUIRE_OBJECT_MESH_ASSETS": "1",
        "HOLOSOMA_ALLOW_LEGACY_OBJECT_URDF_FALLBACK": "0", "HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE": "mesh",
        "HOLOSOMA_OBJECT_COLLIDER_TYPE": "convex_decomposition", "HOLOSOMA_ACTIVATE_OBJECT_CONTACT_SENSORS": "0",
        "HOLOSOMA_REQUIRE_CONTACT_INTERVAL_COVERAGE": "0", "HOLOSOMA_REQUIRE_CONTACT_TARGET_COVERAGE": "0",
        "HOLOSOMA_MOTION_METRICS_INTERVAL": "16", "HOLOSOMA_SKIP_INITIAL_CHECKPOINT": "1",
        "HOLOSOMA_DISABLE_AUTO_RESET": "0", "HOLOSOMA_DISABLE_CLIP_END_RESET": "0", "HOLOSOMA_DISABLE_MOTION_END_RESET": "0",
        "HOLOSOMA_PERCEPTION_INJECT_INTO_POLICY_MODULES": "True", "HOLOSOMA_PERCEPTION_INCLUDE_ROBOT_MESH": "1",
        "CONTACT_EXPORT_ROOT": CONTACT, "CONTACT_SIDECAR_MODE": "runtime-intervals",
        "REQUIRE_MOTION_GENERATOR_TEACHER_MATCH": "0",
        "HOLOSOMA_EXTERNAL_AS_MOTION_GENERATOR_TEACHER_SHA256": "78bd1ad6143edf93a50e2f58a366c64729bc575d8364a25c78af76e933e7f1f4",
        "WANDB_DIR": str(work / "wandb"), "WANDB_ENTITY": "zihanw22", "WANDB_CONSOLE": "off",
        "WANDB_MODE": "online" if mode == "formal" else "disabled", "HOLOSOMA_REQUIRE_WANDB_RUN": "1" if mode == "formal" else "0",
    })
    if mode != "formal":
        env["WANDB_DISABLED"] = "true"
    return env


def worker(campaign_path, arm, mode):
    campaign = json.loads(Path(campaign_path).read_text())
    if campaign["definitions"][arm] != json_sha(definition(arm)):
        raise ValueError("Git-bound experiment definition drift")
    host = campaign["nodes"][arm]["ip"]
    if host not in run(["hostname", "-I"]).split():
        raise ValueError("Wrong host for this arm")
    work = Path(campaign["persist"]) / arm / mode
    work.mkdir(parents=True, exist_ok=True)
    (work / "wandb").mkdir(exist_ok=True)
    verification = work / "git_verification.json"
    run([sys.executable, ROOT / "scripts/verify_formal_git_checkout.py", "--source-root", ROOT,
         "--remote-url", REMOTE, "--remote-ref", "main", "--commit", campaign["source"]["commit"],
         "--tree", campaign["source"]["tree"], "--output", verification])
    if sha(NCCL + "/libnccl.so.2") != NCCL_SHA or sha(INIT) != INIT_SHA or sha(TEACHER) != TEACHER_SHA:
        raise ValueError("Checkpoint/runtime checksum mismatch")
    save(work / "gpu_preflight.json", {"rows": gpu_health()})
    for item in campaign["assets"]:
        path = Path(item["path"])
        if not path.is_file() or sha(path) != item["sha256"]:
            raise ValueError(f"Missing or changed asset: {path}")
    env = worker_environment(campaign, arm, "canary" if mode == "preflight" else mode, work, verification)
    run([PYTHON, ROOT / "scripts/verify_python_runtime_overlay.py", "--site-packages", RUNTIME,
         "--manifest-sha256", RUNTIME_SHA, "--require-distribution-closure", "--require-current-runtime-binding"], env=env)
    run_id = None
    if mode == "formal":
        acceptance = json.loads((Path(campaign["persist"]) / arm / "canary_acceptance.json").read_text())
        contract = json.loads((Path(campaign["persist"]) / arm / "run_contract.json").read_text())
        if acceptance.get("accepted") is not True or acceptance["commit"] != campaign["source"]["commit"] or acceptance["definition_sha256"] != campaign["definitions"][arm]:
            raise ValueError("Missing exact-source/config canary acceptance")
        if contract["canary_acceptance_sha256"] != sha(Path(campaign["persist"]) / arm / "canary_acceptance.json"):
            raise ValueError("Canary acceptance changed")
        if contract["campaign_sha256"] != sha(campaign_path):
            raise ValueError("Campaign identity changed")
        run_id = contract["run_id"]
    cli = training_args(arm, "formal" if mode == "formal" else "canary", campaign["persist"], run_id)
    if mode == "formal" and contract["cli"] != cli:
        raise ValueError("Final CLI differs from immutable formal contract")
    save(work / "training_cli.json", cli)
    run([PYTHON, ROOT / "scripts/validate_train_cli.py", "--expected-motion-end-mode", "episodic", "--", *cli], env=env)
    mix = arm_flags(arm)[0]
    provenance_args = [PYTHON, ROOT / "scripts/compute_training_provenance.py", "--training-regime", "distillation" if mix else "pure_rl",
        "--motion-dir", BANK, "--object-map", BANK + "/_clip_object_urdf_map.json", "--contact-root", CONTACT,
        "--contact-sidecar-mode", "runtime-intervals", "--motion-shard-manifest", env["HOLOSOMA_MOTION_SHARD_MANIFEST"],
        "--policy-init-checkpoint", INIT, "--contact-interval-runtime-prepend-compensation", "true", "--source-root", ROOT]
    if mix:
        provenance_args += ["--teacher-checkpoint", TEACHER, "--student-motion-end-mode", "episodic"]
    provenance = json.loads(run(provenance_args, env=env))
    provenance["factorial_experiment"] = {"arm": arm, "definition_sha256": campaign["definitions"][arm],
        "campaign_sha256": sha(campaign_path), "source": campaign["source"], "node_git_verification": json.loads(verification.read_text()),
        "offline_producer": "ch2ckwzw/model_13000.pt", "online_teacher": "ch2ckwzw/model_40000.pt" if mix else None}
    if mode == "formal":
        provenance["factorial_experiment"]["formal_contract_sha256"] = sha(Path(campaign["persist"]) / arm / "run_contract.json")
    env["HOLOSOMA_TRAINING_PROVENANCE"] = json.dumps(provenance, sort_keys=True, separators=(",", ":"))
    save(work / "provenance.json", provenance)
    preflight = [PYTHON, ROOT / "scripts/box23k_policy_init_preflight.py", "--world-size", "8", "--output", work / "initializer_preflight.json"]
    if mix:
        preflight += ["--allow-distillation"]
    run([*preflight, "--", *cli], env=env)
    print(f"[INFO] factorial_preflight_passed arm={arm} mode={mode} ws=8 envs_per_gpu=2048", flush=True)
    if mode == "preflight":
        return
    gpu_health()
    os.chdir(ROOT)
    os.execve(PYTHON, [PYTHON, "-m", "torch.distributed.run", "--nnodes=1", "--node_rank=0", "--master_addr=127.0.0.1",
        "--nproc_per_node=8", "--max_restarts=0", "--master_port=" + env["MASTER_PORT"],
        "src/holosoma/holosoma/train_agent_rank_visible.py", *cli], env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "canary", "formal"))
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    args = parser.parse_args()
    worker(args.campaign, args.arm, args.mode)


if __name__ == "__main__":
    main()
