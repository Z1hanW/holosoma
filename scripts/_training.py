"""Shared launch plumbing for the two public training scripts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shlex
from string import Template
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def bind_nccl(env):
    """Pin the installed collective library before child processes import Torch."""
    package = importlib.metadata.distribution("nvidia-nccl-cu12")
    libraries = [Path(package.locate_file(file)).resolve() for file in package.files or []
                 if str(file).endswith("/libnccl.so.2")]
    if len(libraries) != 1:
        raise ValueError("Install the nvidia-nccl-cu12 library matching PyTorch before training")
    library = libraries[0]
    with library.open("rb") as stream:
        env["NCCL_LIB_SHA256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    env["NCCL_LIB_DIR"] = str(library.parent)
    env["LD_PRELOAD"] = str(library)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [str(library.parent), env.get("LD_LIBRARY_PATH")]))


def run(command, env, *, capture=False):
    return subprocess.run(
        [str(value) for value in command], cwd=ROOT, env=env, check=True,
        text=True, stdout=subprocess.PIPE if capture else None,
    ).stdout


def prepare(role, cli_template, environment, nodes, argv=None):
    parser = argparse.ArgumentParser(
        prog="train_teacher.sh" if role == "teacher" else "train_student.sh",
        description=f"Train {role}: {nodes} node(s), 8 GPUs per node, 2048 environments per GPU."
    )
    parser.add_argument("--motion-bank", type=Path, required=True,
                        help="Prepared single-slot motion bank, including its rank shards")
    if role == "distillation":
        parser.add_argument("--contact-bank", type=Path, required=True)
        parser.add_argument("--robot-assets", type=Path, required=True)
        parser.add_argument("--teacher-checkpoint", type=Path, required=True)
        parser.add_argument("--initializer-checkpoint", type=Path, required=True,
                            help="Box policy actor initializer used by 4kocpixa")
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"), required=not os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--project", default="carry")
    parser.add_argument("--name", default=role)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / role)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", default="127.0.0.1" if nodes == 1 else None)
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--source-ref", default="release/4kocpixa-clean")
    parser.add_argument("--source-commit", help="Full remote Git SHA, identical on every node")
    parser.add_argument("--check", action="store_true",
                        help="Parse and print the training command on CPU, without starting training")
    args = parser.parse_args(argv)
    if not 0 <= args.node_rank < nodes:
        parser.error(f"--node-rank must be in [0, {nodes - 1}]")
    if not 1024 <= args.master_port < 65535:
        parser.error("--master-port must be between 1024 and 65534")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.name):
        parser.error("--name must be a simple run name without spaces or slashes")
    if not args.check:
        if not args.master_addr:
            parser.error("--master-addr must be the address of node 0")
        if not args.source_commit or not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
            parser.error("--source-commit must be the full SHA pushed to --source-ref")
    bindings = {
        "MOTION_BANK": str(args.motion_bank.expanduser().resolve()),
        "TEACHER_MOTION_BANK": str(args.motion_bank.expanduser().resolve()),
        "WANDB_ENTITY": args.entity, "WANDB_PROJECT": args.project,
        "RUN_NAME": args.name, "LOG_DIR": str(args.output.expanduser().resolve()),
    }
    if role == "distillation":
        for key in ("contact_bank", "robot_assets", "teacher_checkpoint", "initializer_checkpoint"):
            bindings[key.upper()] = str(getattr(args, key).expanduser().resolve())
    cli = [Template(value).substitute(bindings) for value in cli_template]
    # Do not inherit semantic switches or an existing training/logging identity.
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("HOLOSOMA_", "WANDB_")) and key not in {
               "CONTACT_EXPORT_ROOT", "CONTACT_SIDECAR_MODE", "FORCE_EIGHT_GPU_CONFIG",
               "PERCEPTION_INJECT_INTO_POLICY_MODULES", "RESET_TO_DEFAULT_POSE",
               "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
           }}
    if os.environ.get("WANDB_API_KEY"):
        env["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
    env.update({key: Template(value).substitute(bindings) for key, value in environment.items()})
    env.update({
        "PYTHONPATH": os.pathsep.join([str(ROOT / "src/holosoma"), str(ROOT / "src/holosoma_inference"), str(ROOT)]),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7"),
        "NODE_RANK": str(args.node_rank), "MASTER_ADDR": args.master_addr or "NODE_0_ADDRESS",
        "MASTER_PORT": str(args.master_port), "HEADLESS": "1",
        "MOTION_DIR": bindings["MOTION_BANK"],
        "OBJECT_SPEC_PATH": bindings["MOTION_BANK"] + "/_clip_object_urdf_map.json",
        "HOLOSOMA_SOURCE_ROOT": str(ROOT),
    })
    devices = env["CUDA_VISIBLE_DEVICES"].split(",")
    if len(devices) != 8 or len(set(devices)) != 8 or any(not device.strip() for device in devices):
        parser.error("CUDA_VISIBLE_DEVICES must select eight distinct GPUs")
    command = [sys.executable, "-m", "torch.distributed.run", f"--nnodes={nodes}",
               "--nproc_per_node=8", f"--node_rank={args.node_rank}", "--max_restarts=0",
               f"--master_addr={env['MASTER_ADDR']}", f"--master_port={args.master_port}",
               str(ROOT / "src/holosoma/holosoma/train_agent_rank_visible.py"), *cli]
    return args, bindings, cli, env, command


def launch(role, cli_template, environment, *, nodes):
    args, bindings, cli, env, command = prepare(role, cli_template, environment, nodes)
    run([sys.executable, ROOT / "scripts/validate_train_cli.py",
         "--expected-motion-end-mode", "episodic", "--", *cli], env)
    if args.check:
        print(shlex.join(command))
        print("CLI check passed; no assets, Git remote, GPU or W&B run were accessed.")
        return

    output = Path(bindings["LOG_DIR"])
    # A distinct directory per node keeps preflight files from racing on shared storage.
    work = output / "launch" / f"node_{args.node_rank}"
    work.mkdir(parents=True, exist_ok=False)
    remote = run(["git", "remote", "get-url", "origin"], env, capture=True).strip()
    tree = run(["git", "rev-parse", args.source_commit + "^{tree}"], env, capture=True).strip()
    verification = work / "git.json"
    run([sys.executable, ROOT / "scripts/verify_formal_git_checkout.py", "--source-root", ROOT,
         "--remote-url", remote, "--remote-ref", args.source_ref,
         "--commit", args.source_commit, "--tree", tree, "--output", verification], env)
    env["HOLOSOMA_FORMAL_GIT_VERIFICATION_PATH"] = str(verification)
    bind_nccl(env)
    provenance_args = [sys.executable, ROOT / "scripts/compute_training_provenance.py",
                       "--training-regime", "pure_rl" if role == "teacher" else "distillation",
                       "--motion-dir", bindings["MOTION_BANK"], "--object-map", env["OBJECT_SPEC_PATH"],
                       "--motion-shard-manifest", env["HOLOSOMA_MOTION_SHARD_MANIFEST"],
                       "--contact-interval-runtime-prepend-compensation", "false" if role == "teacher" else "true",
                       "--source-root", ROOT]
    if role == "distillation":
        provenance_args += ["--contact-root", bindings["CONTACT_BANK"], "--contact-sidecar-mode", "runtime-intervals",
                            "--policy-init-checkpoint", bindings["INITIALIZER_CHECKPOINT"],
                            "--teacher-checkpoint", bindings["TEACHER_CHECKPOINT"], "--student-motion-end-mode", "episodic"]
    provenance = json.loads(run(provenance_args, env, capture=True))
    env["HOLOSOMA_TRAINING_PROVENANCE"] = json.dumps(provenance, sort_keys=True, separators=(",", ":"))
    (work / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (work / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    if role == "distillation":
        run([sys.executable, ROOT / "scripts/box23k_policy_init_preflight.py", "--world-size", "8",
             "--allow-distillation", "--output", work / "initializer.json", "--", *cli], env)
    else:
        run([sys.executable, Path(__file__), "--teacher-onnx", str(work / "teacher.onnx"), *cli], env)
    os.chdir(ROOT)
    os.execve(sys.executable, command, env)


def teacher_onnx_preflight(path, cli):
    """Export and check the fresh teacher actor before starting the simulator."""
    import dataclasses
    import torch
    import tyro
    from holosoma.agents.modules.module_utils import setup_ppo_actor_module
    from holosoma.config_values.experiment import AnnotatedExperimentConfig
    from holosoma.observation.config_utils import apply_observation_overrides
    from holosoma.perception.config_utils import apply_perception_overrides
    from holosoma.utils.inference_helpers import export_policy_as_onnx, validate_exported_policy_onnx
    from holosoma.utils.runtime_asset_manifest import finalize_runtime_asset_provenance
    from holosoma.utils.tyro_utils import TYRO_CONIFG

    torch.set_num_threads(2)
    config = tyro.cli(AnnotatedExperimentConfig, args=cli, config=TYRO_CONIFG)
    config = dataclasses.replace(config, training=dataclasses.replace(config.training, num_envs=2048))
    config = apply_perception_overrides(apply_observation_overrides(config))
    finalize_runtime_asset_provenance(config)
    torch.manual_seed(config.training.seed)
    actor = setup_ppo_actor_module({"actor_obs": 178}, config.algo.config.module_dict.actor,
                                   29, 0.01, "cpu", {"actor_obs": 1})

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.policy = actor

        def forward(self, actor_obs):
            return self.policy.act_inference({"actor_obs": actor_obs})

    wrapper = Wrapper().eval()
    example = {"actor_obs": torch.randn(14, 178)}
    export_policy_as_onnx(wrapper, path, example)
    report = validate_exported_policy_onnx(wrapper=wrapper, onnx_file_path=path, example_obs_dict=example)
    Path(path).with_suffix(".json").write_text(json.dumps({"scope": "fresh_teacher_actor", "parity": report}, indent=2) + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 4 or sys.argv[1] != "--teacher-onnx":
        raise SystemExit("Use train_teacher.sh or train_student.sh")
    teacher_onnx_preflight(sys.argv[2], sys.argv[3:])
