#!/usr/bin/env python3
"""Prepare assets and launch independent eight-GPU factorial arms via remote Git."""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import contact_sampling_factorial as exp

NODES = [
    ("zzzihanw-26", "10.99.0.18"), ("zzzihanw-27", "10.99.0.227"),
    ("zzzihanw-34", "10.99.0.167"), ("zzzihanw-35", "10.99.0.77"),
    ("zzzihanw-39", "10.99.0.39"), ("zzzihanw-45", "10.99.0.183"),
    ("zzzihanw-47", "10.99.0.24"), ("zzzihanw-65", "10.99.0.201"),
]


def ssh(host, command, timeout=900):
    return exp.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "-o", "LogLevel=ERROR", "ubuntu@" + host, command], timeout=timeout)


def prepare(root):
    from prepare_as_rank_shards import compute_rank_shard_source_digest, prepare_rank_shards
    from holosoma.utils.runtime_asset_manifest import build_urdf_asset_manifest
    root.mkdir(parents=True, exist_ok=True)
    if (root / "campaign.json").exists():
        raise ValueError("Refusing to replace an existing campaign")
    bank, object_map = Path(exp.BANK), Path(exp.BANK) / "_clip_object_urdf_map.json"
    if exp.sha(bank / "manifest.json") != "f162e31fa38a63ac679158184d8a9f4864e17ae83dd1f8e7aa001b9e64619487":
        raise ValueError("Unexpected source command bank")
    teacher_source = Path("/data/holosoma_training_audits/ch2_40k_rollout137_20260908/checkpoint/model_40000.pt")
    if exp.sha(teacher_source) != exp.TEACHER_SHA:
        raise ValueError("Teacher checksum mismatch")
    teacher = Path(exp.TEACHER)
    teacher.parent.mkdir(parents=True, exist_ok=True)
    if not teacher.exists():
        shutil.copy2(teacher_source, teacher)
        teacher.chmod(0o444)
    if exp.sha(teacher) != exp.TEACHER_SHA:
        raise ValueError("Existing teacher checksum mismatch")
    shard_digest = compute_rank_shard_source_digest(motion_dir=bank, object_map=object_map, world_size=8, environments_per_rank=2048)
    shard_root = bank / "_rank_shards/by-source" / shard_digest / "ws8"
    shard = prepare_rank_shards(motion_dir=bank, object_map=object_map, output_root=shard_root,
                               world_size=8, environments_per_rank=2048, expected_source_digest=shard_digest)
    if shard["clip_count"] != 137 or not shard["exact_clip_partition"] or set(shard["clip_cover_counts"].values()) != {1}:
        raise ValueError("Invalid 137-clip closure")
    files = set()

    def record(path):
        path = Path(path).absolute()
        if path.is_file():
            files.add(path)

    for path in bank.glob("*.npz"):
        record(path)
    record(bank / "manifest.json")
    record(object_map)
    mapping = json.loads(object_map.read_text())["clips"]
    for clip, spec in mapping.items():
        urdf = bank / spec["object_urdf_path"]
        build_urdf_asset_manifest(urdf, role=f"object.{clip}", require_mesh=True, identity_recorder=record)
    for path in shard_root.rglob("*"):
        record(path)
    for path in Path(exp.CONTACT).rglob("*"):
        record(path)
    binding = json.loads((exp.ROOT / "scripts/box23k_robot_assets.json").read_text())
    for path in Path(binding["asset_root"]).rglob("*"):
        record(path)
    record(exp.INIT)
    record(exp.TEACHER)
    assets = []
    for path in sorted(files):
        if path.suffix.lower() in {".py", ".pyc", ".sh", ".yaml", ".yml", ".so"}:
            raise ValueError(f"Executable code in asset transfer: {path}")
        assets.append({"path": str(path), "sha256": exp.sha(path), "size": path.stat().st_size})
    commit = exp.run(["git", "-C", exp.ROOT, "rev-parse", "HEAD"]).strip()
    tree = exp.run(["git", "-C", exp.ROOT, "rev-parse", "HEAD^{tree}"]).strip()
    import hashlib
    manifest = hashlib.sha256(subprocess.check_output(["git", "-C", str(exp.ROOT), "ls-tree", "-r", "--full-tree", commit])).hexdigest()
    source = {"remote_url": exp.REMOTE, "remote_ref": "main", "commit": commit, "tree": tree, "manifest": manifest}
    campaign = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source": source, "checkout": "/data/holosoma_git/contact_sampling_" + commit[:12], "persist": str(root),
        "definitions": {arm: exp.json_sha(exp.definition(arm)) for arm in exp.ARMS},
        "nodes": {arm: {"alias": alias, "ip": ip, "az": "ap-northeast-2a", "port": 36820 + index}
                  for index, (arm, (alias, ip)) in enumerate(zip(exp.ARMS, NODES, strict=True))},
        "dataset": {"bank": exp.BANK, "contact_root": exp.CONTACT, "clip_count": 137,
                    "single_slot_source_digest": "42903c7e443ccd836af133700058b0772545efcbc5af11d3193b60f0ec72dddd",
                    "single_slot_view_digest": Path(exp.BANK).name, "shard_digest": shard_digest,
                    "shard_root": str(shard_root), "shard_manifest_sha256": exp.sha(shard_root / "manifest.json"),
                    "rank_clip_counts": [s["clip_count"] for s in shard["shards"]]},
        "assets": assets,
    }
    exp.save(root / "campaign.json", campaign)
    (root / "asset_files.txt").write_text("".join(x["path"].lstrip("/") + "\n" for x in assets))
    print(json.dumps({"campaign": str(root / "campaign.json"), "asset_files": len(assets),
                      "asset_bytes": sum(x["size"] for x in assets), "rank_clip_counts": campaign["dataset"]["rank_clip_counts"]}), flush=True)


def install(root, campaign, arm):
    host, source = campaign["nodes"][arm]["ip"], campaign["checkout"]
    commit, tree = campaign["source"]["commit"], campaign["source"]["tree"]
    command = f"""set -eu
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
if ! test -d {source}/.git; then
 git clone --reference-if-able /home/ubuntu/FAR/holosoma --dissociate --no-checkout {exp.REMOTE} {source}
fi
git -C {source} fetch origin main
git -C {source} checkout --detach {commit}
git -C {source} submodule update --init --recursive --jobs 4 -- submodules/PointTransformerV3 submodules/defm
python3 {source}/scripts/verify_formal_git_checkout.py --source-root {source} --remote-url {exp.REMOTE} --remote-ref main --commit {commit} --tree {tree}
mkdir -p {root}
"""
    output = ssh(host, command)
    exp.save(root / arm / "install.json", {"output": output})
    print("Git installed", arm, host, flush=True)


def sync_assets(root, campaign, arm):
    host = campaign["nodes"][arm]["ip"]
    exp.run(["rsync", "-aL", "--ignore-existing", "--files-from=" + str(root / "asset_files.txt"),
             "--relative", "/", "ubuntu@" + host + ":/"], timeout=1800)
    exp.run(["scp", "-q", root / "campaign.json", "ubuntu@" + host + ":" + str(root / "campaign.json")])
    print("Assets synchronized", arm, host, flush=True)


def worker_command(root, campaign, arm, mode):
    return shlex.join([exp.PYTHON, campaign["checkout"] + "/scripts/contact_sampling_factorial.py", mode,
                       "--campaign", str(root / "campaign.json"), "--arm", arm])


def preflight(root, campaign, arm):
    output = ssh(campaign["nodes"][arm]["ip"], worker_command(root, campaign, arm, "preflight"), timeout=1800)
    exp.save(root / arm / "preflight_result.json", {"output": output})
    print("Preflight accepted", arm, flush=True)


def launch(root, campaign, arm, mode):
    if mode == "formal":
        acceptance = json.loads((root / arm / "canary_acceptance.json").read_text())
        contract = json.loads((root / arm / "run_contract.json").read_text())
        if not acceptance["accepted"] or acceptance["commit"] != campaign["source"]["commit"]:
            raise ValueError("Formal launch requires accepted exact-source canary")
        if contract["cli"] != exp.training_args(arm, "formal", root, contract["run_id"]):
            raise ValueError("Formal CLI drift")
    host = campaign["nodes"][arm]["ip"]
    session = "factorial_" + arm.replace("+", "_") + "_" + mode
    log = root / arm / (mode + ".log")
    body = worker_command(root, campaign, arm, mode) + " >" + shlex.quote(str(log)) + " 2>&1"
    body += '; rc=$?; printf "%s\\n" "$rc" >' + shlex.quote(str(log) + ".exit") + '; exit "$rc"'
    command = f"""set -eu
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
test ! -e {shlex.quote(str(log))}
! tmux has-session -t {shlex.quote(session)} 2>/dev/null
mkdir -p {shlex.quote(str(root / arm))}
tmux new-session -d -s {shlex.quote(session)} {shlex.quote('bash -lc ' + shlex.quote(body))} 8>&-
"""
    ssh(host, command)
    print("Launched", mode, arm, host, session, flush=True)


def status(root, campaign, arm, mode):
    host = campaign["nodes"][arm]["ip"]
    code = """import json,subprocess,re
from pathlib import Path
root=Path(ROOT); log=root/ARM/(MODE+'.log'); exitfile=Path(str(log)+'.exit')
text=log.read_text(errors='replace') if log.exists() else ''
progress=[line for line in text.splitlines() if 'Learning iteration' in line or 'HOLOSOMA_PROGRESS' in line]
apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True)
print(json.dumps({'exit':exitfile.read_text().strip() if exitfile.exists() else None,'tail':text.splitlines()[-8:],
                 'progress':progress[-2:],'gpu_apps':len(apps.stdout.strip().splitlines()) if apps.stdout.strip() else 0}))
""".replace("ROOT", repr(str(root))).replace("ARM", repr(arm)).replace("MODE", repr(mode))
    result = json.loads(ssh(host, "python3 -c " + shlex.quote(code)))
    exp.save(root / arm / (mode + "_status.json"), result)
    print(arm, json.dumps(result), flush=True)


def gather(root, campaign, arm):
    host = campaign["nodes"][arm]["ip"]
    code = "from pathlib import Path;import json;r=Path(" + repr(str(root / arm / "canary")) + ");print(json.dumps({n:[str(p) for p in r.rglob(n) if '.wandb' not in p.parts] for n in ['model_00002.pt','model_00002.onnx','model_00002.pair.json','holosoma_config.yaml','initializer_preflight.json','git_verification.json','train_rank_*.log']}))"
    paths = json.loads(ssh(host, "python3 -c " + shlex.quote(code)))
    if ssh(host, "cat " + shlex.quote(str(root / arm / "canary.log.exit"))).strip() != "0":
        raise ValueError("Canary did not exit successfully")
    dest = root / arm / "canary_artifacts"
    dest.mkdir(parents=True, exist_ok=True)
    for name, matches in paths.items():
        if len(matches) != (8 if name == "train_rank_*.log" else 1):
            raise ValueError(f"Wrong canary artifact multiplicity: {name}={matches}")
        for path in matches:
            exp.run(["scp", "-q", f"ubuntu@{host}:{path}", dest / Path(path).name], timeout=300)
    exp.run(["scp", "-q", f"ubuntu@{host}:{root / arm / 'canary.log'}", dest / "controller.log"])
    print("Gathered", arm, flush=True)


def contract(root, campaign, arm):
    import wandb
    path = root / arm / "run_contract.json"
    if path.exists():
        raise ValueError("Refusing to replace formal identity")
    acceptance_path = root / arm / "canary_acceptance.json"
    acceptance = json.loads(acceptance_path.read_text())
    if not acceptance["accepted"] or acceptance["commit"] != campaign["source"]["commit"]:
        raise ValueError("Invalid canary")
    run_id = wandb.util.generate_id()
    result = {"arm": arm, "run_id": run_id, "name": arm, "fresh": True, "resume": None,
        "campaign_sha256": exp.sha(root / "campaign.json"), "canary_acceptance_sha256": exp.sha(acceptance_path),
        "source": campaign["source"], "node_verification": acceptance["git_verification"],
        "dataset": campaign["dataset"], "training_export_onnx": True, "save_interval": 500, "target_updates": 40000,
        "canary_weights_loaded": False, "formal_launch_video_required": False,
        "cli": exp.training_args(arm, "formal", root, run_id)}
    exp.save(path, result)
    path.chmod(0o444)
    host = campaign["nodes"][arm]["ip"]
    for item in (path, acceptance_path):
        exp.run(["scp", "-q", item, f"ubuntu@{host}:{item}"])
    print("Reserved fresh identity", arm, run_id, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "install", "sync", "preflight", "canary", "formal", "status_canary", "status_formal", "gather", "contract"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=exp.ARMS)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.stage == "prepare":
        prepare(root)
        return
    campaign = json.loads((root / "campaign.json").read_text())
    for arm in args.arms or exp.ARMS:
        if args.stage in {"canary", "formal"}:
            launch(root, campaign, arm, args.stage)
        elif args.stage.startswith("status_"):
            status(root, campaign, arm, args.stage.removeprefix("status_"))
        else:
            {"install": install, "sync": sync_assets, "preflight": preflight, "gather": gather, "contract": contract}[args.stage](root, campaign, arm)


if __name__ == "__main__":
    main()
