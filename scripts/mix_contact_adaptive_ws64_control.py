#!/usr/bin/env python3
"""Controller for the isolated 216-clip, 64-GPU mixed training launch."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

import contact_sampling_factorial as recipe
from contact_sampling_factorial_control import ssh
import mix_contact_adaptive_ws64 as exp

IPS = ["10.99.0." + x for x in ("18", "227", "167", "77", "116", "117", "165", "176")]


def prepare(root):
    if (root / "campaign.json").exists():
        raise ValueError("Campaign already exists")
    dataset = json.loads((root / "dataset_identity.json").read_text())
    if dataset["clip_count"] != 216 or dataset["source_counts"] != {"corl79": 79, "ch2_40k_rollout137": 137}:
        raise ValueError("Unexpected data union")
    inventory = json.loads((root / "node_inventory.json").read_text())
    for host in IPS:
        if inventory[host]["apps"] or inventory[host]["az"] != "ap-northeast-2a":
            raise ValueError(f"Selected node was not idle in ap-northeast-2a: {host}")
    recipe.run(["git", "-C", recipe.ROOT, "fetch", "origin", "main"])
    commit = recipe.run(["git", "-C", recipe.ROOT, "rev-parse", "HEAD"]).strip()
    recipe.run(["git", "-C", recipe.ROOT, "merge-base", "--is-ancestor", commit, "origin/main"])
    tree = recipe.run(["git", "-C", recipe.ROOT, "rev-parse", "HEAD^{tree}"]).strip()
    manifest = hashlib.sha256(subprocess.check_output(["git", "-C", str(recipe.ROOT), "ls-tree", "-r", "--full-tree", commit])).hexdigest()
    files = set()
    for directory in [Path(dataset["bank"]), Path(dataset["contact_root"]), Path(json.loads((recipe.ROOT / "scripts/box23k_robot_assets.json").read_text())["asset_root"])]:
        files.update(p.absolute() for p in directory.rglob("*") if p.is_file())
    files.update([Path(recipe.INIT), Path(recipe.TEACHER)])
    assets = []
    for path in sorted(files):
        if path.suffix.lower() in {".py", ".pyc", ".sh", ".yaml", ".yml", ".so"}:
            raise ValueError(f"Executable file in data transfer: {path}")
        assets.append({"path": str(path), "sha256": recipe.sha(path), "size": path.stat().st_size})
    campaign = {"created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "source": {"remote_url": recipe.REMOTE, "remote_ref": "main", "commit": commit, "tree": tree, "manifest": manifest},
                "checkout": "/data/holosoma_git/mix216_append02_" + commit[:12], "persist": str(root),
                "dataset": dataset, "assets": assets, "port": 36920,
                "nodes": [{"ip": host, "az": "ap-northeast-2a", "interface": "enp135s0"} for host in IPS]}
    campaign["definition_sha256"] = recipe.json_sha(exp.definition(campaign))
    campaign["definitions"] = {exp.ARM: campaign["definition_sha256"]}
    recipe.save(root / "campaign.json", campaign)
    (root / "asset_files.txt").write_text("".join(x["path"].lstrip("/") + "\n" for x in assets))
    print(json.dumps({"commit": commit, "asset_files": len(assets), "asset_bytes": sum(x["size"] for x in assets)}), flush=True)


def install(root, campaign, rank):
    host, source = campaign["nodes"][rank]["ip"], campaign["checkout"]
    commit, tree = campaign["source"]["commit"], campaign["source"]["tree"]
    output = ssh(host, f"""set -eu
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
if ! test -d {source}/.git; then
 git clone --reference-if-able /home/ubuntu/FAR/holosoma --dissociate --no-checkout {recipe.REMOTE} {source}
fi
git -C {source} fetch origin main
git -C {source} checkout --detach {commit}
git -C {source} submodule update --init --recursive --jobs 4 -- submodules/PointTransformerV3 submodules/defm
python3 {source}/scripts/verify_formal_git_checkout.py --source-root {source} --remote-url {recipe.REMOTE} --remote-ref main --commit {commit} --tree {tree}
mkdir -p {root}
""")
    recipe.save(root / f"node_{rank}" / "install.json", {"output": output})
    print("Installed exact Git", rank, host, flush=True)


def sync(root, campaign, rank):
    host = campaign["nodes"][rank]["ip"]
    recipe.run(["rsync", "-aL", "--ignore-existing", "--files-from=" + str(root / "asset_files.txt"),
                "--relative", "/", "ubuntu@" + host + ":/"], timeout=3600)
    recipe.run(["scp", "-q", root / "campaign.json", f"ubuntu@{host}:{root}/campaign.json"])
    print("Assets synchronized", rank, host, flush=True)


def command(root, campaign, rank, mode):
    return shlex.join([recipe.PYTHON, campaign["checkout"] + "/scripts/mix_contact_adaptive_ws64.py", mode,
                       "--campaign", str(root / "campaign.json"), "--node-rank", str(rank)])


def preflight(root, campaign, rank, mode="preflight"):
    output = ssh(campaign["nodes"][rank]["ip"], command(root, campaign, rank, mode), timeout=2400)
    recipe.save(root / f"node_{rank}" / (mode + ".json"), {"output": output, "commit": campaign["source"]["commit"]})
    print("Preflight accepted", rank, mode, flush=True)


def launch(root, campaign, rank, mode):
    for index in range(8):
        proof = root / f"node_{index}" / ("formal-preflight.json" if mode == "formal" else "preflight.json")
        if json.loads(proof.read_text())["commit"] != campaign["source"]["commit"]:
            raise ValueError("Missing all-node preflight")
    session = "mix216_append02_ws64_" + mode
    log = root / exp.ARM / f"{mode}_node_{rank}.log"
    body = command(root, campaign, rank, mode) + " >" + shlex.quote(str(log)) + " 2>&1"
    body += '; rc=$?; printf "%s\\n" "$rc" >' + shlex.quote(str(log) + ".exit") + '; exit "$rc"'
    ssh(campaign["nodes"][rank]["ip"], f"""set -eu
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
test ! -e {shlex.quote(str(log))}
! tmux has-session -t {session} 2>/dev/null
mkdir -p {shlex.quote(str(log.parent))}
tmux new-session -d -s {session} {shlex.quote('bash -lc ' + shlex.quote(body))} 8>&-
""")
    print("Launched", mode, rank, campaign["nodes"][rank]["ip"], flush=True)


def status(root, campaign, rank, mode):
    log = root / exp.ARM / f"{mode}_node_{rank}.log"
    code = f"""from pathlib import Path
import json,subprocess
p=Path({str(log)!r}); text=p.read_text(errors='replace') if p.exists() else ''
exitfile=Path(str(p)+'.exit')
progress=[x for x in text.splitlines() if 'Learning iteration' in x or 'HOLOSOMA_PROGRESS' in x]
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip().splitlines()
print(json.dumps({{'exit':exitfile.read_text().strip() if exitfile.exists() else None,'progress':progress[-2:],'tail':text.splitlines()[-5:],'gpu_apps':len(apps)}}))
"""
    result = json.loads(ssh(campaign["nodes"][rank]["ip"], "python3 -c " + shlex.quote(code)))
    recipe.save(root / f"node_{rank}" / (mode + "_status.json"), result)
    print(rank, json.dumps(result), flush=True)


def gather(root, campaign, rank):
    host = campaign["nodes"][rank]["ip"]
    log = root / exp.ARM / f"canary_node_{rank}.log"
    if ssh(host, "cat " + shlex.quote(str(log) + ".exit")).strip() != "0":
        raise ValueError(f"Canary node {rank} did not exit successfully")
    work = root / exp.ARM / "canary"
    code = f"""from pathlib import Path
import json
r=Path({str(work)!r})
print(json.dumps({{n:[str(p) for p in r.rglob(n)] for n in ['model_00002.pt','model_00002.onnx','model_00002.pair.json','holosoma_config.yaml','train_rank_*.log']}}))
"""
    paths = json.loads(ssh(host, "python3 -c " + shlex.quote(code)))
    dest = root / exp.ARM / "canary_artifacts"
    dest.mkdir(parents=True, exist_ok=True)
    for name, matches in paths.items():
        expected = 8 if name == "train_rank_*.log" else (1 if rank == 0 else None)
        if expected is not None and len(matches) != expected:
            raise ValueError(f"Artifact multiplicity {rank}:{name}={matches}")
        if rank and name != "train_rank_*.log":
            continue
        for path in matches:
            recipe.run(["scp", "-q", f"ubuntu@{host}:{path}", dest / Path(path).name])
    for name in ("initializer_preflight.json", "git_verification.json"):
        recipe.run(["scp", "-q", f"ubuntu@{host}:{work}/node_{rank}/{name}", dest / (f"node_{rank}_" + name)])
        if rank == 0:
            recipe.run(["scp", "-q", f"ubuntu@{host}:{work}/node_{rank}/{name}", dest / name])
    recipe.run(["scp", "-q", f"ubuntu@{host}:{log}", dest / ("controller.log" if rank == 0 else f"controller_node_{rank}.log")])
    print("Gathered", rank, flush=True)


def accept(root, campaign):
    from contact_sampling_factorial_accept import accept as accept_base
    import torch
    torch.set_num_threads(2)
    artifacts = root / exp.ARM / "canary_artifacts"
    for rank in range(8):
        proof = json.loads((artifacts / f"node_{rank}_git_verification.json").read_text())
        initializer = json.loads((artifacts / f"node_{rank}_initializer_preflight.json").read_text())
        if not proof["accepted"] or proof["commit_sha"] != campaign["source"]["commit"] or not initializer["accepted"]:
            raise ValueError(f"Invalid node proof {rank}")
    accept_base(root, exp.ARM, world_size=64, bank=campaign["dataset"]["bank"], append_duration_s=.2)


def contract(root, campaign):
    import wandb
    path = root / exp.ARM / "run_contract.json"
    if path.exists():
        raise ValueError("Refusing to replace formal identity")
    acceptance_path = root / exp.ARM / "canary_acceptance.json"
    acceptance = json.loads(acceptance_path.read_text())
    if not acceptance["accepted"] or acceptance["ranks"] != 64 or acceptance["commit"] != campaign["source"]["commit"]:
        raise ValueError("Unaccepted canary")
    run_id = wandb.util.generate_id()
    verifications = [json.loads((root / exp.ARM / "canary_artifacts" / f"node_{r}_git_verification.json").read_text()) for r in range(8)]
    result = {"run_id": run_id, "name": exp.NAME, "fresh": True, "resume": None,
        "source": campaign["source"], "node_verifications": verifications,
        "campaign_sha256": recipe.sha(root / "campaign.json"), "canary_acceptance_sha256": recipe.sha(acceptance_path),
        "dataset": campaign["dataset"], "training_export_onnx": True, "save_interval": 500, "target_updates": 40000,
        "world_size": 64, "envs_per_gpu": 2048, "canary_weights_loaded": False, "formal_launch_video_required": False,
        "cli": exp.training_args(campaign, "formal", run_id)}
    recipe.save(path, result)
    path.chmod(0o444)
    for node in campaign["nodes"]:
        recipe.run(["scp", "-q", path, acceptance_path, f"ubuntu@{node['ip']}:{root}/{exp.ARM}/"])
    print("Reserved fresh identity", run_id, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "install", "sync", "preflight", "canary", "status_canary", "gather", "accept", "contract", "formal-preflight", "formal", "status_formal"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ranks", nargs="+", type=int, choices=range(8))
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.root)
        return
    campaign = json.loads((args.root / "campaign.json").read_text())
    if args.stage in {"accept", "contract"}:
        globals()[args.stage](args.root, campaign)
        return
    for rank in args.ranks or range(8):
        if args.stage.startswith("status_"):
            status(args.root, campaign, rank, args.stage.removeprefix("status_"))
        elif args.stage in {"canary", "formal"}:
            launch(args.root, campaign, rank, args.stage)
        elif args.stage in {"preflight", "formal-preflight"}:
            preflight(args.root, campaign, rank, args.stage)
        else:
            globals()[args.stage](args.root, campaign, rank)


if __name__ == "__main__":
    main()
