# CORL79 + Rollout137 Mixed Training

Requested replacement of the four September 18 pure-RL factorial runs only.
The four existing mixed runs and unrelated training processes remain untouched.

## Recipe

- Exactly 79 legacy CORL clips plus all 137 ch2ckwzw/model_40000 rollouts: 216
  clips, with 60 box, 38 ball, 69 barrel and 49 bin. No old debug30 duplication.
- Preserve original numeric trajectories, real visual/collision meshes, inertial
  properties and available contact sidecars. No geometry or contact fallback.
  The legacy79 interval NPY files are explicitly converted to the runtime JSON
  schema in a separate immutable contact export; every integer pair and every
  original file is verified unchanged. Missing labels are not invented.
- Use the existing precomputed turn-then-forward algorithm, with pickup latch,
  exclusive x/yaw and drop clearing root commands. Peak-height button labels
  retain SW alpha=0.91, five-frame smoothing and five-frame debounce.
- Match `mix-contact+adaptive`: online ch2/40K teacher, student-controlled DAgger,
  PPO coefficient .01 to .9 in 700-update tiers; contact weights 1/1/1;
  adaptive within-clip sampling and uniform across clips, no T1 boost.
- Same strict box d9m3z369-recovered/23K actor initialization, trainable native
  depth CNN, actor and critic [512,256,128], SW robot-depth meshes and 27+10deg
  camera. Fresh critic, optimizers, iteration and run identity; not exact resume.
- Add `runtime_default_pose_append_duration_s=0.2`: ten 50Hz targets return from
  the terminal frame to standing at the terminal root XY/yaw. Original source
  frames and labels remain unchanged; the appended root command is zero.
- Eight nodes, eight L40S each, 2048 environments/GPU. Rank-local clip counts
  divide 2048; the shard union covers every clip once, with existing rank loss
  weighting. Increasing the global rollout batch eightfold is an intentional
  topology change, not claimed to be optimizer-equivalent to the single node.
- 40,000 updates; native atomic PT+ONNX every 500 updates, uploads enabled.

The nearly motionless `prism_cf_bin_m1_v10` remains in the requested complete
bank and is reported in its identity; it is not silently removed or relabeled.

## Reproduction

Data builder: `scripts/prepare_corl79_rollout137_mix_bank.py`.
Contact format converter: `scripts/prepare_mix216_contact_labels.py`.
Git-bound worker: `scripts/mix_contact_adaptive_ws64.py`.
Controller: `scripts/mix_contact_adaptive_ws64_control.py`.

All nodes fetch the same pushed commit into clean independent Git checkouts.
Only manifest-bound data/geometry/checkpoints are transferred outside Git.
Each node verifies strict initialization and real CPU ONNX parity before the
64-rank two-update canary; formal launch additionally requires the complete
64-rank PT+ONNX pair and per-node source evidence. Canary weights are not used.
No replay recording or video upload is part of the launch path.

Audit root:
`/data/holosoma_training_audits/mix_contact_adaptive_corl79_rollout137_append02_ws64_20260922`.

## Formal Launch

- W&B: <https://wandb.ai/zihanw22/carry-any/runs/zksf8buh>.
- Immutable execution commit: `d9c63682732fd77e1565118ea9eb56682d909a80`.
- Git tree: `e23c654f9d482636f0fa3185b67d0668a3455702`.
- Active evidence is under `attempt2_git_reachability/` in the audit root;
  its `mix-contact+adaptive/run_contract.json` is the formal contract.
  The parent-directory reserved identity `7sakymlv` was never launched.
- All eight nodes are in `ap-northeast-2a`, in rank order:
  `10.99.0.18`, `.227`, `.167`, `.77`, `.116`, `.117`, `.165`, `.176`.
  Transport is eight local NCCL groups with eight Gloo CPU leaders.
- Stopped only the four pure-RL factorial runs: `aulhhzxm`, `fgwvovxy`,
  `5zmbxh3t`, `legiadml`. Their durable checkpoint pairs were retained.
  No stop/restart was sent to the four mixed runs or unrelated experiments.
- Fresh 64-rank canary completed two actual updates; all rank RNG/env states
  and tensors passed validation. Native PT/ONNX pairing passed; independent
  14-row PyTorch/ORT parity had maximum absolute error `3.814697265625e-6`.
  All 64 rank logs confirm ten appended control steps. No canary weights or
  optimizer state are inherited by the formal run.
- At `2026-09-22 10:34 UTC`, read-only live acceptance verified all 64 worker
  PIDs/ranks, exact checkout/environment bindings and at least 11 completed
  updates on every node. No NCCL timeout, OOM, non-finite loss or uncorrectable
  ECC was observed. Fresh W&B reported `running`, PPO `0.01`, BC `0.99`, and
  distributed loss weights summing to 64. Full evidence is in
  `mix-contact+adaptive/formal_start_acceptance.json` in the active attempt.
  The first formal 500-update upload boundary had not yet been reached;
  native pair export/parity was verified by the full-rank canary, not claimed
  as a completed formal checkpoint upload.

The first canary passed, but its formal preflight stopped before launch when
a provenance consumer incorrectly equated a pinned commit with the moving
remote branch tip. The fix revalidates live HEAD/tree and Git ancestry; it
does not weaken clean-checkout or remote-source requirements. Regression
tests cover advancing tips, unrelated histories, moved HEAD and invalid
proofs. A new exact-source checkout, new 64-rank canary and fresh identity
were used after the fix. The original failure remains in `attempt1_outcome.json`.

## Published Data

Data-only NFS directory:
`/nfs/zzzihanw/ds_as_data/_distill/corl79_ch2_40k_rollout137_mix216_append02`.

- Archive basename:
  `ad8b834a6f0dce7e1426417bf8796d14941143d9187016e42e04321a960cecbf.tar.gz`.
- Archive SHA256:
  `b461df2fb2d70cdb734ef72edd6a04b441339a2da74b736be4305135dfb298c0`.
- Adjacent same-basename `.manifest.json` binds all 14,223 data files.
- The published archive was fully reread and hashed through NFS. It contains
  the 216-clip precomputed bank, geometry/shards and normalized contact export,
  not executable source. Append is applied at runtime, not baked into NPZs.
- All eight nodes received and independently hashed the complete required
  assets. The exact installed paths/digests are in `dataset_identity.json`
  and `campaign.json` in the active audit directory.

## Additional 2a Launch After Preparing Spare Nodes

At the user's request, prepare zzzihanw-65 before stopping `xfn7ydac` and
combine the released GPUs with the seven spare nodes. This is a new fresh
box23K-initialized run with the recipe above, not another model_01000 resume.
The separately running recovery `3e07dm2e` was not stopped or modified.

- New W&B: <https://wandb.ai/zihanw22/carry-any/runs/5sk26e4t>.
- Execution remains pinned to `d9c63682732fd77e1565118ea9eb56682d909a80`;
  every node independently fetched and verified the clean remote Git checkout.
  No training algorithm, camera, reward, command or data definition changed.
- Rank order: `zzzihanw-17/.97`, `-47/.24`, `-49/.180`, `-46/.54`,
  `-40/.244`, `-93/.61`, `-65/.201`, `-39/.39`, all `10.99.0.*` in
  `ap-northeast-2a`. Master is `10.99.0.97:36960`.
- Each GPU has 2048 environments: 131072 total. All 56 directed network
  probes had zero packet loss, with mean RTT 0.210-0.910 ms.
- Prepared one verified-unused NVMe on -65 for `/data`. Storage inspection
  also found -40 and -93 using their system disks for training paths. Their
  inactive training directories were checksum-copied onto one unused NVMe
  each; originals remain as `*.rootfs-preserved-20260922`. Existing Ray and
  other service paths were not changed. All new mounts have UUID-bound
  persistent entries; unavailable mounts leave unwritable directories.
  Instance-store contents remain ephemeral across instance stop/termination.
- Only `xfn7ydac` was stopped, after -65 passed asset/runtime/ONNX preflight.
  Verified parent/worker PID identities before signaling; all eight GPU
  workers exited. Last observed progress was 32035; durable model_32000 PT
  and ONNX hashes matched the remote pair manifest. W&B was closed with an
  explicit user-stop reason, no training-history append, and unchanged
  scientific config. Its `failed` status means user interruption, not a
  numerical failure or completed training target.
- The new 64-rank canary completed two updates and exited zero on all nodes.
  Complete rank RNG/environment state and finite tensors were verified;
  independent PyTorch/ORT maximum absolute error was `2.86102294921875e-6`.
  Formal training loads no canary weights.
- At `2026-09-22 23:01:40 UTC`, formal acceptance confirmed all eight nodes
  at 11 completed updates, 64 distinct GPU workers with ranks 0-63, no fatal
  log matches or uncorrectable ECC, and finite W&B losses. PPO/BC was
  .01/.99 and distributed loss weights summed to 64. Native saves remain
  PT+ONNX every 500 updates; the first formal upload boundary was not yet
  reached at this startup acceptance.

Audit root:
`/data/holosoma_training_audits/mix216_second_ws64_20260922_2220`.
See `REPORT.md`, `formal_start_acceptance.json`, `xfn_stop.json`,
`xfn_wandb_stop_verification.json`, `storage*_ready.json`, `network_probe.json`,
and `mix-contact+adaptive/run_contract.json`. Controller-local admin helpers
are not distributed as training source. The pre-existing `agent.md` edits
were not staged or modified.
