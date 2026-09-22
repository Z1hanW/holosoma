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
