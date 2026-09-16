# Box 23K Pure PPO Preparation

Status: both formal runs launched and independently accepted on 2026-09-16 UTC.

- 13K rollout137: https://wandb.ai/zihanw22/carry-any/runs/rp2leis2
- 40K rollout137: https://wandb.ai/zihanw22/carry-any/runs/izz2f2df
- Each job: 4 nodes x 8 L40S, 2048 environments/GPU, 65536 global environments.
- Execution commit: `886b2afc5e9366a6e4a61dbb44a87f9ee5603385` from origin/main,
  clean detached Git checkout on every node. Later documentation commits are
  not hot-installed into the live runs.

Requested initially: four GPUs, actor initialization from
`wandb://zihanw22/boxer/d9m3z369-recovered/model_23000.pt`, historical
27-degree mount plus 10-degree pitch, and rollout137 data.

Latest clarification: run BOTH ch2ckwzw/model_13000 (EQ/52C data) and
ch2ckwzw/model_40000 (W33 data), each on four eight-GPU nodes, 2048 envs/GPU.
Do not filter either 137-clip bank. The producer is not an online teacher.

Depth reference: https://chatgpt.com/share/6aaa2f1e-b7ac-83e8-a609-2cbb8b700388
Restore the historical TRAINING depth, not the legacy RealSense deployment
pipeline: raw 60x106, crop 2/0/4/4, bicubic 58x87, range 0.3..3 m,
normalize, synthetic holes filled with far, 30 Hz configured updates,
random latency 3..4 / buffer 6. Preserve historical edge/hole/additive noise,
legacy hole seed and refresh semantics, and the initializer's 4096-env hole
normalization reference while running 2048 actual environments per GPU.

## Verified Initializer

- SHA256: `e9de2954556f7f39c98cc5e90de2e28550dad4ba656c986280918c929af1256d`.
- Cache: `/data/holosoma_checkpoint_cache/zihanw22_boxer_d9m3z369-recovered/by-sha256/e9de2954556f7f39c98cc5e90de2e28550dad4ba656c986280918c929af1256d.pt`.
- Actor: `[512,256,128]`, small depth CNN, 94 scalar inputs and 5046 depth inputs,
  29 actions. Strict full actor-state loading succeeded without shape changes.
- Camera: torso-relative offset `[0.01,0.01,0.44]`, mount xyzw quaternion
  `[0.00644801,0.23350163,0.00644801,0.97231365]`, additional pitch 10 degrees.
- CPU neural ONNX checker, ORT loading, and PT/ORT parity passed;
  maximum absolute error `2.5033950805664062e-6`.
- Audit: `/data/holosoma_training_audits/box23k_pure_rl_ws4_20260916/initializer_preflight.json`.
  This is neural-only preparation, not full deployment or formal-launch acceptance.

## Code Change

Legacy feedforward actor configs omit the subsequently introduced
`lstm_hidden_dim=256` and `lstm_num_layers=1`. Canonical policy-init comparison
now materializes those exact inert defaults for non-LSTM actors. Non-default
values still fail; camera, observation, command and control validation is unchanged.
The existing HMI rejection test now asserts the offending field because the
failure is detected by the common contract comparison.

Validation: policy-init, worker and command/deployment suites, 124 passed. The real 23K actor
contract also survives current config deserialization without semantic drift.

## Dataset And Initialization Boundaries

The two banks have distinct trajectories. The 13K sidecar bank contains 137 clips,
of which 130 have valid arm-contact intervals; this does not authorize filtering
seven clips or introducing contact-window fallback. Both jobs retain all 137.

The opt-in box profile uses exact actor-only initialization, retaining all CNN,
MLP and std tensors. An explicit migration changes tracking-error commands to
precomputed turn/forward, uses kinematic pickup cues for every clip, and makes
drop exclusive. This is deliberate input-semantic transfer, not full behavioral
equivalence. Native-CNN pretrained/frozen flags were historically inert; their
metadata is corrected without changing tensor values or trainability.

All other worker profiles retain their previous behavior. Current pure-PPO
tracking rewards/DR/sampler remain; positive contact guidance, online teacher,
BC and DAgger are disabled. Critic, optimizer and iteration are fresh. The
initializer is the three-hidden-layer model, not a widened five-layer actor.

Historical merged pelvis/wrist depth meshes are provided as a separate hashed
91-file asset package, without altering current robot URDF physics or accepting
missing-mesh substitution. Source distribution is direct exact-commit remote Git;
assets and checkpoint alone are copied outside the source checkout.

## Acceptance And Run Contract

- Audit/controller: `/data/holosoma_training_audits/box23k_rollout137_ab_ws32_20260916/`.
  Each arm contains an immutable `run_contract.json`, `canary_acceptance.json`,
  `initializer_preflight.json`, actual `formal_config.yaml` and
  `formal_start_acceptance.json`.
- Both 32-rank canaries completed two updates and exited 0 on every node.
  Native PT+ONNX pairs, all rank logs, finite tensors, complete Git identities,
  exact data partition, actual camera metadata and command adapter were checked.
  Independent ONNX parity max absolute errors: 13K `3.5762786865234375e-6`,
  40K `3.814697265625e-6`. No canary weights are loaded into formal training.
- Formal acceptance observed at least 14 updates on every 13K rank and 15 on
  every 40K rank. Fresh W&B API reported both running, finite reward/loss,
  PPO coefficient 1, BC/distill/DAgger 0, distributed loss weight sum 32.
  All nodes had eight GPU workers bound to the exact source and UECC 0.
- Actual saved formal configs exactly match their immutable contracts; W&B's
  complete perception config also matches. W&B's lossy empty-dict/enum logging
  representation is not used as a substitute for the exact saved YAML.
- Each run targets 40000 new updates, saving paired PT+ONNX every 1000.
  Actor and critic initial LR `1e-3`, adaptive KL `0.01`, LR bounds
  `[1e-5,1e-2]`, entropy `0.005`. No full resume, stage4 init or teacher actions.
- 13K nodes: `10.99.0.141`, `.97`, `.54`, `.201`.
  40K nodes: `10.99.0.116`, `.117`, `.165`, `.176`.
  All eight were verified in `ap-northeast-2a`, idle before launch.
- Final configs differ only in data/object-map paths and logging identity/paths.
  Other running jobs and global perception presets were not modified.

Preparation failures were fixed before formal launch: duplicate CLI options,
dictionary/tuple CLI syntax, and finalizing runtime assets before strict
initializer validation. None was bypassed. No video preflight or upload was used.
Startup acceptance is not a claim of task success or improved real-world policy
quality; those require later checkpoint evaluation.
