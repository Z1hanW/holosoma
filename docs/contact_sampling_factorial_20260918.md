# Contact and within-clip sampling factorial

Eight independent one-node/eight-L40S jobs, 2048 environments per GPU:
`rl-nocontact+uniform`, `rl-contact+uniform`, `rl-nocontact+adaptive`,
`rl-contact+adaptive`, and the corresponding four `mix-` jobs.

The corrected shared baseline uses the ch2ckwzw/40K-generated 137-motion bank,
box d9m3z369/23K actor initialization, a strict [512,256,128] actor and
trainable native depth CNN, legacy torso camera mount plus 10 degrees,
precomputed turn-then-forward commands and restored SW peak-height buttons.
Drop zeros the root command. This is a new experiment generation, not a hot
change or exact resume of the initial 13K-data jobs.

The user requested SW's May training robot depth geometry. The factorial
launcher now explicitly selects `pelvis.STL` and
`combined_{left,right}_wrist_rubberhand.STL`, while leaving the other 26 depth
meshes and the entire robot physics asset package unchanged. All 29 selected
mesh files were SHA256-compared with Git `df701cff64bd698e130d9533ceb47a3825d87e14`
and matched. A new actor-init migration permits only these three mappings in
addition to the already declared command/button/native-CNN metadata changes.
It does not permit different camera extrinsics, preprocessing, latency, other
meshes, actor topology, or robot control. No fallback is added. Current camera
pose DR is deliberately unchanged; this is not complete historical observation
distribution equivalence. Existing box profiles retain their old contracts.

Positive contact guidance outer weight is 0 or 1. Its contact and wrist
component weights are both 1. Collision penalties stay unchanged. Missing
annotations stay inactive; no replacement contact window is fabricated.
Across-clip weighting stays uniform, T1 boosting stays disabled. Only the
failure-adaptive within-clip sampler is switched. The common start-zero
probability still ramps from 0 to 1 over iterations 0..39999, so the
nonzero-start sampling branch gradually shrinks in every arm.

Mix uses strict ch2ckwzw/model_40000.pt labels on student-controlled states,
not teacher-controlled rollouts. PPO starts at 0.01, BC at 0.99; the existing
nine-tier interpolation changes every 700 iterations and reaches 0.9/0.1
at iteration 6300 (increments are 0.89/9, not exactly 0.1). The offline
motion producer and online label teacher are now both ch2/40K. Pure RL never
loads or calls the label teacher. Contact points/intervals are from that exact
40K collection, not the old 13K sidecars. The publication verifies all 137
rollout-reference hashes and preserves all ten motion arrays while adding the
precomputed command; all clips remain present, including failed rollouts.

Command bank digest:
`7a7433fe27a16f8bdb59276e62895692b4297e793466d1b2f871dd6c238d37dc`.
Contact bank digest:
`d30629f57051d22c1dd0ead6578992743e098d1e1534c0a6808e9c4328ddc5e1`.
The 49183 command frames contain 16745 zero, 27230 forward and 5208 yaw phases;
dy is zero and forward/yaw never overlap. Runtime pickup latch and drop-zero
gating remain; commands are not recomputed from live robot tracking error.

All arms target 40000 updates, saving and uploading atomic PT+ONNX pairs
every 500 updates. No replay/video upload is a launch prerequisite.

Reproduction uses `scripts/contact_sampling_factorial_control.py` in order:
prepare, install, sync, preflight, canary, gather; then
`scripts/contact_sampling_factorial_accept.py`, followed by contract and
formal. Each node directly fetches the same pushed Git commit into a clean
checkout. Only explicit manifest-listed non-code assets are synchronized.
CPU initializer/ONNX parity and real eight-rank two-update PT+ONNX canary
acceptance precede formal launch. Canary weights are never formal inputs.
Each job uses the existing single-node CPU/Gloo gradient reduction path;
multi-node hierarchical reduction is explicitly disabled. Before creating
formal tmux, the worker repeats initializer/ONNX parity with the exact full
formal CLI and immutable run identity, without starting W&B or allocating
training GPUs.

The campaign JSON, per-arm immutable run contracts, Git verification,
initializer proof, checkpoint-pair checks, independent ONNX parity, and
per-rank logs are retained under the caller-specified audit root.

## September 18 replacement generation

Execution commit: `76df1abfbb6404f06aeb2321147b450c4ea00d4c`.
Execution tree: `3d7ac77a34048ab460c786084a441c5f50f463d6`.
All eight nodes fetched this commit directly from `origin/main` and passed
clean-checkout/submodule verification. Subsequent documentation commits do
not change the running source.

Audit root:
`/data/holosoma_training_audits/contact_sampling_latest40k_swdepth_ws8x8_20260918`.

| Arm | Node | New Run | Superseded Run |
| --- | --- | --- | --- |
| rl-nocontact+uniform | zzzihanw-26 | [aulhhzxm](https://wandb.ai/zihanw22/carry-any/runs/aulhhzxm) | wh9cbn82 |
| rl-nocontact+adaptive | zzzihanw-27 | [fgwvovxy](https://wandb.ai/zihanw22/carry-any/runs/fgwvovxy) | v21wabn6 |
| rl-contact+uniform | zzzihanw-34 | [5zmbxh3t](https://wandb.ai/zihanw22/carry-any/runs/5zmbxh3t) | oky97w4s |
| rl-contact+adaptive | zzzihanw-35 | [legiadml](https://wandb.ai/zihanw22/carry-any/runs/legiadml) | jdzbcq4p |
| mix-nocontact+uniform | zzzihanw-39 | [xfn7ydac](https://wandb.ai/zihanw22/carry-any/runs/xfn7ydac) | e17vgbqb |
| mix-nocontact+adaptive | zzzihanw-45 | [fw3009av](https://wandb.ai/zihanw22/carry-any/runs/fw3009av) | e078w1cx |
| mix-contact+uniform | zzzihanw-72 | [4kocpixa](https://wandb.ai/zihanw22/carry-any/runs/4kocpixa) | kj4h1y1c |
| mix-contact+adaptive | zzzihanw-f | [4u4fo0dq](https://wandb.ai/zihanw22/carry-any/runs/4u4fo0dq) | km2wgymr |

The eight old torchrun parents were individually authenticated against their
original CLI, Git checkout, W&B identity and eight GPU workers before SIGINT.
Their files were preserved. Their W&B `failed` states reflect the intentional
exit-code-130 replacement, not a numerical training failure; lifecycle summary
and tags identify the reason. No unrelated training was stopped.

Focused regression validation: 125 tests passed. All eight two-update canaries
exited zero with all-rank state and valid same-iteration PT+ONNX pairs. Each
passed full ONNX checking, ORT loading and an independent 14-row actor probe;
maximum absolute differences ranged from 3.81e-6 to 6.44e-6, within the fixed
`rtol=1e-3, atol=2e-6` elementwise acceptance rule. These are startup/export
checks, not evidence of task success or improved learning.

`depth_and_height_audit.json` independently binds all 29 selected files to the
historical Git blobs. The full 137-clip bank includes `prism_cf_bin_m1_v10`,
whose smoothed world-z range is only 0.000536227 m and whose maximum rise above
its initial height is zero. This anomaly was reported; no clip was filtered
and no replacement lift/contact label was fabricated.

The new generation is fresh actor-only initialization from box/23K, not resume
of the superseded jobs. Critic, optimizers, iteration and run identity are
fresh. Formal startup acceptance is recorded separately in
`formal_start_acceptance.json`; a launch request alone is not acceptance.

All eight formal jobs passed independent live acceptance at
2026-09-18 21:02:48 UTC. In table order, completed updates were
114 / 81 / 89 / 63 / 69 / 52 / 47 / 29. Fresh W&B state was `running` for all
eight, with finite loss/reward, correct PPO/BC weights, eight distinct GPU
workers per job, zero volatile uncorrected ECC, clean exact-commit source,
and no fatal rank-log entries. All non-factor scientific settings matched
across arms; each old/new comparison permitted only the declared data/contact
paths, three depth meshes and explicit initializer migration identity.

Acceptance SHA256:
`8b2bb42932b6c205e9fe1e9d02bb1ef1b9466a687126b3aa4c04d144ef2c9707`.
The controller-only audit was corrected to respect the existing pure-RL
provenance schema: offline producer identity is bound through the immutable
campaign and data manifests, not the distillation-only top-level generator
field. Worker environment, producer SHA and both bank manifests were also
verified. No live code/configuration was changed for this correction.
The first formal 500-update upload boundary had not yet been reached; this
acceptance is not a claim of its remote upload or of learning quality.
