# Eight contact/sampling experiments, 2026-09-18

Status: all eight formal jobs passed live-startup acceptance at
2026-09-18 08:13:11 UTC and remain running.

| Experiment | Node | Private IP | W&B identity |
| --- | --- | --- | --- |
| rl-nocontact+uniform | zzzihanw-26 | 10.99.0.18 | wh9cbn82 |
| rl-nocontact+adaptive | zzzihanw-27 | 10.99.0.227 | v21wabn6 |
| rl-contact+uniform | zzzihanw-34 | 10.99.0.167 | oky97w4s |
| rl-contact+adaptive | zzzihanw-35 | 10.99.0.77 | jdzbcq4p |
| mix-nocontact+uniform | zzzihanw-39 | 10.99.0.39 | e17vgbqb |
| mix-nocontact+adaptive | zzzihanw-45 | 10.99.0.183 | e078w1cx |
| mix-contact+uniform | zzzihanw-72 | 10.99.1.89 | kj4h1y1c |
| mix-contact+adaptive | zzzihanw-f | 10.99.1.122 | km2wgymr |

Each identity is `https://wandb.ai/zihanw22/carry-any/runs/<id>`.
These are eight independent one-node jobs, not one distributed 64-rank job.
Every job has 8 L40S GPUs and 2048 environments per GPU (16384 per job).
The last two nodes are in ap-northeast-2b; the other six are in 2a. There is
no inter-node gradient communication. Existing unrelated training was not
stopped or updated. Node -47 acquired another task during preparation and
was excluded without touching that task. Unprepared -65/-93 were not used.

## Common scientific settings

- Exact actor initialization: recovered d9m3z369/model_23000.pt, SHA256
  `e9de2954556f7f39c98cc5e90de2e28550dad4ba656c986280918c929af1256d`.
  Actor/CNN/std loaded strictly; no layer resizing, no critic/optimizer resume,
  no canary weights loaded into formal training.
- Actor [512,256,128], native trainable depth CNN, 32-dimensional embedding;
  94 scalar inputs plus 58x87 processed depth. Critic [512,256,128].
- Common offline data: ch2ckwzw/model_13000.pt-generated rollout137, with
  precomputed turn-then-forward commands. Command bank is
  `/data/holosoma_inputs/ch2ckwzw_model13000_rollout137_precomputed_turn_forward_v1/by-source/95e5a54bb1e429874af7a93bfb6bb902b5fc9bd7564eab481d3bce5cd01c0303`.
- Online mix teacher: latest completed ch2ckwzw/model_40000.pt, SHA256
  `14e644323b8e6a7b769dbf641d9f625bee895742b7064368b55afd0710a0d665`.
  Teacher has privileged 178-dimensional actor_obs; student input is unchanged.
  Student always controls the environment. Pure RL disables teacher inference.
- Legacy torso depth: xyz [0.01,0.01,0.44], mount quaternion
  [0.00644801,0.23350163,0.00644801,0.97231365], additional pitch 10 degrees
  (approximately 27+10 degrees). Raw 106x60, processed 58x87, latency [3,4],
  hole reference batch 4096, legacy_fixed_v1 hole seed and legacy_full_v1 reset.
- Restored SW peak-height pickup/drop buttons; drop zeros all root command
  coordinates. Real visual mesh and existing convex-decomposition collisions.
- Actor/critic initial LR 1e-3, adaptive actor KL schedule; 24 rollout steps,
  7 epochs, 4 minibatches. Other reward, termination and DR settings are shared.
- 40000 fresh updates; atomic uploaded PT+ONNX pairs every 500 updates.
  No replay/video creation or upload is a launch gate.

## Factors

- Contact: fused offline_contact_guidance outer weight 1, contact_weight 1,
  wrist_weight 1. No-contact changes only that outer weight to 0. Collision
  penalties and disabled legacy duplicate positive contact terms are unchanged.
  All nine regions have point/count/interval files for all 137 clips. Valid
  absence of annotations stays inactive; no synthetic contact window is added.
- Adaptive toggles the failure-based within-clip timestep sampler only.
  Across-clip objective remains uniform_clip with rank-local loss corrections.
  Uniform T1 boosting is disabled throughout.
- The shared start-zero probability ramps 0 to 1 over iterations 0..39999;
  hence the nonzero-start sampling branch shrinks in both sampling variants.
- Mix PPO/BC is 0.01/0.99 initially, with the existing nine-step interpolation
  every 700 iterations, reaching 0.9/0.1 at 6300. Each increment is 0.89/9,
  not exactly 0.1. DAgger replay is disabled, matching std is enabled, fixed-BC
  diagnostics run every 100 iterations, and the stopping guard is disabled.

## Source and acceptance

All eight nodes independently fetched origin/main and executed clean detached
commit `15c5d6af3e70250697b93e011942811b0dbaf23d`, tree
`b1f74abb4e30c2aaa121a7d243216f1bbfd20aa0`, from
`https://github.com/Z1hanW/holosoma`. Checkout:
`/data/holosoma_git/contact_sampling_15c5d6af3e70`.
Declared submodules were verified at their recorded gitlinks. No working-tree
archive or source rsync was used. Non-code assets were checked against 11349
file records (21034239877 bytes).

The exact-once 137-clip ws8 shard has counts [32,32,16,16,16,16,8,1], each
dividing 2048, with manifest source digest
`70019668441a24164f64b35ec57a7ed0efdc8d14fc59e148bc1222bff2954c1c`.
Secondary object sharding is disabled. Local unequal clip counts are corrected
in distributed losses; this is not a claim of identical raw per-clip exposure.

All eight final-source canaries completed two real updates and exited normally.
Acceptance verified finite checkpoint state, all eight rank RNG/env states,
effective configuration, strict initializer equivalence, source/submodule
cleanliness, matching PT+ONNX boundaries and hashes, full ONNX checker, CPU
ORT, and an independent 14-row actor parity test per arm. Before each formal
tmux, the exact 40000-update CLI and reserved identity also passed a new CPU
initializer/export/ORT parity preflight. This is startup acceptance, not a
claim that any factor improves pickup, gait, or sim2real.

Audit root (controller and each assigned node):
`/data/holosoma_training_audits/contact_sampling_factorial_ws8x8_20260918_v2/`.
It contains campaign.json, per-arm run_contract.json, canary_artifacts,
canary_acceptance.json, formal_preflight_result.json and formal logs/status.

## Live-startup acceptance

| Experiment | Completed updates | W&B history step |
| --- | ---: | ---: |
| rl-nocontact+uniform | 56 | 53 |
| rl-nocontact+adaptive | 35 | 33 |
| rl-contact+uniform | 52 | 50 |
| rl-contact+adaptive | 38 | 36 |
| mix-nocontact+uniform | 56 | 53 |
| mix-nocontact+adaptive | 42 | 39 |
| mix-contact+uniform | 54 | 51 |
| mix-contact+adaptive | 37 | 34 |

These are sequential snapshots, not throughput comparisons. Fresh W&B queries
confirmed all eight names, running status, 8x2048 environments, source/init/data
identities, contact and sampling factors, and finite loss/reward histories.
Live PPO/BC weights were 1/0 for RL and 0.01/0.99 for mix. Distributed loss
weights summed to 8. All 64 workers had unique per-node GPU bindings, the exact
Git checkout as cwd, correct rank identities, and zero GPU uncorrectable ECC.
All 64 rank logs entered fresh iteration 0 with the strict box23K initializer;
the controller/rank fatal-error scan was empty. Source and submodules remained
clean. The exact formal CPU initializer/export/ORT proofs were re-read and
confirmed to use the 40000-update config rather than the two-update canary.

After removing the declared factors, distill-only settings and experiment
names, training/algo/command/reward/observation/perception/randomization/
termination/simulator/robot/action/curriculum configs matched across all arms.
Their common configuration digest is
`5d296c6b071497bbe0f0ae74bb2ccfcd9a8781d4e486ce5e1b14277f06f4c48b`.
Full evidence is in `formal_start_acceptance.json` and each arm's file of the
same name under the audit root; `accept_formal_start.py` is a read-only remote
health/W&B validator that writes evidence only on the controller.

No formal run has reached its first 500-update checkpoint at this acceptance
time. The accepted canary pairs and formal initialization exports do not imply
that a formal periodic checkpoint has already been uploaded, or that these
settings improve task performance. Runs must remain at the source commit
above; later documentation commits do not change their execution identity.

## Preparation findings

The first CPU preflight rejected a two-update canary declaring a 39999-iteration
curriculum. The bounded canary now uses end=1 and end_probability=1/39999,
preserving the formal probabilities at both executed iterations exactly.

The first real canary rejected the mistakenly enabled multi-node hierarchical
gradient reducer at world_size=local_world_size=8. The final source uses the
existing single-node CPU/Gloo gradient path, with hierarchical reduction and
CPU-leader mode explicitly disabled. The original failed canaries remain under
the sibling audit root without `_v2`; none created a remote formal W&B run.
No PPO equation, simulator stepping, mesh geometry, or live unrelated job was
changed to resolve these launcher errors.

Final-source focused preflight/button/launcher regression had 283 passes,
including 19 factorial launcher tests. A broader test run retained
the previously documented legacy active-contact-sidecar digest test failure;
its validation was not weakened. The peak-height path used here completed the
real eight-arm export checks above.

Known data limitation retained equally in all arms: prism_cf_bin_m1_v10 has
only about 0.000887 m object-z variation, so the restored peak-height rule can
label drop near the beginning. No clip was silently filtered or relabelled.
Isaac/PhysX also reports thin-hull cooking adjustments and CPU collision-cooking
warnings for some existing decomposition pieces. No replacement box geometry
or new fallback was introduced; these warnings remain in the raw audit logs.
