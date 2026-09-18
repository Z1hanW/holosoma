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
