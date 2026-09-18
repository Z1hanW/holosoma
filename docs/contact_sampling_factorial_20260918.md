# Contact and within-clip sampling factorial

Eight independent one-node/eight-L40S jobs, 2048 environments per GPU:
`rl-nocontact+uniform`, `rl-contact+uniform`, `rl-nocontact+adaptive`,
`rl-contact+adaptive`, and the corresponding four `mix-` jobs.

The shared baseline uses the ch2ckwzw/13K-generated 137-motion bank,
box d9m3z369/23K actor initialization, a strict [512,256,128] actor and
trainable native depth CNN, legacy torso camera mount plus 10 degrees,
precomputed turn-then-forward commands and restored SW peak-height buttons.
Drop zeros the root command. Existing jobs are not changed.

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
motion producer remains ch2/13K and is separately identified from the online
ch2/40K label teacher. Pure RL never loads or calls the label teacher.

All arms target 40000 updates, saving and uploading atomic PT+ONNX pairs
every 500 updates. No replay/video upload is a launch prerequisite.

Reproduction uses `scripts/contact_sampling_factorial_control.py` in order:
prepare, install, sync, preflight, canary, gather; then
`scripts/contact_sampling_factorial_accept.py`, followed by contract and
formal. Each node directly fetches the same pushed Git commit into a clean
checkout. Only explicit manifest-listed non-code assets are synchronized.
CPU initializer/ONNX parity and real eight-rank two-update PT+ONNX canary
acceptance precede formal launch. Canary weights are never formal inputs.

The campaign JSON, per-arm immutable run contracts, Git verification,
initializer proof, checkpoint-pair checks, independent ONNX parity, and
per-rank logs are retained under the caller-specified audit root.
