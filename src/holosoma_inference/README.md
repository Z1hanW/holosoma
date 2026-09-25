# HoloSoma inference package

The package provides ONNX policy execution, observation preprocessing,
command/latch adapters and simulation/robot interfaces. The runtime validates
saved observation, motion, geometry and command contracts before inference.

See the [README](../../README.md) for installation. Import configuration comes from
`holosoma_inference/config`; executable entrypoints are `run_policy.py` and
`holosoma/run_sim.py` in the training package. Select checkpoints and motions
explicitly.

A policy ONNX is deployed with its authenticated metadata and runtime
adapters; an actor-only graph is not a complete policy bundle.
