1. Start MuJoCo Environment


python src/holosoma/holosoma/run_sim.py robot:g1-29dof

The robot will spawn in the simulator, hanging from a gantry.
2. Launch the Policy

In another terminal, run the policy inference:

source scripts/source_inference_setup.sh
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference: \
    --task.model-path ***.onnx \                                
    --task.no-use-joystick \
    --task.use-sim-time \
    --task.rl-rate 50 \
    --task.interface lo