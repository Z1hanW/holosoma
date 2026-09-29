# PRISM runtime

This branch provides the HoloSoma training and inference packages used by PRISM:
privileged teacher PPO, depth-policy distillation, teacher trajectory/contact
export, and matching ONNX command and observation handling.

Install both packages into the same Python 3.11 environment:

```bash
pip install "holosoma @ git+https://github.com/Z1hanW/holosoma.git@dev-prism-release#subdirectory=src/holosoma"
pip install "holosoma-inference @ git+https://github.com/Z1hanW/holosoma.git@dev-prism-release#subdirectory=src/holosoma_inference"
```

The PRISM repository supplies `train_teacher.sh`, `rollout.sh`,
`train_student.sh`, experiment recipes and checkpoints. Install PyTorch 2.7.0,
Isaac Sim 5.1.0 and IsaacLab 0.47.2 separately in the training environment.
Motion banks, contacts and object assets are supplied separately.

For reproducible training, install both packages at the same full Git commit
instead of a moving branch name. Every node must fetch and verify the matching
clean HoloSoma and PRISM checkouts. PRISM verifies the installed packages against
that Git source before training. No local `src/` path override is needed.
