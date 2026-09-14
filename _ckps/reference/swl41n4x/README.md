# SW policy debugging reference

Known-working policy from W&B run `zihanw22/carry-any/swl41n4x` (`sw`).

- Checkpoint: `model_20000.onnx`
- W&B upload date: 2026-05-28
- SHA-256: `627bc573f89f6454587b20f774ff23c2442cc9e356eb13faf75ea8eb62fb277a`
- ONNX iteration metadata: `20000`
- Inputs: `actor_obs[1,94]`, `perception_obs[1,5046]`
- Output: `action[1,29]`
- Actor MLP: `126 -> 512 -> 256 -> 128 -> 29`
- Parameters: `258621`
- Camera: D435, effective approximately 37 degrees down (27-degree mount plus 10-degree pitch)
- Camera frame: `106x60`, cropped and resized to `87x58`
- Camera range/FOV: `0.3-3.0 m`, `89.5 x 58.6 degrees`, 30 FPS

The ONNX file is intentionally ignored by Git and stored on the robot at:

```text
_ckps/reference/swl41n4x/model_20000.onnx
```

Download the exact file again with the W&B API from run `swl41n4x`, file
`model_20000.onnx`, and verify the SHA-256 above before using it as a reference.
