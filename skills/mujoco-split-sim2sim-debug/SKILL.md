---
name: mujoco-split-sim2sim-debug
description: Use this skill when debugging Holosoma MuJoCo split sim2sim rollouts, especially for G1 whole-body tracking or object-carry policies that do not actuate, drift away from the reference, fail to carry the object, disagree with web or viser views, or reset incorrectly.
---

# MuJoCo Split Sim2Sim Debug

Treat split MuJoCo as the authoritative path.

- Trust `run_sim.py` + `run_policy.py` + `sim-state` logs before trusting web or `viser` views.
- If the web demo and split sim disagree, debug split sim first.
- A running policy process is not evidence that MuJoCo is actuated.

## Primary Entry Points

- `./run_wobj_tracking_validated.sh`
- `./vis_mujoco_sim_state.sh`
- `./sim2sim_box_split_tracking.sh <motion.npz> <model.onnx>`

## Required Invariants

1. Only one rollout may own ports `5655/5657/5659` at a time.
2. `use_zmq_lowcmd=True` must also bring up the split `sim-control` subscriber on the simulator side.
3. Object-tracking inference must use simulator clock and simulator state.
4. Per-joint action scaling must come from ONNX metadata, not a flat fallback.
5. When simulator ref-body state is available, inference should prefer that over FK-estimated ref-body pose.
6. Reset must mean `sim-control -> simulator reset -> clock rewind -> motion clip restarts from frame 0`, not a full process restart.

The current reference implementation lives in:

- `src/holosoma/holosoma/simulator/shared/simulator_bridge.py`
- `src/holosoma_inference/holosoma_inference/config/config_values/inference.py`
- `src/holosoma_inference/holosoma_inference/config/config_types/task.py`
- `src/holosoma_inference/holosoma_inference/policies/base.py`
- `src/holosoma_inference/holosoma_inference/policies/wbt.py`
- `src/holosoma_inference/holosoma_inference/utils/sim_control.py`

## First Checks

When the robot does not move or only jitters:

1. Check that the simulator received lowcmd.
   Look in `logs/sim2sim_runs/.../mujoco.log` for `Received first ZMQ lowcmd` or reset/control-channel activity.
   Relevant code: `src/holosoma/holosoma/simulator/shared/simulator_bridge.py`
2. Check that inference is using the correct config.
   For current object tracking, start from `inference:g1-29dof-w-obj`.
   Relevant code: `src/holosoma_inference/holosoma_inference/config/config_values/inference.py`
3. Check that inference is aligned with split sim semantics.
   Required task flags are `use_sim_time`, `use_sim_state`, `prefer_sim_ref_from_sim_state`, and `restart_motion_on_clock_reset`.
   Relevant code: `src/holosoma_inference/holosoma_inference/config/config_types/task.py`
4. Check that action scales were restored from ONNX metadata.
   Relevant code: `src/holosoma_inference/holosoma_inference/policies/base.py`
5. Check that ref-body pose comes from split sim-state when available.
   Relevant code: `src/holosoma_inference/holosoma_inference/policies/wbt.py`

## Expected Evidence

When the split stack is healthy, expect to see all of these:

1. `mujoco.log` shows first active lowcmd reception on the simulator side.
2. `policy.log` shows motion timesteps advancing after policy start.
3. For resets, `mujoco.log` shows reset queued from `sim-control`, then simulator reset, then `policy.log` shows clock rewind handling and clip restart from frame 0.
4. `viser` or web viewers show the same trend as split sim, not the other way around.

If any one of those is missing, debug that missing link first.

## Object-Carry Specific Checks

When G1 moves but the object does not move with it:

1. Enable split sim trace with `HOLOSOMA_SPLIT_SIM_STATE_TRACE_PATH`.
2. Inspect `robot_root_state`, `actors.object`, `object_robot_contact_count`, and `object_robot_contact_bodies`.
3. If hand contacts disappear and knee/foot contacts dominate later, the issue is MuJoCo contact/material setup, not inference shape mismatch.
4. Tune object contact parameters from the simulator path, not from the web viewer.
5. The first knobs to check are `MUJOCO_OBJECT_GEOM_FRICTION`, `MUJOCO_OBJECT_TERRAIN_PAIR_FRICTION`, `MUJOCO_LIMIT_OBJECT_CONTACTS_TO_CARRY_BODIES`, and `MUJOCO_OBJECT_CONTACT_BODY_MARKERS`.

Relevant files:

- `sim2sim_box_split_tracking.sh`
- `src/holosoma/holosoma/simulator/shared/simulator_bridge.py`
- `src/holosoma/holosoma/simulator/mujoco/scene_manager.py`

## Reset Semantics

Correct reset behavior:

1. Send `{"action": "reset"}` over split `sim-control`.
2. Simulator sets `_pending_reset`, restores motion-init state, and rewinds sim clock.
3. Policy sees clock move backwards and restarts the motion clip from frame 0.
4. Viewer should measure reset completion only after `sim_time_ms` rewinds, not after the first cached state message.
5. Expected latency is tens of milliseconds, not seconds.

Measured reference sequence for the fixed path:

- `mujoco.log`: reset queued at `2026-03-19 01:06:51.909`
- `mujoco.log`: simulator reset at `2026-03-19 01:06:51.917`
- `policy.log`: clip restart triggered by clock rewind at `2026-03-19 01:06:51.930`
- `viser`: first post-reset rewound state at `2026-03-19 01:06:51.944` (`~35.5 ms`)

If reset takes seconds, you are almost certainly restarting the whole split sim process instead of issuing a simulator reset.

Relevant files:

- `src/holosoma_inference/holosoma_inference/utils/sim_control.py`
- `src/holosoma/holosoma/simulator/shared/simulator_bridge.py`
- `src/holosoma_inference/holosoma_inference/policies/wbt.py`
- `src/holosoma/holosoma/viser_mujoco_sim_state.py`

## Common Failure Modes

- Stale publishers on the same ZMQ ports make viewers consume old `sim-state`.
- Reset requests can be dropped if they share a saturated control channel and are sent as fire-and-forget.
- Using FK-estimated ref bodies instead of simulator-measured ref bodies causes avoidable tracking drift.
- Flat action scale fallbacks can make MuJoCo behavior look qualitatively wrong even when inference is running.
- Measuring reset on the first received state instead of the first rewound-clock state underestimates true reset latency.
- Browser or `viser` presentation bugs can hide the real issue; always confirm split sim logs first.

## Practical Workflow

1. Kill stale split sim processes.
2. Run `./run_wobj_tracking_validated.sh` and confirm authoritative behavior.
3. If needed, attach `./vis_mujoco_sim_state.sh` to inspect `robot / robot_ref / object`.
4. If carry looks wrong, enable `HOLOSOMA_SPLIT_SIM_STATE_TRACE_PATH` and debug contacts before changing frontend code.
5. Only after split sim looks correct, debug the browser or `viser` presentation layer.
