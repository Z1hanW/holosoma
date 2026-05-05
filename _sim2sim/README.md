# MuJoCo Sim2Sim Policy Launchers

Each script is the same rollout path as `../mj_rollout.sh`; only the W&B policy run and matching inference observation config differ.

Run from the repo root:

```bash
bash _sim2sim/contact-aware-near0p3.sh box_75
bash _sim2sim/g1_box_perception_pure_sd_ppo_first_contact14.sh box_75
bash _sim2sim/w5qostjn_linvel_contact_aware.sh box_75
bash _sim2sim/w5qostjn_linvel_action_history.sh box_75
bash _sim2sim/w5qostjn_action_history.sh box_75
bash _sim2sim/w5qostjn_linvel.sh box_75
```

Defaults match `mj_rollout.sh`:

```bash
SIM_MOTION_INIT_MODE=training_default_pose
POLICY_STDIO=inherit
HOLOSOMA_KEYBOARD_ROOT_COMMAND=1
```

The scripts use W&B run URLs, so a running W&B run will resolve to its latest ONNX checkpoint at launch time. To pin a specific local or W&B model, override `MODEL_INPUT`:

```bash
MODEL_INPUT=/path/to/model.onnx bash _sim2sim/w5qostjn_linvel.sh box_75
```

Current run mapping:

| Script | W&B run id | Inference config |
|---|---|---|
| `contact-aware-near0p3.sh` | `xxehngzo` | `g1-29dof-wbt-object-contact-aware-depth-distill` |
| `g1_box_perception_pure_sd_ppo_first_contact14.sh` | `shoo7sr1` | `g1-29dof-wbt-object-distill` |
| `w5qostjn_linvel_contact_aware.sh` | `tvtwx4to` | `g1-29dof-wbt-object-linvel-contact-aware-depth-distill` |
| `w5qostjn_linvel_action_history.sh` | `haap1tjl` | `g1-29dof-wbt-object-linvel-action-history-depth-distill` |
| `w5qostjn_action_history.sh` | `5aotqbdq` | `g1-29dof-wbt-object-action-history-depth-distill` |
| `w5qostjn_linvel.sh` | `c1gaknfu` | `g1-29dof-wbt-object-linvel-depth-distill` |
