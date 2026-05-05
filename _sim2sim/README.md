# MuJoCo Sim2Sim Policy Launchers

Run from the repo root:

```bash
bash _sim2sim/contact-aware-near0p3.sh box_75
bash _sim2sim/g1_box_perception_pure_sd_ppo_first_contact14.sh box_75
bash _sim2sim/w5qostjn_linvel_contact_aware.sh box_75
bash _sim2sim/w5qostjn_linvel_action_history.sh box_75
bash _sim2sim/w5qostjn_action_history.sh box_75
bash _sim2sim/w5qostjn_linvel.sh box_75
```

These scripts select the matching W&B run and inference observation config, then call `mj_rollout.sh`.
The policy process is launched automatically, but policy actions still wait for an explicit start command.

Override the model with `MODEL_INPUT`:

```bash
MODEL_INPUT=/path/to/model.onnx bash _sim2sim/w5qostjn_linvel.sh box_75
```

| Script | W&B run id | Inference config |
|---|---|---|
| `contact-aware-near0p3.sh` | `xxehngzo` | `g1-29dof-wbt-object-contact-aware-depth-distill` |
| `g1_box_perception_pure_sd_ppo_first_contact14.sh` | `shoo7sr1` | `g1-29dof-wbt-object-distill` |
| `w5qostjn_linvel_contact_aware.sh` | `tvtwx4to` | `g1-29dof-wbt-object-linvel-contact-aware-depth-distill` |
| `w5qostjn_linvel_action_history.sh` | `haap1tjl` | `g1-29dof-wbt-object-linvel-action-history-depth-distill` |
| `w5qostjn_action_history.sh` | `5aotqbdq` | `g1-29dof-wbt-object-action-history-depth-distill` |
| `w5qostjn_linvel.sh` | `c1gaknfu` | `g1-29dof-wbt-object-linvel-depth-distill` |
