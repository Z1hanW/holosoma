# Realtime visualization validation logs

Snapshot captured on `2026-09-14T16:09:02-07:00` from the G1 deployment host.

- Depth source: `logs/real_depth_20260914_160327`
- CORL source: `logs/real_corl_20260914_160350`
- Code: `sim2real` commit `2cc12454`
- Depth checkpoint camera profile: `n94vaeq7`, D435 at 37.0 degrees down
- CORL checkpoint camera profile: `swl41n4x`, D435 at approximately 37.0 degrees down

The snapshot includes text logs, resolved checkpoint paths, checkpoint-derived
camera profiles, the latest policy telemetry, and the CORL depth-command trace.
Raw depth images are intentionally excluded.

## Complete archive

All 32 available `real_drop` run directories and this snapshot are stored in
W&B artifact [`real-robot-runlogs:v0`](https://wandb.ai/zihanw22/carry-any/artifacts/runlogs/real-robot-runlogs/v0),
created by [archive run `k1judqez`](https://wandb.ai/zihanw22/carry-any/runs/k1judqez).
The artifact contains 97 files and 7,311 valid JSONL records. One blank/truncated
historical record at `logs/real_drop_20260825_201416/depth_command.jsonl:380`
is preserved unchanged.
