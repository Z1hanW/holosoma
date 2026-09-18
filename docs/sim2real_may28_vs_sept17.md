# CORL deployment review: May 28 versus September 17, 2026

## Scope

Historical baseline is `c416c75ac5bd09c3449336ba3de4b466874ff728`, the last
sim2real first-parent revision present by the end of May 28 (commit dated May 26
Pacific / May 27 UTC). There is no separate May 28 commit in the fetched history.
Current reviewed source/log archive is `7721748df49c38d51d29aa447d252c20a5d233f1`.
Git cannot establish uncommitted changes or the exact environment on the robot in May.
During the review, remote commit `1d9776bb` restored the May source. Diagnostic
logging was rebased onto that restoration without reinstating the September
preprocessing, command changes, or deleted launchers. Thus "September current"
in the table below means the archived test version, not the post-rollback branch.

This review concerns successful policy inference with a live camera, not startup
failures. A successfully started controller is not evidence of successful pickup.
No training job or robot process was started, stopped, or reconfigured for this review.

## What actually changed

| Item | May baseline | September legacy real_drop + real_depth |
| --- | --- | --- |
| CORL camera preset | torso XYZ [.01,.01,.44], effective RPY [1.2174,36.9983,1.1157] degrees | unchanged |
| Raw depth / crop / actor raster | 848x480, top16/left32/right32, 87x58 | unchanged |
| Invalid zero depth | interpolated, then clamped near; all-zero region becomes -0.5 | zeros replaced by far before resize; becomes approximately +0.5 |
| Range handling | clamp after resize | clamp before and after resize |
| Resize | effective bilinear (third OpenCV argument was dst) | actual bicubic after 494d16f9 on Sept17 |
| Extra camera queue | fixed 3 frames, buffer4 | random 2 or 3 frames, buffer6 |
| Forward joystick | proportional 0.1*ly, deadband0.1 | fixed0.11 once ly>0.02; negative ly no longer commands backward |
| Keyboard w | increment0.025 | set0.15 |
| Drop | separate input | also suppresses root command and clears offsets |
| Auxiliary processes | basic command display | additional Viser and pose-synchronized sim-depth renderer |

The shared bilinear-to-bicubic change in 494d16f9 also affected legacy entrypoints.
The earlier description that they were unchanged was too broad: their checkpoint
selection was preserved, but their pixels were not. That is an unvalidated
behavior change for historical real-robot performance, even though bicubic matches
the training operator on the tested synthetic simulator raster.

The SW model `swl41n4x_model_15500.onnx` in Git is byte-identical between both
revisions: SHA256 `e70823f20d30ce6c25752e9494aaf9be4e136940ef8af59cf439ec9107b564b0`.
The successful Sept17 SW log resolves that filename; its on-robot file hash was
not recorded, so the robot copy cannot retrospectively be proven identical.
The base policy, observation configuration, robot configuration and Unitree
state/action mapping source are unchanged between the compared revisions.
Stiff-hold blending remains disabled in the normal preset.

## Successful September 17 attempts

Source: `logs/real_drop_20260917_{160342,161827,163858}/` in the reviewed commit.
Times below are the robot's log times, not controller UTC.

| Session | Actual model | Usable evidence |
| --- | --- | --- |
| 16:03:42 | izz2f2df/11000 | 1339 inference rows, finite inputs/actions, drop0 throughout, changing depth |
| 16:18:27 | izz2f2df/11000 | begins with live depth; use only steps0..607 for drop0 analysis |
| 16:38:58 | swl41n4x/15500 | 810 inference rows, finite inputs/actions, drop0 throughout, changing depth |

In the second session drop becomes1 at step608 and stays1; from step868 to3475
all recorded depth summary statistics and seven quantiles are identical. Those
rows are excluded from the valid-carry comparison. They are not evidence that
the pickup policy fails with live input. Quantile equality alone is not pixelwise
proof; older logs do not contain full tensors or capture timestamps.

In the SW session root command is exactly zero for775/810 inference rows and
[0.11,0,0] for35/810; drop is zero for810/810. Therefore a persistent nonzero
drop command is not an explanation for this SW attempt. Logs generally show a
50Hz controller during active inference; no sustained control-loop slowdown is
established. Peak logged angular velocity/action is an observed outcome, not a
diagnosis of why it happened.

Both attempts used the legacy separate launchers, NOT `real_training.sh`.
The camera launcher resolved n94vaeq7/40000 metadata while actors resolved IZZ or
SW; all selected `real_d435i`. This mismatch is an identity/audit weakness, not
proof that different physical extrinsics were applied: the real camera's mount
is physical and the legacy pixel preset is the same for these models.

## Hypothesis and limits

Highest-priority controlled test: same exact SW15500 weights + same recorded
proprioception + same physical raw frame, changing only the historical/current
depth preprocessing. The zero-to-far and interpolation changes are real; their
closed-loop contribution to poor pickup is not established by summary logs.
Latency, physical camera mounting/calibration, and robot/contact conditions remain
separate hypotheses. Do not change several of these together or attribute all
failures to depth without a controlled result.

The new recorder saves lossless metric raw depth, processed/published depth,
capture metadata, exact ONNX inputs/actions, requested targets and gain levels.
Its offline comparator checks model hash, same-host boot, exact pixel matching,
fresh capture age, drop0/active policy, and input-to-action replay parity before
performing single-step depth ablations. It does not claim closed-loop success or
physical calibration. Stale, ambiguous or unmatched frames are not substituted.

## Validation of the logging-only patch

- 10 core recorder/comparator tests passed under hssim; 2 policy/config tests
  passed under hsinference after rebasing onto the May restoration.
- Tests cover lossless copies, bounded recording, explicit queue-overflow/disk
  failure, exact replay, wrong hashes, stale frames, drop exclusion and host-boot
  mismatch. Recording on/off produces identical mocked policy inputs/actions,
  requested targets, commands and gains.
- AST comparison against `1d9776bb` confirms unchanged resize/preprocessing,
  sparse-root command, joystick command, drop handler and hardware-latency
  calculation. Shell syntax and diff whitespace checks pass.
- Local asynchronous logging probe: 200 policy records + 20 raw depth samples,
  no drops/errors; policy enqueue median0.036ms, p99 0.100ms, max0.242ms. This
  measures this controller's CPU/filesystem, NOT the robot's timing or full loop.
- No new real-robot recording or closed-loop ablation was performed. The old
  summary logs cannot be retroactively converted into exact policy-I/O evidence.
