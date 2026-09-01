# HoloSoma agent instructions

仓库中较长的科学训练约束和实验交接记录位于 `agent.md`。涉及正式训练、resume、policy init、evaluation 或 W&B 生命周期时，先查阅其中对应章节，不得仅凭旧命令猜测实验契约。

## 正式训练启动不上传视频

未来任何 formal training 的新建、重启、resume 或迁移都不得把 replay/video 录制、上传或远端 W&B media 验证作为启动门：

- launcher/worker 不得要求 `RULE90_*`、`REPLAY_PREFLIGHT_*` 或预先存在的 `vis/replay`；source、data、object、shard、teacher/checkpoint 和 ONNX 合同通过后应直接启动训练。
- `FRESH_WANDB_RUN_ID` 仅可作为可选的预分配 identity；未提供时由训练 logger 创建 fresh run，不能为了取得 run ID 先上传视频。
- 用户显式要求 replay/video 时，使用独立的录制/上传流程，不得阻塞、延迟或改变 training launch，也不得把缺少视频当成训练失败。

## Policy rollout 视频的默认指令：抬起后持续纯前向

除非用户明确要求 checkpoint-native/reference-tracking、指定其他 command，或要求 reference-motion replay，今后录制 checkpoint policy rollout 视频时默认使用“物体抬起后持续纯前向”，不得只录 native motion-derived command 后就作为默认交付：

- 抬起前 root command 为 `[dx,dy,dyaw]=[0,0,0]`，保留 checkpoint-native pickup cue；drop 必须 override 为 0。
- 以该 rollout 初始化时物体 world-z 为基线，首次达到 `object_z - initial_object_z >= 0.30 m` 就触发；`consecutive_steps=0` 表示不 debounce，但不能绕过高度阈值。
- 触发后直到录制结束，实际 actor input 必须逐帧严格为 `[dx,dy,dyaw,drop]=[0.15,0,0,0]`。这里是 constant robot-heading-frame command，不使用 heading lock，不叠加 NPZ/native trajectory command，也不允许 reference motion 的 yaw/drop 继续进入 actor。
- 默认实现参数为 `manual_forward_after_lift_command_m=0.15`、`manual_forward_after_lift_rel_z_delta_m=0.30`、`manual_forward_after_lift_consecutive_steps=0`、`manual_forward_after_lift_preserve_native_pickup_button=true`、`manual_forward_after_lift_preserve_native_drop_button=false`、`manual_forward_heading_lock=false`、`manual_forward_after_lift_command_semantics=legacy_constant_robot_heading_frame`。
- 每条都必须保留逐帧 policy-I/O，验收 trigger 前 root command 为零、drop 全程为零、trigger 后 command 恒为 `[0.15,0,0]`；未达到 lift gate 的条目必须如实标记 `not_triggered`，不能提前发 command、改阈值或伪称完成了 forward 测试。
- 用户明确要求 native rollout 时可以单独录 native 版本；如果同一请求还要默认 forward 版本，两种语义必须放在不同的带时间戳目录并清晰标记，禁止混成一组或把 native 版本误标为 pure-forward。

## 未来正式训练必须导出 ONNX

从 2026-08-03 起，任何新建、重启或迁移的 formal training run 都必须把可部署且与训练语义等价的 ONNX 作为硬交付，不再允许新建 `PT-only` 正式实验：

- immutable run contract、launcher 和最终 CLI 必须显式固定 `training.export_onnx=true`；缺失、为 false 或可被 node ambient 覆盖时，必须在创建 W&B run、启动 tmux 或占用正式训练 GPU 前 fail closed。
- 正式启动前必须用 exact source/config/checkpoint 通过真实 ONNX 导出、`onnx.checker`、ONNX Runtime 加载和 PyTorch-vs-ORT 数值 parity。训练使用的 observation/perception preprocessing、normalizer、command/latch 状态机、recurrent/time state、motion/transition metadata 与 geometry support 都必须在 deployment runtime 中有经验证的等价实现；尚未实现时先补 inference parity，禁止靠关闭 ONNX 绕过。
- 每个要求保存的 checkpoint iteration 必须生成并上传同 iteration 的 `.pt` 与 `.onnx`，二者绑定 completed/next iteration、checkpoint SHA256、source/config/data/observation/command contract 和 ONNX SHA256；不得把 actor-only、缺 command adapter 或未经 parity 的图标成完整 policy ONNX。
- PT 成功但 ONNX 导出、验证、原子发布或上传失败时，该 checkpoint boundary 必须整体 fail closed，不能继续训练并留下新的 PT-only 周期点。terminal checkpoint、terminal ONNX 与 completion marker 必须是同一 iteration；fresh W&B API 必须复核一一配对后才能报告 formal run 完成。
- 旁路 backfill 可用于修复历史 run，但不能替代未来 formal launch 的原生 PT+ONNX 契约。当前已在运行的 `kpl2p2gn` 是启动时明确记录的历史 PT-only 例外，只允许不中断地继续当前进程；不得把该例外复制到新 run，也不得热改其 immutable live contract。若其后需要 resume/迁移，必须先实现并验证 `precomputed_turn_then_forward + runtime pickup latch` 的 deployment parity，并以符合上述规则的新 formal identity 处理。

## 用户所说的“replay 到 W&B”

除非用户明确要求 policy evaluation / policy rollout，否则“把 N 份 replay 录制、log 或上传到 W&B”固定指下面这种批量 **reference-motion replay**：

- 使用 `src/holosoma/holosoma/replay.py` 直接回放 reference motion；它不是 student/teacher policy 推理，不加载 policy checkpoint，也不能标成 evaluation。
- 从目标 run contract 所绑定的最终 immutable single-slot motion view 取完整 effective bank。默认按 canonical clip 顺序让每条 motion 恰好出现一次，不得手选、漏掉、重复或改用同名旧 bank；用户明确要求改变 sequence 时才按其指定的选择/顺序执行并记录。
- 每条 clip 必须使用同一 view 内 `_clip_object_urdf_map.json` 解析出的正确物体，单环境、单机器人、`randomization:disabled`、从 timestep 0 开始、freeze 概率为 0、initial-pose noise 为 0。
- 标准成片为 H.264、1280x720、正常 1.0x playback，并覆盖该 reference sequence 的完整 approach / pickup / carry / drop（若源 motion 含这些阶段）。不要录 frustum/depth debug overlay，也不要把 Viser 交互窗口录屏当作这类 replay。
- 每条录制后必须用 `ffprobe` 验证恰好一个有效视频流、正尺寸、正 FPS、非零帧数和非零时长；全批次必须与 source clips 精确一一对应。上传前至少按各 object category 做人工抽帧检查，确认机器人、物体和动作连续且无 default-pose/错物体替换。

W&B 交付契约：

- 默认建立独立的 `job_type=reference-replay` run（entity/project 默认 `zihanw22/carry-any`），而不是污染训练 run 的 policy-evaluation keys。
- config 必须显式写入 `semantics=reference_motion_replay_not_policy_rollout`、`is_policy_rollout=false`、`policy_checkpoint=null`、parent training-run context、source bank/view digest、object-map SHA256、clip count 和 replay manifest SHA256。
- 每条视频作为独立 history media 写到 `reference_replay/{index:02d}_{clip_id}`；同时写 `reference_replays/table`，并发布一个 `reference-replay-bundle` Artifact，内容为全部 MP4 加 `reference_replay_manifest.json`。
- 上传完成后用 fresh W&B API 复核 run 已 `finished`、独立 MP4 数等于 source clip 数、Artifact 中为 `N` 个 MP4 加一个 manifest，并核对 summary count、manifest digest 与 source-view digest。只有远端复核通过才可向用户报告完成。
- 本地保留带时间戳的 output root、分片日志、逐 clip manifest、最终 `reference_replay_manifest.json` 和 `wandb_result.json`，便于重试与审计。

以上仅适用于用户显式要求的 replay 交付；它不是 formal-training 的启动门，也不得在训练启动前自动执行。

### 已确认的黄金样例

- W&B run：`zihanw22/carry-any/3auc6xt3`
- URL：<https://wandb.ai/zihanw22/carry-any/runs/3auc6xt3>
- 本地结果：`outputs/corl80_actual79_reference_replays_wandb_20260722_190521/`
- 该数据集俗称 CORL80，但 run contract 与磁盘实际为 79 条：box 25、ball 4、barrel 35、bin 15。正确交付是 79/79 个独立 MP4；禁止为了凑名称中的 80 而复制或伪造一条。
- manifest SHA256：`0bcc05af102fe77a2f97391b46e6f51de4ac6942b74638c3b2ce74c10c41fa07`

实现参考（允许参数化复用，但每次必须重新绑定并验证 exact bank）：

- `outputs/prism_debug30_reference_replays_wandb_20260722_182522/record_shard.sh`
- `outputs/prism_debug30_reference_replays_wandb_20260722_182522/upload_reference_replays.py`
