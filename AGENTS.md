# HoloSoma agent instructions

仓库中较长的科学训练约束和实验交接记录位于 `agent.md`。涉及正式训练、resume、policy init、evaluation 或 W&B 生命周期时，先查阅其中对应章节，不得仅凭旧命令猜测实验契约。

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

这与 formal-training Rule-90 的单条 `vis/replay` 不同：Rule-90 是训练前只回放 canonical 第一条 clip 的 provenance gate；这里的用户交付是把 effective bank 的全部 reference sequences 各录一条并在独立 W&B run 中可浏览。

### 已确认的黄金样例

- W&B run：`zihanw22/carry-any/3auc6xt3`
- URL：<https://wandb.ai/zihanw22/carry-any/runs/3auc6xt3>
- 本地结果：`outputs/corl80_actual79_reference_replays_wandb_20260722_190521/`
- 该数据集俗称 CORL80，但 run contract 与磁盘实际为 79 条：box 25、ball 4、barrel 35、bin 15。正确交付是 79/79 个独立 MP4；禁止为了凑名称中的 80 而复制或伪造一条。
- manifest SHA256：`0bcc05af102fe77a2f97391b46e6f51de4ac6942b74638c3b2ce74c10c41fa07`

实现参考（允许参数化复用，但每次必须重新绑定并验证 exact bank）：

- `outputs/prism_debug30_reference_replays_wandb_20260722_182522/record_shard.sh`
- `outputs/prism_debug30_reference_replays_wandb_20260722_182522/upload_reference_replays.py`
