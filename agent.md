# AS / Distill 审查交接

更新日期：2026-07-15（UTC）

## 本轮分工

- `fix_ppo`：审查并修复 PPO、distill、DAgger、teacher normalizer、分布式 checkpoint 与 loss weighting。
- `fix_as`：审查并修复 adaptive sampling、contact/T1 组合分布、rank-local shard、AS checkpoint 与 per-clip asset assignment。
- `fix_launchers`：审查并修复 distill/AS launcher 的 resume、checkpoint、数据集、teacher history、noise 配置和输入校验。
- 主代理：交叉审查修改，补充集成测试，检查分布式保存路径、训练语义和 Git 交付边界。

完整修复记录见 `logs/as_distill_review_fix_20260711.log`。

## 后续修改必须保持的约束

1. PPO 系数必须在 rollout 前按当前 iteration 派生；resume 后不能短暂回到旧的 BC/PPO 状态。
2. teacher-controlled action 不能与会产生 PPO loss 的 rollout 混用，除非同时实现与行为策略一致的 log-prob/importance correction。
3. AS failure bins、采样 timestep 和 contact/T1 权重必须处于同一个 `valid_start_counts` 离散坐标系。
4. `start_at_timestep_zero_prob` 是对零点 delta mass 的显式混合；日志指标必须报告混合后的最终分布。
5. rank-local motion shards 必须携带 global clip count、cover count 和 rank loss scale；world-size/rank/geometry 不匹配时应 fail-fast。
6. 分布式 checkpoint 保存必须由所有 rank 参加 collective，只允许 main rank 写文件；恢复时按 global rank 取回环境状态。
7. `load -> learn -> reset_all` 不能清除已恢复的 AS EMA、统计量或权重。
8. distill launcher 必须校验 actor 类型、输入键、hidden dims、teacher observation history 和数据集 identity，不能静默使用猜测值。
9. 104-rank 正式训练使用 Gloo 默认/小张量 collective、每节点 8-rank NCCL 梯度归约和 13 个 CPU/Gloo leader；不得静默退回 104-rank flat NCCL。
10. rank-visible worker 必须保留原始 local rank/world size 供拓扑和 shard 使用，同时每个进程只暴露一张物理 GPU。
11. fixed-BC 全局预算必须用 quotient/remainder 精确分配；104 ranks 的 4096 样本为 40 ranks x 40 加 64 ranks x 39。
12. fresh student schedule 在 iteration 0 的 PPO 系数必须为 0，但 target/end 是每个实验的显式契约：历史 r21 每 700 iteration 增加 0.1 并在 6300 达到 0.9；r22 每 700 iteration 增加 0.1 并在 4900 达到 0.7，之后保留 0.3 BC 且启用遗忘 guard。不得把任一 run 的默认 schedule 静默套用到另一 run。
13. tmux 创建时必须显式关闭子进程 FD8（`tmux new-session ... 8>&-`）；launcher 父进程仍需持锁直到 ownership 元数据绑定完成。
14. launcher active state、tmux、日志、双端口 reservation 必须绑定同一个 token、command SHA、snapshot、target 和 launch epoch。
15. adaptive KL 只能修改 actor optimizer/LR；critic optimizer/LR 必须独立保存、恢复和更新。纯 BC 阶段的 KL 只能作为诊断量。
16. camera sensor noise 与 reset-pose randomization 必须真实作用于所有 perception backend；声明了非空范围却缺少 runtime tensor 时必须 fail-closed。
17. full resume 必须在任何 live-state mutation 前验证 `iter/next_iter`、actor/critic、启用的 normalizer 与全部 optimizer state；actor/critic LR 必须分别有限、为正且在各自 bounds 内。
18. policy init 只加载 actor 及启用时的 actor normalizer；unused critic 不应阻塞 actor-only 初始化，但实际加载的数值必须全部 finite。
19. actor `std` 的 checkpoint 值必须已经满足有效下界、component floor/mean floor 与 hard cap；full resume、policy init 和 teacher load 都不得先加载再静默裁剪。`max_noise_std` 必须先作用，再检查 mean floor。
20. AS sampler v3 必须序列化并精确核对 kernel size、lambda、uniform ratio 和 alpha；v1/v2 只可在历史生产默认值 `(1, 0.8, 0.1, 0.001)` 下兼容恢复。
21. fixed-BC 保存与恢复必须先通过真实全 rank Gloo envelope 汇总 rank-local state/plan、错误和 runtime allocation fingerprint；budget 为 0 的 rank 也不能提前退出 collective，`noop` 不能与 `restore/clear` 混合。
22. strict teacher load 必须在写入新 teacher module 前验证 checkpoint config、完整 actor state、finite 数值、`std` 域和全部 observation normalizer alias；任何坏 alias 都不能造成部分 normalizer load。
23. 科学 provenance 必须显式记录训练 regime 与 teacher 是否启用；pure RL 使用 domain-separated disabled-teacher identity，不能伪造 teacher，也不能把缺 provenance 的旧 generalist checkpoint 当 scientific resume。
24. runtime/object-bank 资产闭包必须覆盖 visual/collision mesh、OBJ/MTL/全部 `map_*` PBR texture、glTF、COLLADA、PLY 与 URDF texture；OBJ 快速扫描必须覆盖 CRLF 且只解析 `mtllib` 指令行。
25. contact export 目录必须先对 active clip ID 做 exact match，再解释 decimal exporter prefix；exact/numbered 重复映射必须 fail-closed，活动 clip 的坏 metadata 不能静默跳过。
26. student inference observation 必须与 checkpoint 的语义/维度精确匹配：current split history-1=175、object-velocity-v2 history-1=181、current/legacy history-5=875；不得按总维度猜测不等价布局。
27. split perception shared-memory 消费必须按 simulator timestamp 选择 `at_or_before` 帧，禁止把未来 perception observation 注入当前 policy step。
28. immutable AS generation 的 NPZ/map/URDF/contact/metadata/manifest 必须为实体只读 payload，并经递归 fsync 与原子发布；外部大 mesh tree 仍需内容 digest 与 pre-sim revalidation，不能声称已经物理 CAS 化。
29. fresh seed 必须在 W&B、barrier、资产 preflight 全部结束之后、env 构造之前设置；`PYTHONHASHSEED`、`CUBLAS_WORKSPACE_CONFIG`、base seed、global-rank offset 和最大 world size 都必须在任何 CUDA/模拟器初始化前验证并写入 provenance。
30. checkpoint 的生产读取必须使用同一文件描述符完成 `O_NOFOLLOW` 打开、双重 identity/SHA256 检查和 `weights_only=True` 反序列化；resume/policy-init provenance 必须绑定实际加载的精确字节。
31. DeFM 等 lazy 模块必须在模型广播和 optimizer 构造前显式 materialize，并完整恢复 Python/NumPy/torch CPU/CUDA RNG；所有 trainable 参数必须被 optimizer 覆盖。冻结 DeFM 的 BatchNorm 必须保持 eval，可能参与 PPO 的可训练 actor BatchNorm 必须 fail-closed。
32. 永久或当前纯 BC 更新不得构造 PPO likelihood ratio，也不得让 diagnostic KL 改动 actor LR；critic 仍应训练。只要 PPO 系数可能非零，hybrid 路径必须保留 ratio 与 actor-only adaptive KL 控制。
33. actor mean/std/scale、teacher action、全部 PPO loss、梯度、模型参数与 buffers、optimizer state、normalizer 和 checkpoint payload 都不得用 `nan_to_num`、记 0 或 skip 掩盖 NaN/Inf。每个 rollout step 必须在 `env.step` 前以一次合并的 all-rank verdict 检查实际 observation/action/value/teacher/log-prob；任一 rank 异常时所有 rank 同点退出。
34. ordinary 与 streamed microbatch backward、gradient clipping 和 optimizer/std projection 的 rank-local 异常必须进入同序分布式 error envelope；`max_grad_norm` 必须 finite 且大于 0，所有 scientific skip 环境开关在 launcher 与 PPO 内部都必须 fail-closed。
35. batch seed 必须真实落入最终 `--training.seed`，不能只 export；EULA 字符串必须安全 quote，PhysX capacity 必须为正整数，direct generalist rank-visible 模式必须选择 rank-visible wrapper。当前非 deterministic CUDA/TF32 配置只支持统计复现，不得声称 bitwise trajectory reproducibility。
36. teacher/perception/selector observation group 只能在当前 rollout 确实需要 teacher label、teacher control 或 fixed-BC capture 时启用；纯 PPO 不得调用、reset 或推进 teacher 和 teacher-only 随机状态。resume 后必须在第一次 canonical reset 前先恢复当前 iteration 的 objective/active groups。
37. DAgger BC 必须按整份 filled rollout 的全局加权 valid sample 总数归一，再按 minibatch 数得到每步固定分母；禁止每个 contiguous minibatch 独立按自身 valid count 归一。empty minibatch 的 all-rank presence/skip 语义和 collective 次序必须保留。
38. full training resume 必须 `load_optimizer=true` 且同时恢复 actor/critic optimizer moments 与各自 adaptive LR；否则只能称为 warm start，不能作为 exact continuation。policy-init 仍是 actor-only，不得混淆两种契约。
39. canonical checkpoint boundary 必须清空 actor、critic、single/multi-teacher recurrent state；forced reset 前后的 curriculum snapshot 必须拥有独立 storage，并且 episode tracker 的 one-shot suppression 只能消费一次、restore 后不得复活。
40. locomotion penalty curriculum 对空 `env_ids` 必须是 no-op；term scale、live reward weights、env mirror 和 log 必须原子一致。PPO 多 rank 不允许 nonzero-degree rank-local penalty curriculum；degree=0 或 single-rank 才允许。所有 scale/threshold/degree/horizon 必须 finite 且满足范围，enabled tag 必须命中 active reward。
41. `CurriculumManagerCfg.params` 是所有 hook 的共享参数，必须合并进 term params，term-local 值优先；不得出现 manager-level 配置被静默忽略。多节点 exact-resume provenance 还必须包含 NNODES 与 Gloo-barrier mode，elastic max-restarts 固定为 0。
42. partial reset 只能刷新指定 `env_ids`，不得改变 surviving env 的 perception/latency buffer 或全局 camera cadence。Perlin temporal field 只在正常 full sensor tick 推进一次；subset reset 必须复用同帧 full-batch extrema，cache 丢失时由 checkpoint `frame_idx` 确定性重建且不得推进 clock。
43. contact sidecar 的加载条件必须覆盖真实 observation consumers，不只覆盖 adaptive sampler/uniform-T1 开关。contact-aware sparse-root、pickup/drop button 配置了 contact root 时必须加载 t1/t2；sampler/T1 的 strict coverage 仍需 fail-closed。
44. scientific worker 必须在 simulator 前把 provenance 的完整 20-key `execution_runtime` 与 live env 精确绑定，并证明 `WORLD_SIZE=NPROC*NNODES`、global/local/node rank 派生关系及 rank-visible original/remapped aliases。TF32 override 只允许 PyTorch c10 实际接受的精确 `0/1`。
45. 纯 BC 下 entropy 只能是 detached diagnostic；在 PPO 与 `dagger_match_std` 都不活跃时，trainable policy std 必须保持 `grad is None`、不得建立 optimizer state 或被 AdamW weight decay 改动。
46. actor/critic/teacher observation group 的有序输入列表必须各自唯一。重复 group 会让 dict slice 与 rollout concat 错位，必须在模型构造前 fail-fast，不能把错误带入 normalization、checkpoint 或 ONNX/inference。
47. 明确配置 contact sidecar root 且任何 policy observation 消费 contact window 时，所有 active local clips 必须有有效 `[t1,t2)`，不能依赖 launcher-only env 才启用 coverage，也不能对缺失 clip 静默切换 kinematic 语义。未配置 root 的 legacy fallback 和 sampler/metrics-only partial bank 仍可保留。
48. ONNX metadata 必须使用非空唯一 key 和 strict finite JSON。producer 更新已有 key 时必须替换而非追加，拒绝已有 duplicate/空 key/NaN/Infinity；consumer 必须在创建 ORT session 前执行同一 fail-closed 校验。
49. PPO/DAgger 的“是否有 PPO”必须按 float32 actor graph 的实际可表示系数判断；正的 start/target/首个离散 PPO tier 不能 materialize 为 0。BC 补数必须先按 Python `1-lambda` 计算再独立 materialize，且 setup 必须验证 schedule 最后一个数学正 BC tier（target<1 用 end，target=1 用 end-1）仍为正。
50. checkpoint save 的 RNG boundary 必须早于任何 state-dict hook、finite validation、env/fixed-BC collection、serialization 或 artifact publication；成功与异常路径都必须在 `finally` 恢复本 rank 的 Python/NumPy/torch 状态。
51. Flow 参数在 Pydantic、直接模块构造、checkpoint/resume 和 launcher 中必须同约束：steps 为严格整数 `[1,4096]`，train/inference noise 为有限 `[0,1e18]`，time epsilon 为 `[0,0.49]`。不得静默截断，也不得接受会造成近无限 forward 或 float32 Inf/NaN 的数值。
52. 只要训练要求最终 ONNX，time-GRU 或正 inference-noise Flow 必须在 train mode、reset 和 rollout 之前失败；纯 BC 也不能训练到最后才发现导出策略不等价。
53. `/proc` launch identity 扫描必须把 token/command/epoch 的不匹配当作待验证数据，不能继承 while-body 最后一次比较的 rc 而误判为无关进程。错误、缺失、重复或非规范字段均应阻止 terminal closure/新 launch。
54. 多节点 `RUN_REPO` 可能是共享文件系统。run log dir 必须通过 per-publisher staging + mode0400 的 8-field exact owner `(version,snapshot,session,log,target,nnodes,token,epoch)` 原子发布；只有全字段相同的节点可共享，且各 node log 仍须 no-clobber。禁止恢复“每节点都要求整个目录 fresh”的实现。
55. launcher completion/control path 中参与空白分隔协议的 logger base/training name 不得包含 whitespace/control；training name 必须是安全 basename。resume/local checkpoint 必须共用 persisted actor contract，不能分别猜测 MLP/Flow/perception 类型或输入。
56. IsaacSim `ObjectRegistry` state proxy 保持 env-major、slot-interleaved布局；读取、clone 与 reset mock 都必须使用同一顺序，并对越界显式报错，不能退回 object-major 假设。
57. direct RunSim perception 必须从已认证 artifact 精确重建 camera reset translation/rotation、`noise_std_mult`、dropout 分布与 producer tick；每个物理 episode 按训练顺序 reset history→抽样 randomization→生成 reset frame。seed 只保证 distribution/同 execution trace 语义，不得声称跨 batch-size 的 bitwise sample-path 等价。
58. split sim-state、lowcmd、ZMQ/SHM perception frame 必须绑定同一个 episode generation。producer session 使用随机有界基值且物理 reset 递增；SHM reset 必须先 invalidate 当前 slot。policy 每个 control tick 只能消费 `generation == pinned generation` 且 `perception_time <= pinned sim_time` 的帧，缺字段/错代/未来帧必须 fail-closed。
59. perception SHM 的 payload、sim time、checksum 与 episode generation 必须在同一个 odd/even seqlock commit 内；segment generation 与 episode generation 是不同身份。reader history、relay 去重与重建检测不得只看 sequence，否则同名 publisher restart 后的 sequence 2 会被误判为旧帧。
60. external perception relay 不得发布 unauthenticated filler，也不得在 source 断流后重复发布旧帧来刷新 wall age；相同像素跨 episode 必须因 generation 改变而转发。source generation 必须是目标 run_sim 的 generation，不是独立 renderer-local counter；无法证明时保持 fail-closed。
61. scientific fixed-BC guard 只能使用 DAgger 真实 label 和一次冻结的精确全 rank 数据集；必须在纯 BC 窗口内建立至少 3 次参考、在 PPO ramp 完成后才计连续超限，并在每次转移/保存/恢复中核对 dataset digest、完整 state 语义和全 rank 一致性。评估和 checkpoint 不得改变训练 RNG；普通 checkpoint 必须可恢复且未 tripped，trip diagnostic 必须先原子发布、显式不可 resume，然后才做 terminal W&B logging 并全 rank 同序退出。当前 4096 样本主要是确定性 reset-state 遗忘哨兵，不能宣称覆盖了完整 on-policy occupancy。
62. source snapshot 的 identity 包含目录 mode；任何由 builder 人工创建、并非由 `rsync -a` 直接恢复的结构父目录都必须先从对应 source path 规范化 mode，再统一去写位。相同 source 在不同 controller `umask` 下必须得到相同 manifest、snapshot ID 和逐字节相同 archive；defm 的 `submodules` 父目录已作为真实回归 fixture 覆盖。
63. 配置要求 final ONNX 的 DAgger run 必须在最后一个真实 update 生成 terminal fixed-BC proof，即使该 iteration 不落在周期 cadence 上。proof 必须绑定 target/completed/checkpoint iteration、dataset/count/guard state/verdict 和自身 SHA；final checkpoint、ONNX 与唯一 completion marker 必须引用同一 proof。off-grid terminal 观察不能改写周期 guard timeline，full resume/policy init 不能继承陈旧 terminal success。
64. 所有会被下游 wrapper 转换为训练语义环境的 launcher alias 必须在 provenance 前 canonicalize 并绑定；当前 AS 固定值为 perception injection `True`、reset-to-default `False`。batch node payload 必须先清除 node-local alias/HOLOSOMA 值和 `FORCE_EIGHT_GPU_CONFIG`，再导出控制器值；禁止 provenance 后重写 NPROC、GPU token 或 semantic environment。
65. scientific prepare/launch/all 默认必须绑定 runtime contract v2 的 AS-core overlay；其精确根为 attrs/numpy/omegaconf，传递闭包还包含 antlr4-python3-runtime 与 PyYAML。Hugging Face 不属于 Holo scientific 执行闭包，DeFM 只能使用本地 SHA-authenticated strict checkpoint，禁止进入 pretrained/network download 分支。
66. controller 必须从已绑定 archive FD 复制并哈希同一字节流，后续 gzip/manifest/SCP 只能消费该私有 sealed snapshot。节点 installer 必须在同一 per-runtime lock 内完成 stale candidate 回收、archive pathname revoke 复检、严格验证和原子 publish；失败 cleanup 后只能用同一锁的 strict-final/missing reconciliation 判定终态。
67. runtime live binding 必须在 prepare、pre-intent barrier、launch preflight、health 和 train payload 五处验证。pre-intent barrier 必须早于 lifecycle hardening、intent、端口 reservation 和 tmux；status/stop 必须忽略即使 malformed 的 stale training runtime 环境，保持应急控制可用。
68. fresh/policy-init 的 learning target 是 update 数量而非可取 iteration；reset curriculum 的最后可执行端点必须是 `target-1`。只有通过 checkpoint/provenance 精确绑定的 legacy full resume 才可兼容 end==target。
69. terminal policy init 必须同时绑定 checkpoint SHA/private inode、target/completed/next、全量 fixed-BC terminal proof、world size、W&B run、fresh provenance 和 source snapshot；direct API 与 CLI 都必须在 simulator import 前执行，PPO 在 actor mutation 前再验一次。
70. launcher 首参数只要以 `-` 开头就必须先作为 option 保留给 owned-field guard；不能因 option value 以 `.pt` 结尾而把它误吃成 positional teacher checkpoint。
71. source snapshot 的所有路径排序与 tar walk 必须显式固定 `LC_ALL=C`。caller locale、时区和 umask 不得改变 manifest、snapshot ID、archive digest 或逐字节 archive。
72. canary terminal actor 只能作为 formal actor-only initializer。critic、optimizers、iteration、RNG、env、curriculum 和 W&B identity 必须 fresh；不得把 warm start 描述为 exact continuation。

## 验证与已知边界

- 2026-07-15 收尾回归覆盖 89 个项目内 Python 测试文件，结果为 2209 passed、4 expected skipped、0 failed。范围排除 vendored `MotionTrackingG1`/`gsplat`、两个归档 inference clone 和需要外部 IsaacGym/GPU simulator 的 e2e，但保留并通过了 IsaacSim state-accessor 纯单元测试。完整 launcher 在修正一个非 guard sentinel fixture 的显式继承后从头复跑，1573.30 秒后以 `[PASS] launcher contracts` 结束，两轮真实 bank/contact preflight 均覆盖 133/133 clips；生产 guard 检查未放宽。
- r22 双快照预检发现并修复了 defm 分支的 controller-`umask` mode 泄漏：旧 builder 的人工 `submodules` 父目录在 `022/077` 下会成为 `0555/0500`，进而改变 manifest 和 snapshot ID。修复后定向 snapshot 套件通过，真实仓库两次独立构建的 ID、manifest、archive SHA 与 tar 字节完全一致。最终候选为 `src-4c16760193077a478e56aa94aa4289451e0ee4e9f3cd8eed6df380a319228d1d`，archive SHA256 `c92722599c7de310c11b4b12f5249c5244d4cd0ef8d48eca40fc1a8c891dce99`；13 个授权节点均已只安装并重验该快照和 Python runtime overlay，`PREPARE_DATA=0`，没有改动数据或 GPU 任务。canary/formal 全配置 dry-run 均为 13 ranks、零错误，且每个 rank 的 schedule/guard/reset/NCCL-CAS 字段一致；正确 NCCL payload 也已在 13/13 节点逐字节重验。
- PPO/distill、AS/shard/checkpoint、command manager、student inference、ONNX 和 launcher contract 的定向回归均已通过。最终 student-policy 组合为 843 passed；RNG/resume/checkpoint/safe-load/IsaacSim proxy/ONNX/evaluation 组合为 321 passed；排除本机未安装 IsaacGym 的唯一外部 e2e 后，完整 Python 范围为 1730 passed、4 expected skipped。包含该文件时仍是 1730 passed/4 skipped，唯一错误为 import 阶段 `ModuleNotFoundError: isaacgym`，没有代码断言失败。主线程 launcher 全套与独立冻结源码复跑均报告 `[PASS] launcher contracts`，证据分别在 `/tmp/tmp.0orMr8mBts`、`/tmp/tmp.D9fD5dsx1w`。真实双进程 CPU/Gloo 还验证了单 rank loss/clip/optimizer/selector/teacher/action/model-buffer/streamed-backward 异常会让两 rank 同序退出且 pre-env failure 不调用 `env.step`。
- 实际 Isaac rollout smoke test 曾启动，但因同机已有长期任务占用约 33 GiB GPU 显存，PhysX 申请额外约 2 GiB 时 OOM，未形成有效 rollout 结论。
- 当前 `start_at_timestep_zero_prob` 默认从 `0.2` 增长到 `1.0`；训练末期非零 adaptive 分支会趋近于零。这是训练意图选择，本轮未擅自改动。

## 当前正式实验（不要重复启动）

- Session：`as_debug30_ws104_scientific_formal_r21_20260713_183354`
- Source snapshot：`src-25e653d70852ba3216288d0581266706126e240e6d497539d08be8620367e8bf`
- Source archive SHA256：`47839f6f4c7bd8e708471b427e0f031441c692be1c43a9ad30cca59ca484f238`
- Launch token：不写入可 push 文件；精确值仅保留在 node-local ownership state 中。
- W&B run：`bpnn852h`
- 拓扑：13 nodes x 8 GPUs = 104 ranks；每 GPU 64 env，总计 6656 env。
- 通信：default/small Gloo；每节点 8-rank pinned NCCL；13 个 CPU/Gloo leaders；rank-visible 开启。
- Master：`10.99.1.60:29871`；provenance：`10.99.1.60:29872`。
- 目标：fresh 40,000 iterations；checkpoint 与 fixed-BC interval 均为 100。
- 主日志：`/home/ubuntu/FAR/holosoma_runs/src-25e653d70852ba3216288d0581266706126e240e6d497539d08be8620367e8bf/logs/batch_ne/as_debug30_ws104_scientific_formal_r21_20260713_183354_20260713_183354/node_0_10.99.1.60.log`
- 训练目录：`/home/ubuntu/FAR/holosoma_runs/training_logs/carry-any/20260713_184552-g1_29dof_as_debug30_ws104_scientific_formal_r21_20260713_183354-locomotion`
- 初始验收已通过：13/13 token/command/epoch bindings、104/104 ranks、104/104 GPU app、一卡一进程、层次归约正确，所有 104 rank 日志 fatal/NCCL/non-finite 为 0。
- 104/104 ranks 均记录了真实 `far_tracking_warp` stochastic semantics；sensor noise、dropout、reset position 与 angle 范围全部非零。
- iteration 0 是 fresh pure BC：`ppo_coeff=0`，actor/critic LR 均为 `1e-3`，fixed-BC raw samples 精确 4096、rank strata 104。前 10 轮 distill loss 从约 0.3366 降至约 0.1928，训练持续推进。
- 首个正式 checkpoint `model_00100.pt` 已深验通过：SHA256 `a11800b58dfb40a4b1c44f3575d8c9e73dea80b7000ea7d64dc7159a706cbdec`，全部 832 个浮点 tensors finite，104-rank env/fixed-BC state 完整，4096 fixed-BC 配额与 30-clip union 精确，provenance 全匹配。
- BC→PPO 正式边界已直接观测：iteration 699 为纯 BC、actor/critic LR `1e-3/1e-3`；iteration 700 起 PPO/DAgger=`0.1/0.9`，actor LR=`1e-5`、critic LR=`1e-3`，iteration 701 KL 已回落到约 0.0521。
- `model_01000.pt` 已稳定深验：SHA256 `251a5a01bf7e29c2704172a0b7d1473bc28505b0280d744b6f55ed2abff167c8`，iteration 999/next 1000，actor/critic/std/两 optimizer 全 finite，步数均 64000，104-rank env/fixed-BC state 与 provenance 完整。
- `model_02000.pt` 已稳定深验：SHA256 `5be13ce0261a758901cc29aff1842ef8cf23d12c1caa3488b7c5fb9e8ce6a943`，iteration 1999/next 2000，actor/critic/两 optimizer 全 finite，optimizer steps 均为 128000；相对 model1900 恰增加 6400，actor 17/17 与 critic 8/8 参数均更新。104-rank env/fixed-BC state 全部 finite，provenance 未漂移，resume=false。
- `model_03000.pt` 已稳定深验：SHA256 `8f96d2bee29c3a293bde24e458e1394bad296d156e70a51ed338bd3dda11ab1c`，iteration 2999/next 3000，1252/832 个相关浮点 tensors 全 finite，actor 17/17 与 critic 8/8 相对 model2900 更新，optimizer step 均为 192000，104-rank env/fixed-BC、4096 样本、30 clips 与 provenance 完整。
- `model_04800.pt` 已稳定深验：SHA256 `2d3ef5bc440c8068774b91b7fbca8457a2c68f84a7830fb978ce7aff83189228`，iteration 4799/next 4800，1252 tensors/832 floating tensors/23,024,806 elements 全 finite；actor 17/17、critic 8/8 相对 model4700 更新，optimizer step 307200，104-rank env/fixed-BC、4096 样本、30 clips 与 provenance 完整。
- `model_04900.pt` 已稳定深验：92,577,411 bytes，SHA256 `23324d087ded4e0393764f78afd82889f1e0b7a16c650bc925a96240e2219f50`，iteration 4899/next 4900，全部 tensor finite；actor 17/17、critic 8/8 相对 model4800 更新，optimizer step 313600，std min/max/mean 约 `0.0640710/0.0930640/0.0767293`，104-rank env/fixed-BC、精确 4096 分配、30 clips 与 provenance 完整。
- `model_05000.pt` 已稳定深验：92,577,411 bytes，SHA256 `804ea66a4f0b546d78554bd5440d8e56f24bfe3de086f9f144b2d2cc47c03cef`，iteration 4999/next 5000，832 个浮点 tensors/23,024,806 elements 全 finite；actor 17/17、critic 8/8 相对 model4900 更新，optimizer step 320000，std min/max/mean 约 `0.0643381/0.0949393/0.0767764`，104-rank env/fixed-BC、精确 4096 分配、30 clips 与 provenance 完整，无临时/open writer。
- `model_05100.pt` 已稳定深验：92,577,411 bytes，SHA256 `10028311e57f450911d6e9d6b35becd774064ec6bd32bfd1bea7aa18f00068d9`，iteration 5099/next 5100，832 个浮点 tensors/23,024,806 elements 全 finite；actor 17/17、critic 8/8 相对 model5000 更新，optimizer step 326400，std min/max/mean 约 `0.0644743/0.0959619/0.0766708`，104-rank env/fixed-BC、精确 4096 分配、30 clips 与 provenance 完整，无临时/open writer。
- iteration 6300 的最终 schedule 边界已直接观测：iteration 6299 为 PPO/DAgger=`0.8/0.2`，6300 起精确为 `0.9/0.1`，actor loss 可在显示舍入内重构。`model_06300.pt` 为 92,577,411 bytes，SHA256 `da824af61b385a6bfd6224be30747552d32516dd788a0aebf75201292e28135f`，iteration 6299/next 6300，1252 tensors/832 floating/23,024,806 elements 全 finite；actor 17/17、critic 8/8 相对 model6200 更新，optimizer step 403200，104-rank env/fixed-BC、精确 4096 分配、30 clips 与 provenance 完整。
- 2026-07-14 15:48:15 UTC 的严格窗口全 rank exact progress 6605（rank0 随后 6606）；active/tmux/main-log ownership=13/13/13，13 parents，104/104/104 workers/apps/unique GPUs，PID 13/13，error/ECC=0，最高 50 C。PPO/DAgger=`0.9/0.1`，全部指标 finite，`model_06600.pt` 稳定且无 tmp writer。
- 2026-07-14 19:16:15 UTC 的严格窗口全 rank exact progress 7696（随后 rank0 7697）；13/13 nodes、104/104 workers/apps/unique GPUs、PID/GPU binding、error/ECC 均正常，最高 49 C。`model_07700.pt` 随后完整发布且无 tmp writer。direct RunSim episode-randomization/transport 修复属于 post-snapshot 工作，未重启或修改 r21。
- 2026-07-14 11:08:06 UTC 的严格健康窗口在 8.6 秒复核后全 rank 精确 progress 5131（较上一窗口 +48）；13/13 nodes、104/104 workers/apps、PID/GPU binding、error/ECC 均正常，最高温 51 C。PPO/DAgger=`0.7/0.3`，actor/BC/critic/KL=`0.0820/0.2474/0.9635/0.0209`，LR=`1e-5/1e-3`，全部 finite；actor loss 重构误差约 `1e-5`。
- 2026-07-14 10:58 UTC 的严格健康窗口在 19 秒复核后收敛到全 rank 精确 progress 5083（较上一窗口 +55）；13/13 nodes、104/104 workers/apps、PID/GPU binding、error/ECC 均正常，日志 6-8 秒新鲜，最高温 51 C。
- 2026-07-14 10:28 UTC 的健康窗口推进到 progress 4920（随后 rank0 4922/full4921）；13/13 nodes、104/104 workers/apps、PID/GPU binding、error/ECC 均正常，最高温 50 C。4900 边界精确从 PPO/DAgger=`0.6/0.4` 切换为 `0.7/0.3`，actor loss 可由 weighted surrogate+BC 在日志舍入误差内重构。
- 2026-07-14 04:28 UTC 的健康窗口推进到 progress 3035（最新完整 iteration 3034）；13/13 nodes、104/104 workers/apps、PID/GPU binding、ECC/error 均正常，最高温 50 C。iteration 3034 的 PPO/DAgger=`0.4/0.6`、actor/BC/critic/KL=`0.1183/0.1894/0.8080/0.0213`、LR=`1e-5/1e-3`、reward=`10.17`、episode length=`79.70`，全部 finite。
- r21 immutable snapshot 仍使用“每 minibatch 按自身 valid count”旧 BC 归一。由于 contiguous time-major mask density 不均，它只能作为健康/诊断 run，不能作为新修复目标下的最终 scientific 结果。最终新 snapshot 验证完成且授权 GPU 可安全切换时，应 fresh relaunch；在此之前不要无替代地杀掉健康且 ownership 明确的 r21。
- 2026-07-14 01:19:45 UTC 的 10 分钟健康窗口推进到 progress 2047（最新完整 iteration 2046）；13/13 nodes、104/104 workers/apps、ECC/fatal/NCCL/OOM/non-finite 均正常，最高温 52 C。iteration 2046 的 PPO/DAgger=`0.2/0.8`、actor/BC/critic/KL=`0.1345/0.1653/0.6371/0.0191`、LR=`1e-5/1e-3`、std=`0.0856`，均 finite。实验尚未完成，需继续监控。
- r21c2 七卡 canary 已真实验证 BC→PPO 过渡：KL 自适应把 actor 降至 `1e-5` 时 critic 仍保持 `1e-3`；五个 checkpoint 全部 finite 且 provenance 完整。
- r20 虽数值健康，但因旧代码会让 actor KL 同时修改 critic LR，且 perception 随机化未完整生效，已在 iteration 5166 token-safe 停止；禁止把 r20 当作科学结果或 resume 来源。
- 不要从 r16/r18/r19/r20 resume；r21 必须保持 fresh-run provenance。
- 工作区另有 post-snapshot resume/policy-init、teacher、rollout-wide BC、AS v3、all-rank checkpoint、runtime asset closure、pure-RL provenance、contact-ID、split-inference、seed/determinism、safe checkpoint loading、lazy student/DeFM、curriculum 和全链路 non-finite fail-closed 防护；launcher 全套与真实 bank/provenance dry run 均通过。它们不在当前不可变 r21 snapshot 中；应进入下一次 fresh scientific snapshot，但切换前仍需完成验证和安全资源交接。
- r21 的旧不可变 checkpoint 不含新 `rng_state_by_rank` 契约；新工作区默认会拒绝把它声称为 deterministic exact resume。发生故障时不能直接用新代码恢复；只有经明确授权设置 `ALLOW_NONDETERMINISTIC_RNG_RESUME=1` 并记录非 bitwise lineage 后才可兼容恢复。

监控时应至少检查：13 个同名 tmux 均存活、每节点 8 个 GPU app、主日志 `HOLOSOMA_PROGRESS` 持续增长、104 rank 日志无 Traceback/RuntimeError/NCCL WARN/non-finite、29871/29872 reservation 仍属于当前 token。每 10 分钟形成一次全节点窗口；发现异常先定位缺席 rank 和最后 collective，再按 token 精确控制当前 session，禁止按模糊进程名批量 kill。

## 当前正式实验（r22，取代上文 r21 状态）

- Session：`as_debug30_ws104_r22f_20260716_020038`
- W&B：`zihanw22/carry-any/fdbn50pr`
- 初始化来源：已认证 replacement canary `zihanw22/carry-any/gwinraxq` 的 terminal `model_00008.pt`，仅加载 actor/启用的 actor normalizer。
- 拓扑：13 nodes x 8 GPUs = 104 ranks；每 GPU 64 env，总计 6656 env；rank-visible，节点内 NCCL、13 个 CPU/Gloo leader。
- 目标：fresh 40,000 updates；save interval=100，fixed-BC interval=100，motion metrics interval=4。
- Actor：MLP `[512,256,128]`；inputs=`actor_obs_root_contact_aware, actor_obs_drop_button, actor_obs_proprio_with_actions_no_linvel`。
- 优化：1 epoch、64 minibatches，actor/critic LR=`1e-3/1e-3`，initial/min std=`0.01/0.01`，entropy=0。
- Schedule：PPO 0.0→0.7，每 700 update 增 0.1，到 4900 后保留 BC 0.3；reset curriculum 2500→39999。
- Forgetting guard：reference end=600，start=4900，ratio=2.0，absolute mu-MSE=0.160，连续 3 次触发 fail-closed。
- 四个实验性 supervised shortcut 均为 0：`HOLOSOMA_DAGGER_SUPERVISED_ONLY`、`HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP`、`HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH`、`HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD`。
- 初始验收：13/13 sessions、104/104 GPU apps；全 rank progress 已从 8-9 收敛到 14；Traceback/RuntimeError/NCCL/OOM/non-finite=0。
- 2026-07-16 02:17 UTC 十分钟健康窗口：13/13 sessions，全 rank progress 34-35，最大日志年龄 17 秒，W&B=`running`，fatal/NCCL/OOM/non-finite=0。
- 最终验证：Python 2344 passed、4 skipped；launcher full rc=0、7 PASS、0 FAIL、1840 秒。
- 这是当前唯一应继续监控的 scientific run；上文 r21 记录仅保留历史审计用途。

## 工作区规则

- 工作区包含大量既有修改和本地数据；不要清理、回滚或暂存与当前任务无关的内容。
- 本次 Git 暂存范围只能新增 `agent.md` 与 `logs/as_distill_review_fix_20260711.log`；代码修改仍由维护者按计划组织提交。
