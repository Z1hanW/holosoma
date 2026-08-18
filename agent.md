# AS / Distill 审查交接

更新日期：2026-07-20（UTC）

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
73. training-step 的梯度有限性检查可由 actor/critic 各自的 `clip_grad_norm_(error_if_nonfinite=True)` 统一承担，但两者的 norm scaling 不得合并，且同步 clip verdict 必须在任一 optimizer step 前完成；不得同时保留逐参数 finite scan 与独立 all-rank verdict 浪费每个 minibatch 的同步。
74. contiguous DAgger minibatch 的 global presence 应按完整 rollout plan 一次批量归约，并继续按 `sum_r(distributed_loss_weight_r * valid_count_r) > 0` 判定。pure-BC 全局空 batch 必须全 rank 跳过 actor step，remote-only valid 必须全 rank step；random/recurrent 路径在没有精确 index plan 时保留逐 minibatch fallback。
75. hierarchical small-control 优化必须显式启用且默认关闭；当前只允许真实 Gloo 回归覆盖的 int32/int64 verdict 走 local reduce→leader all-reduce→local broadcast。KL、denominator、metrics、bool、complex 和其他 dtype 保持 flat Gloo，不能悄悄改变浮点归约树。
76. 任何 hierarchical subgroup 创建前，所有 distributed rank 必须先通过 default process group 对 flag、world/rank 和 preserved local topology 做一致性判定；不一致时全 rank 在 `new_group` 前失败。进入任一 collective 后禁止按 rank 局部 fallback 到其他 group/backend。
77. `HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES` 是 execution identity。旧 provenance v2 缺少这个由新代码首次引入的字段时只能规范化为 false；显式 true、依赖配置及其他字段仍必须 exact。CPU→GPU leader 切换只能作为新的 fresh canary identity，不能热改或伪装成当前正式 job 的 exact resume。
78. `_update_algo_step` 的 forward/loss/finite/backward 区域必须完全 collective-free；所有可能 rank-local 失败的 KL payload/weight/shape/device 构造都必须在统一 outcome verdict 前完成。KL SUM 可在 gradient reduction 后执行，但 adaptive actor LR 必须仍在 optimizer step 前更新。
79. DAgger authoritative/fallback 模式必须由 rollout 级全 rank 一致事实选择，禁止按 rank-local minibatch 字段是否存在来分叉。fallback count collective 前后的本地准备都必须有全 rank outcome verdict；异常退出必须清空 actor/critic 的 partial/stale gradients。
80. 训练日志可延迟 device→host，但必须 detach graph、拒绝 complex/NaN/Inf、保留源 dtype mean 与历史 Python-float 加法顺序。不得为每个 loss field/minibatch 调用 `.item()`；iteration boundary 应按 device 批量复制。
81. fixed-BC 只在配置 cadence 或强制 terminal proof 时执行；off-grid terminal observation 不得推进或改写周期 guard timeline。非 cadence iteration 不应支付空的 distributed eval verdict。
82. step-level heartbeat 只能由 `HOLOSOMA_DEBUG_HEARTBEAT_VERBOSE` 启用；粗粒度 `HOLOSOMA_DEBUG_HEARTBEAT` 不得在每个 env.step 产生日志或 GPU→host reset-count 同步。
83. WObject curriculum 使用的受支持 tracking error 必须在 reset 前每步刷新；lift/contact/AS 等纯 diagnostics 可按 interval sample-and-hold。任意 enabled `similarity_metric_key` 都要启动时验证，拼错或不支持的 key 禁止静默产生 count=0；logging meter 保存的值不得是随后会原地变化的 simulator view。
84. hierarchical subgroup 必须显式使用冻结且进入 provenance 的 timeout；default process-group/I/O timeout 与 subgroup/NCCL watchdog timeout 是不同契约。不得声称 leader failure 会立即方法级传播；只能依赖已验证的 group order、watchdog 和有界 timeout fail closed。
85. `NUM_MINI_BATCHES=16` 是改变 Adam/PPO trajectory 的吞吐 A/B，不是 math-equivalent 优化。默认必须保持 64；16 需要显式 canary 双钥匙、1 epoch、authenticated snapshot horizon、整除校验和可见的 local/global minibatch geometry。旧 provenance 缺 canary label 时只能规范化为 false。
86. scientific AS batch 的碰撞执行身份固定为 `HOLOSOMA_OBJECT_COLLIDER_TYPE=convex_decomposition` 与 `HOLOSOMA_ACTIVATE_OBJECT_CONTACT_SENSORS=0`；两值必须在 node ambient 清除后由 authenticated payload 重设并进入 runtime manifest。裸 direct RunSim 未显式设置这两值时不能声称复现训练物理。
87. object `convex_hull` 不是当前生产优化：严格 24-rank collider-only A/B 只得到约 1% collection 改善且伴随明显 reward/tracking/fixed-BC 漂移，默认必须保持 decomposition。若未来重审，必须使用独立 fresh A/B 和物理/学习门，不能只看 mesh 面数或单步 FPS。
88. rollout-reference direct gather 的逐字段负索引 guard 会把 17 次两-kernel路径变成 17 次三-kernel路径，已在 24-rank canary 中导致 collection/total 回退并被逐字节恢复。不得仅凭 CPU microbenchmark 或显存流量估计重新引入；任何新方案必须先把 fail-closed 校验安全地提升到一次采样级，再过端到端 acceptance gate。
89. `debug_rollout_viewer` 的灰色 `Training G1` 必须优先从显式 `--original-motion-dir` 解析 exact clip，再兼容旧 rollout metadata。immutable bank 位于 content-addressed `by-source/<digest>` 时，不得因旧 flat path 缺失而把 default pose 误报成 motion 数据，也不得用跨 generation glob 或 flat symlink 掩盖来源歧义。
90. 任何 formal training launch 都必须先通过 motion-object replay 视频门：在最终 source snapshot、effective global motion bank、object map、rank-shard assignment 和 motion-transition contract 全部冻结后，由这些最终训练输入按其规范顺序解析第一条 clip（禁止手选、禁止使用旧 rollout metadata），以训练实际的关节/四元数映射、对应物体 URDF/mesh 及 prepend/append 语义做不含 student policy 的 kinematic replay。必须录制一个且仅一个可解码的 MP4，并在正式 worker/tmux 启动前上传到该次目标 W&B run 的精确 key `vis/replay`；同时绑定 clip ID、motion/object/source/rank-shard/transition digest，旧 bank 或旧 run 的视频不得复用。MP4 缺失、空文件、帧数/FPS/机器人-物体对应关系校验失败、W&B 上传未确认，或尚未完成人工可视检查时，formal launch 必须 fail closed，不能占用正式训练 GPU。
91. 当前无 replay 的 `dagger` 实现是 current-rollout online BC，不是经典的聚合 DAgger。任何新增 replay 必须默认关闭、只消费与 fixed-BC gate 不相交的 teacher-labelled student observations，并使用有界 rank-local buffer、独立确定性 RNG、finite/schema/digest 校验和 full-resume 精确 checkpoint；policy-init 不得继承 replay state。第一版 replay 只允许整个 target 内 operational PPO 恒为 0 的 pure-BC Stage-1，禁止把旧 observation 的 supervised loss仓促混入 PPO likelihood-ratio 路径。
92. formal PPO/BC launcher 必须显式拥有并记录 actor/critic 初始 LR、各自 min/max LR、`adaptive|fixed` schedule 与 desired KL；这些值必须在 source build/SSH 前校验、清除 node-local ambient override、进入最终 CLI/startup log/command identity/provenance。不能再让正式实验静默继承 `adaptive/0.01/[1e-5,1e-2]`。只要 PPO 系数可表示为正，BC 引起的总 actor-policy KL 也不能从 trust-region 控制中排除。
93. fixed-BC 全 8-rank 数据可被拟合到低于 0.160 只证明容量和该冻结分布上的可拟合性，不等于对新 episode、camera pose、latency/noise realization 的泛化。下一轮 student 验收必须同时报告与训练 replay 不相交的 fixed reset sentinel，以及按 rank/phase/contact/episode 分层的 held-out 指标；teacher privileged/noisy observation 与 student 的 pickup/drop/history/temporal-depth 可观测性必须作为显式实验契约，不能用放宽 gate 掩盖 aliasing。
94. privileged teacher 的 `base_lin_vel` 必须是 robot-base frame 的精确 3D 状态项，scale=1、noise=0，并只加入 teacher actor group。当前 history-1 teacher actor 因此是178维，current student 仍是175维；不得把178维 teacher checkpoint 当成 student 或旧175维 teacher 直接加载，distillation 两侧 observation contract 必须分别验证。
95. original-motion / rollout-motion teacher A/B 必须对真实 motion arrays 做内容级区分，同时固定 object map、URDF/mesh、contact sidecar、rank-to-clip assignment、seed、PPO、randomization、source/runtime 与 topology。不能把两个 rollout copy 改名成 original，也不能只看目录名判断 lineage；motion SHA、frame count和共同数组的数值差异必须写入 run contract。
96. “teacher from scratch”要求 `training_resume_enabled=false`、`policy_init_enabled=false`，actor、critic、两 optimizer、RNG 和 iteration 全部 fresh。`WANDB_RESUME=must` 只允许连接训练前已经写入 `vis/replay` 的预绑定 run identity，不得恢复模型状态。周期 checkpoint 固定每1000次 update，普通保存不得 reset live rollout。
97. HIL式并行 tracking/task Stage-2 必须在每条 fixed clip 内独立满足目标 task fraction，不能让按 env index 交替的 mask 与 2-clip round-robin 相关而把整条 clip 固定成单一 regime。fixed env-to-clip 的每 rank clip 数必须整除每 rank env 数；若 warm-start checkpoint 绑定了 camera-hole reference batch size，禁止通过擅改 env 数绕过，必须生成 exact-once、env-compatible 的 1/2/4 等 clip-count rank shards。
98. 从2026-08-03起，未来任何新建、重启或迁移的formal training必须原生交付与训练语义等价的ONNX，禁止再新建PT-only正式run。immutable run contract/launcher/final CLI必须显式且不可被ambient覆盖地设置`training.export_onnx=true`；formal GPU/W&B/tmux副作用前必须用exact source/config/checkpoint通过真实导出、`onnx.checker`、ORT加载及PyTorch-vs-ORT parity。observation/perception preprocessing、normalizer、command/latch、motion/time/recurrent state、metadata与geometry support若尚无deployment等价实现，必须先补实现和测试，不能关掉ONNX绕过。每个required checkpoint iteration必须原子生成、验证并上传同iteration的PT+ONNX，绑定completed/next iteration及两者SHA和完整source/config/data/observation/command contract；actor-only/缺外部command adapter/未经parity的图不得冒充完整policy。任一required ONNX导出、验证、发布或上传失败时checkpoint boundary整体fail closed，不得继续产生新的PT-only周期点；terminal PT/ONNX/completion marker必须同iteration且经fresh W&B API一一配对复核。历史旁路backfill不能替代这一launch contract；当前live `kpl2p2gn`仅是已记录的历史PT-only例外，可不中断当前进程但禁止复制或热改，后续resume/迁移前必须先补齐`precomputed_turn_then_forward + runtime pickup latch` deployment parity并使用符合本规则的新formal identity。

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

## 新增独立吞吐实验（r23exp，不替代或修改 r22）

- 2026-07-16 UTC 盘点到 5 台完全空闲的 8xL40S 节点；其中 `10.99.1.134`、`10.99.0.117` 缺少可用数据/runtime 且本地仅约 4.18/3.69 GB 空间，未启动任务。选择环境完整、磁盘充足且 24 卡均为 `0 MiB / 0% / ECC 0` 的 `10.99.0.18`、`10.99.0.116`、`10.99.0.165`。
- 新实验 session：`as_debug30_ws24_b56f370d2ac8_mb16_gpu_exp_20260716_223353`；W&B：`zihanw22/carry-any/bgivas1t`；source：`src-b56f370d2ac85e47ce2bf777ecc77aa0cd32ecfd684851cb6e7596f0fc73e1bc`。
- 拓扑：3 nodes x 8 GPUs = 24 ranks；每 GPU 64 env，总计 1536 env；seed=42，fresh random，无 resume、policy init 或 W&B identity reuse；目标 40,000 learning iterations，save/fixed-BC interval=100。
- 通信：节点内 NCCL、GPU/NCCL leaders、hierarchical gradient 与 int32/int64 small collectives 开启；CPU leader 关闭；subgroup timeout=300 秒。训练配置为 1 epoch、16 minibatches，必须由 `HOLOSOMA_MINIBATCH_THROUGHPUT_CANARY=1` 显式授权。
- 同源 32-iteration 配对实验均完成且 checkpoint/ONNX/24-rank state 深验通过：MB16=`zihanw22/carry-any/um7o81cn`，MB64=`zihanw22/carry-any/lvv2vs0o`。排除 warmup 0--4 后，MB16/MB64 mean iteration=`4.9626/6.4026s`，mean learning=`0.5322/2.0613s`；MB16 的端到端实测提速为 `1.290x`，learning 部分为 `3.873x`。
- 该 1.290x 仅是相同 source/node/seed/config 下 `NUM_MINI_BATCHES=16` 对 64 的配对结果。两者每 iteration 都有 36,864 rollout samples，但每次 optimizer update 的 global batch 分别为 2,304/576；它改变 Adam step 数、PPO update granularity 与学习轨迹，不能声称与 r22 数学或学习质量等价，也不能把它直接称为对 r22 的纯 infra speedup。
- 完整 launcher contract 随后从头运行约 32 分钟并以 `[PASS] launcher contracts`、rc=0 结束。正式 r23exp 通过 24/24 worker transactional startup handshake；2026-07-16 22:50:36 UTC 三节点 progress 精确同步为 41，tmux/torchrun/workers/apps 每节点为 1/1/8/8，24 卡约 25.86--26.17 GiB、40--45 C、ECC=0；三份完整日志 fatal/Traceback/RuntimeError/NCCL/OOM/non-finite 均为 0。
- 首个正式 `model_00100.pt` 已原子发布并深验通过：93,162,980 bytes，SHA256 `f37f2c00de0c7a283f7bb0e59ec912e06fee573d592281d0af5a0bb8731217dd`，completed iteration=99、next=100、target=40000，全部浮点状态 finite；24-rank RNG/env/fixed-BC maps 精确完整，4096 fixed-BC 样本与 digest 匹配，actor/critic AdamW step 均精确为 `100x16x1=1600`。对应 ONNX SHA256 `bf482b5f513af7c2f6c71e76f83dcede2064bc8ccc03384a17ce250120c4b8e4`。
- 独立 release gate 结束时，三节点 24/24 ranks 均已完成 iteration 148，日志年龄 3--5 秒，错误扫描仍为 0；去冷启动后的周期约 5.05 秒、CV 约 1.9%。实际 perception runtime 明确启用了 sensor noise、reset-pose randomization、非零深度乘性噪声和 dropout。
- r22 session `as_debug30_ws104_r22f_20260716_020038` 与 W&B `zihanw22/carry-any/fdbn50pr` 全程未被修改、attach、重启或停止；它仍是受保护的 scientific reference。r23exp 是额外资源上的 algorithm-changing 长程实验，计划在 iteration 100/600/700/800/1400 做外部健康与学习门审查。

## Simulator / collection 最终状态（2026-07-17 UTC）

- 当前保留的最高价值等价优化是 R6 的 observation base-term same-compute reuse：在 24-rank canary `zihanw22/carry-any/ht0x9jn7` 中，collection 相对 R3b 改善 5.128%（`1.05405x`），总迭代改善 3.290%（`1.03402x`）。连同此前 simulator/collection 改造，相对 b56 reference 的已测结果仍约为 collection `1.787x`、end-to-end `1.633x`。
- 本轮额外保留两项 exact 热路径修复：object-contact reward 缓存 host names，消除每 rank 每 iteration 24 次 GPU→CPU name-index 同步；标准 episodic `motion_ends` 不再执行可证明为空的 per-step clip-rollover scan。custom/continuing/disabled/malformed 情况均保留 fail-closed fallback。
- scientific launcher 当前明确固定 `convex_decomposition`、object contact reporters off；两值写入 runtime asset manifest、W&B 与 checkpoint provenance。直接 scientific RunSim 必须同样 export 这两值；当前裸库默认仍是 hull/reporters-on，二者不等价。
- 已拒绝并精确回退：R7 reset fusion（W&B `4gpqq98b`，慢约 0.7% total）、convex-hull collider（`cdwlh649` 对 decomposition `cbb3wnff`，仅约 1% collection 变化且质量漂移）、deferred rollout finite verdict（两 rank error-path 不安全）、direct rollout-reference gather（`nfnvnumc`，collection +2.857%、total +2.139%）。最终 worktree 不含这些候选行为。
- direct-gather canary 完成 32/32 后已按 exact identity 关闭；`10.99.0.77`、`10.99.0.186`、`10.99.1.154` 均为对应 tmux/process/GPU app/30701--30702 listener 0。r22/r23 的节点、session 与 W&B 未被触碰。
- 下一项值得独立评估的是 MotionCommand 当前 raw motion-frame 的同状态缓存；仅在 `step()` 内就有至少 144 次/iteration 重复 full-frame gather。它必须以 clip/time generation 精确失效并覆盖 reset、clip reassignment、checkpoint restore 与 out-of-band mutation；未通过独立 24-rank 1--2% collection 门之前不得进入默认路径。
- 完整证据、snapshot/checkpoint/ONNX SHA、A/B 窗口与回退记录见 log 第 56--59 节。

## 独立单 motion 实验状态（r26/r27 均已 fail-closed）

- 三组均使用只读 source `src-f51464f69b20762924e79e47011656924b949c15c8e9cb70ea298247685f36f4`，fresh seed42、ws8、每 rank 64 env、16 minibatches、1 epoch、目标 40,000 updates；不从 r25 或其他 policy/checkpoint resume。
- ball：`unscale__any_ball_29`，节点 `10.99.1.21`，session `as_single_ball29_ws8_f514_gloo_mb16_r26_20260717_233502`，W&B `zihanw22/carry-any/1x5hd1in`。
- bin：`unscale__any_bin_29`，节点 `10.99.0.141`，session `as_single_bin29_ws8_f514_gloo_mb16_r26_20260717_233502`，W&B `zihanw22/carry-any/hk07h1af`。
- barrel：`scaledown__any_barrel_25`，节点 `10.99.1.122`，session `as_single_barrel25_ws8_f514_gloo_mb16_r26_20260717_233502`，W&B `zihanw22/carry-any/uwyau5wp`。
- r25 的 flat-NCCL 路径分别在 iter 4/7/86 出现无错误输出的 training-step hang；纯通信 canary 独立复现同型 stochastic hang，强烈指向但尚不能严格证明同一 transport 根因。r26 将 default process group 与 gradient SUM 改为 Gloo，已有 Gloo barrier/small-control 保持开启；policy/data/loss/optimizer 配置不变，但浮点归约顺序可能不同，因此是 fresh identity。
- 三份 formal run 均在 worker 启动前通过 Rule 90：最终 one-motion bank、object/rank-shard/transition provenance 全冻结，人工审核的唯一 MP4 已分别绑定同一 W&B run 的 `vis/replay`。不得删除或用旧 run/video 替换。
- 2026-07-17 23:55 UTC 长窗口：ball/bin/barrel 的 `completed_iteration` 分别为 327/337/331；每组 8/8 workers 与 8/8 GPU apps、一进程一卡，严格 runtime/non-finite/kernel 错误均为 0。三组 `model_00200.pt` 与 ONNX 均非空且发布后继续训练 125+ updates。
- r26 三组后来均由 fixed-BC guard 在 iteration 5101 主动终止，不是 transport crash：step 5100 fixed-BC mu-MSE 分别为 ball `0.374865`、bin `0.325596`、barrel `0.288165`，均连续高于 `0.160`。三者在 step 700 首个 PPO tier 的 KL 分别约为 `1286/1763/1903`，并把共享 actor LR 从 `1e-3` 压到下限 `1e-5`；因此不得原样重复 r26 或把 W&B 的历史 `finished` 标签解释成成功完成。

### r27 single-motion pure-BC bootstrap（2026-07-18 UTC）

- r27 Stage-1 将 PPO 系数严格保持为 0，共运行 1000 updates；其目的只是在 fixed 4096-sample/8-strata 数据集上验证 student learning，避免已复现的 step-700 PPO/KL 冲突。actor/critic LR=`1e-3`，init/min std=`0.1`，MB16/1 epoch、seed42、64 env/rank、Gloo transport、motion/object/contact/physics 均保持同构。reset distribution 在整个 bounded run 固定为 start-at-zero `0.2`、freeze `0.0`。
- fixed-BC 仍每 100 step 评估，reference window 截止 600，guard 从 700 开始，threshold=`min(2x reference_min, 0.160)`，连续 3 次超限即 fail-closed。只有 step 600 达到 `mu-MSE <= 0.160`、teacher mask/finite/no-signal/LR 等门均正确，才允许从该 motion 自己的 terminal actor 进入后续 PPO transition canary；禁止跨 motion 或 generic box policy init。
- source 为只读 `src-3ed93464e82ff5b78b0083502ddab10de006589ad8cfae9c87e79383d355d377`，archive SHA256=`06af018c3b0951e92e0e44a8bd6514f3de6a2d45965a07cfade701bf588a09ae`。相对 r26 source 只改变 `train_agent.py` 的 W&B exit-code finish 语义及其 unit test，训练/回放路径未改变；snapshot tree checksum diff 精确只有这两个文件。
- ball：node `10.99.0.167`，session `as_single_ball29_ws8_3ed934_bcboot_r27_20260718_043223`，W&B `zihanw22/carry-any/vp43rmai`。
- bin：node `10.99.0.141`，session `as_single_bin29_ws8_3ed934_bcboot_r27_20260718_043223`，W&B `zihanw22/carry-any/d8lgzoss`。
- barrel：node `10.99.0.165`，session `as_single_barrel25_ws8_3ed934_bcboot_r27_20260718_043223`，W&B `zihanw22/carry-any/4xheno9a`。
- 三组在分配 GPU 前均重新验证 Rule 90 的唯一 `vis/replay`；随后通过 `distributed_provenance=8/8`、`final_workers=8/8` 和 10 秒稳定门。初始进度为 ball/bin/barrel `5/8/5`，iteration time 约 `2.49--2.56s`；每节点严格为 8 GPU apps，约 `25.7--26.0 GiB/GPU`，kernel Xid/OOM=0。W&B 首批 history 显示 `ppo_coeff=0`、actor LR=`1e-3`、actor skipped-no-signal=`0`、world_size=8、source/replay identity 正确。
- 首个 step-100 学习门已通过执行完整性检查：fixed-BC MSE 从 step 0 到 100 分别为 ball `0.422380 -> 0.331072`、bin `0.383715 -> 0.341447`、barrel `0.397482 -> 0.308193`；三者 actor LR 仍为 `1e-3`、PPO=0、skipped-no-signal=0，且各自 `model_00100.pt` 与 ONNX 已原子发布为非空文件。它们仍高于最终 `0.160` 门，不能提前宣称 student 已通过。
- r27 的 step 200/300/400/500/600 fixed-BC MSE 分别为：ball `0.340293/0.344324/0.354997/0.309249/0.314394`，bin `0.299993/0.299554/0.297290/0.281024/0.278482`，barrel `0.309615/0.311614/0.263864/0.297566/0.284076`。三者 step 600 均未通过 `0.160` acceptance gate，因此任何 r27 actor/checkpoint 都不得作为 Stage-2 policy init。
- guard 在 step 700/800/900 对三条 run 精确得到 `consecutive_exceedances=1/2/3`；step 900 MSE 为 ball `0.345611`、bin `0.334910`、barrel `0.265235`。W&B `vp43rmai/d8lgzoss/4xheno9a` 均真实标为 `failed`，证明新的 exit-code finalization 没有再把 guard failure 伪装成成功。三个 `diagnostic_fixed_bc_guard_00901.pt` 均原子落盘；SHA256 分别为 `5fe9e361a29c341ac934d43ee342805e343eaaa52ebdf9c4b9bae76ba25aa358`、`ddff7f34e385512459de72fa0a5d13a2daa6d6d652348a2c6831c8ab78b48475`、`678ff01df230f4ca84efb02ebd96a482efdbb3b1c31a38670f98545841d0bb86`。
- 终止后只按三条 exact token/session 执行 lifecycle closure；`10.99.0.167/.141/.165` 均已验证 GPU app=0、同名 tmux=0、对应双端口 listener=0，reservation pair 已释放，未触碰其他任务。
- 认证 r26 ball frozen fixed set 的同架构离线诊断进一步定位了失败：从 terminal mean policy 出发，stratified 80/20 在 25 updates 的 held-out MSE=`0.148568`，leave-rank7-out=`0.154866`；500 updates 分别为 `0.089143/0.132605`，说明容量、MSE/backward 和同分布拟合可达到 gate。更严格的 fresh-random 6-rank-train/2-rank-held-out 四折 full-depth 均值却为 train/test `0.120101/0.182170`，只比 lowdim test `0.185141` 好约 `0.003`；这同时证明 current-rollout-only 遗忘和跨 camera/RNG/phase strata 可观测性不足都必须修，不能只调 LR。
- 下一 fresh identity 的优先级是：bounded disjoint DAgger replay + 真正 supervised actor-only Stage-1；保留相同 fixed gate并新增 held-out strata。可部署的 pickup/drop 双按钮和 teacher-label noise 应做显式 A/B；任何 PPO Stage-2 只能从通过这些门的 same-motion terminal actor 以 actor-only policy init 新建 lineage。

## 独立单 motion 实验状态（r28 bounded replay，已按科学门控结束）

- r28 已实现并启用默认关闭的 rank-local bounded DAgger replay：只在全 rank fixed-BC 数据冻结后的后续 rollout 入池；每 rank capacity=512、batch=96、replay fraction=0.5、独立 CPU RNG seed42，raw student observation 用当前 normalizer 重算。checkpoint 精确保存每 rank buffer/counter/RNG/fixed-set binding，policy-init 清空 replay lineage。
- Stage-1 使用真正 supervised actor-only step；PPO coefficient 在完整 1000-step target 内精确为 0，critic loss/optimizer path 为 0，actor LR 固定 `1e-3`。fixed-BC gate、8 ranks x 64 env、MB16/1 epoch、Gloo、seed42、reset distribution 与 r27 同构；save interval 保持 100，因为 checkpoint reset 是算法轨迹的一部分。
- source=`src-26e70b910747dcf48f746683da8a08b577077428b8666322a844e88938e67c7b`，archive SHA256=`1aa1f8b857a7ac29dbbbbd453df385acee1ce29ade769a7302c5505a206e3fe4`。独立 diff 审计确认相对 r27 只有预期 replay/LR/resume/launcher/tests 增量，无 simulator/data/camera/student architecture/teacher drift。
- Rule 90 已用最终 r28 source 和各自最终 one-motion bank 重新录制、人工检查并分别上传唯一 MP4；不得复用 r27 media。ball/bin/barrel 视频 SHA256 分别为 `72facb2e...c1c78a4`、`d2dd7f93...eb83def`、`8a2f770c...eaa496`，replay manifest SHA256 分别为 `a9b763cc...c61ae`、`9540e5fb...5da79b`、`a21580af...7726`。
- ball：node `10.99.0.167`，session `as_single_ball29_ws8_26e70b_replay_r28_20260718_061349`，W&B `zihanw22/carry-any/j3jangxj`，ports `30911/30912`。
- bin：node `10.99.0.141`，session `as_single_bin29_ws8_26e70b_replay_r28_20260718_061349`，W&B `zihanw22/carry-any/ae540b96`，ports `30921/30922`。
- barrel：node `10.99.0.165`，session `as_single_barrel25_ws8_26e70b_replay_r28_20260718_061349`，W&B `zihanw22/carry-any/jx8pwxcw`，ports `30941/30942`。
- 三组均通过 source/runtime/data/teacher/W&B provenance、8/8 distributed provenance、8/8 final-worker 和初始稳定门。W&B 已直接确认 replay 从 step0 inactive/empty 切换为 active、每 rank buffer=512 且 sample/seen counters 持续增长；actor LR=`1e-3`、PPO=0、critic loss=0、skipped-no-signal=0。
- step0→100 fixed-BC mu-MSE 为 ball `0.422158→0.282207`、bin `0.383715→0.274364`、barrel `0.397906→0.282896`，均使用固定 4096 samples/8 rank strata。三份 174.4 MB `model_00100.pt` 与匹配 ONNX 已原子发布且训练继续推进；这只是执行与早期学习门，不是最终 acceptance，仍需等待 step600 和 held-out/generalization 证据。
- 2026-07-18 06:36 UTC 的首个持续健康门已覆盖启动后约 12m50s：ball/bin/barrel 到 iteration `255/257/252`，相对初始快照推进 `+243/+243/+241`，06:31--06:35 每分钟均单调。三节点均为 exact tmux owner、8 apps/8 unique GPUs/8 unique PIDs，显存约 25.8--26.1 GiB/卡、42--48C、ECC=0，worker log/kernel 的 OOM/Xid/fatal/traceback/collective-abort 扫描为 0。
- 全 W&B history 验证每轮 replay draw 精确为 `16×96=1536/rank`、`bc_loss=0.5×current+0.5×replay` 最大残差 `2.33e-9`。同节点同 motion 的 iter20--170 配对测量显示 r28 相对 r27 的真实 wall-throughput 提升 ball/bin/barrel 为 `3.28%/2.08%/1.53%`；replay 未造成端到端回退。约 79--80% wall time 仍在 simulator/collection，后续性能优化重点不在 actor forward/backward。
- r28 的 step600 fixed-BC MSE 为 ball/bin/barrel `0.283126/0.281373/0.299239`，均未通过 `0.160`。step700/800/900 的 exceedance 计数严格为 `1/2/3`，step900 为 `0.275406/0.298222/0.279673`；W&B `j3jangxj/ae540b96/jx8pwxcw` 均真实标记 `failed` 并发布诊断 checkpoint。终止后旧三节点 exact session/process/GPU app/listener 均为0。

## 独立单 motion 长跑（r28b neutralized-quality-stop，当前运行中）

- 用户明确要求优先让三条训练长期运行。r28b 复用完全相同的 immutable r28 source/bank/simulator/actor-only BC/replay 契约，fresh target=`40000`，无 resume/policy-init。fixed-set 每100步仍记录；因 replay 需要其 authenticated capture boundary，guard 结构保留，但质量停止阈值放宽为 `min(reference_min×100, 100)`，不会再因约 `0.3` 的普通有限 MSE 结束。
- ball：node `10.99.1.21`，session `as_single_ball29_ws8_26e70b_noguard_r28b_20260718_070700`，W&B `zihanw22/carry-any/ujovfciq`，ports `31111/31112`。
- bin：node `10.99.1.60`，session `as_single_bin29_ws8_26e70b_noguard_r28b_20260718_070700`，W&B `zihanw22/carry-any/06hwut8r`，ports `31121/31122`。
- barrel：node `10.99.1.122`，session `as_single_barrel25_ws8_26e70b_noguard_r28b_20260718_070700`，W&B `zihanw22/carry-any/hbhgoyhb`，ports `31141/31142`。
- 三条 fresh identity 均各有且仅有一个 `vis/replay` MP4，且通过 source/runtime/data/teacher/replay、8/8 distributed provenance、8/8 final-worker 和10秒稳定握手。2026-07-18 07:18:48 UTC 到达 iter `17/31/18`，均为8 apps/8 unique GPUs，fatal/OOM/collective error=0；三条 W&B 均为 `running`。
- 启动后超过10分钟的持续健康门已通过：2026-07-18 07:28:31--33 UTC，ball/bin/barrel 分别到 iter `231/250/234`；每条仍是 exact tmux 存活、8 apps/8 unique GPU UUID，温度 `37--48C`、uncorrected ECC=0，未见 fatal/traceback/OOM/collective/guard trip。随后 W&B API 独立读到三条均为 `running`，summary step=`234/252/236`，且仍持续增长。
- 完整独立监控窗口 07:23:41→07:33:41 UTC 再次通过：W&B step ball `122→342 (+220)`、bin `140→364 (+224)`、barrel `124→346 (+222)`，state 均为 `running`；07:34 exact log 已到 `359/381/363`。owner token/source/control SHA 全程未漂移，24卡一卡一worker、ECC=0，log+kernel 的 OOM/Xid/CUDA bad-state/traceback/Gloo/NCCL timeout/nonfinite/guard trip 均0。step300 仍严格 PPO=0、critic=0、actor LR=1e-3、replay active/capacity512/draw460800。
- 第三个独立窗口 07:38:47→07:49:39 UTC 同样通过：remote iter ball `456→695 (+239)`、bin `480→724 (+244)`、barrel `461→701 (+240)`；exact pane 与24个worker PID均未变化，8卡/任务、37--48C、UECC=0，log+kernel anomaly=0。iter694/724/701 仍满足 `actor=distill=bc=.5(current+replay)`、draw=`iter×1536`、PPO=critic=0、LR=1e-3。
- 下一版 dual pickup/drop 的 publisher→WBT inference→Viser/manual/reset 与 `mj_track` offset=1 契约已补齐并通过359项相关测试，但这些 working-tree 改动不在 r28b immutable source 内，也未暂存；不得把当前 r28b 误标为 dual-button。
- 为避免影响上述24张训练卡，dual-button candidate archive 仅在零 GPU app 的 `10.99.1.154` 做 CPU-only 隔离复核：297项 student-policy inference tests、bash syntax 与 MuJoCo metadata launcher contract 均通过；真实 ONNX dry-run 因该节点缺少兼容 fixture/runtime 明确 skip。
- dual inference 的 canonical config 与两个 public alias 均已覆盖默认 motion-index offset=1，显式 offset 仍优先，参数化 launcher test 通过。另发现正式 dual launch 的真实语义 blocker：现有 wrist-first sidecar 经10-frame补偿后的 ball/bin/barrel button window 为 `[262,279)/[30,282)/[16,283)`，而 exact motion rel-z carry window 为 `[59,232)/[58,226)/[60,231)`；ball 的 pickup 位甚至在物体落地后才翻转。当前 r28b 不含 pickup-button actor feature，但 ball 的 adaptive+uniform-T1 sampler 也复用该 selector，因此其采样重点同样错误地落在 motion 末端；bin29/barrel25 的 selected window 与 all-carry-region union 相同。三条按用户要求继续作为 baseline 运行，禁止把 `ujovfciq` 伪称 corrected；下一 source 修复后必须 fresh 补跑 ball，并重录展示转折帧的 Rule-90，禁止复用旧视频。
- contact/adaptive selector v2 已在 working tree 同步修改 training、standalone inference 和 patched-artifact builder：选择所有 recognized carry regions 的 union，不再 wrist-first 提前返回；unknown fallback 保留。全30条只读证据从 old `3/30` 与 lift 完全不相交变为 union `30/30` overlap；空闲 `.154` 隔离跑 selector+artifact suite 为27 passed。该 union 只定义 contact envelope，不能冒充 dual phase label；dual 仍需显式、metadata-bound 的 `kinematic_lift` button mode。
- 第四个持续健康窗口以 07:55:48 UTC 的 exact remote baseline `829/863/839` 开始，08:09:55--57 到达 ball/bin/barrel `1140/1177/1153`，净增 `+311/+314/+314`。三条 W&B 仍为 `running`（API step `1136/1174/1146`）；启动时的24个 worker PID 全部原样保留、每任务8个唯一 GPU UUID、温度 `37--49 C`、volatile UECC=0。完整约19 MB/任务日志与自启动前 kernel journal 的 hard-error scan 均为0；step1100 fixed-BC mu-MSE 为 `0.2790/0.3136/0.2764`，均为4096样本/8 strata、exceedance=0。
- 第五个完整窗口 08:12:58→08:25:03--05 UTC 继续通过：exact remote iteration 为 ball `1207→1474 (+267)`、bin `1244→1515 (+271)`、barrel `1221→1491 (+270)`；08:26 独立 pane 复核又推进至 `1495/1537/1511`。24个原始 worker PID 未变，每任务8个独立 GPU UUID，37--49C、UECC=0，runtime/kernel hard hit=0。W&B 均为 `running`；latest fixed-BC 为 ball step1400 `0.2604796448`、bin step1500 `0.3022740897`、barrel step1400 `0.2794506136`，均4096样本/8 strata/exceedance0。完整日志每条各有77行仅发生在启动阶段的 Isaac headless renderer/GPU-foundation `[Error]`，运行阶段未复现且不妨碍持续推进；不得隐瞒或误算为运行期 crash。
- 第六个完整窗口 08:32:11--13→08:45:16--19 UTC 继续通过：ball `1630→1917 (+287)`、bin `1676→1971 (+295)`、barrel `1649→1941 (+292)`。三个 exact pane dead=0，原始24 worker/PID/cwd 全部精确绑定 r28b immutable source；每任务8 apps/8 unique UUID，37--48C、UECC=0，约30MB/任务完整 runtime log 与 kernel hard scan=0。W&B 均 running，summary step `1934/1988/1960`；step1900 fixed-BC MSE=`0.260053633645/0.282545072343/0.269231848290`，均4096/8 strata/exceedance0。
- 第七个窗口 08:49:32→09:01:14--16 UTC 同样通过：ball `2011→2270 (+259)`、bin `2066→2331 (+265)`、barrel `2035→2297 (+262)`；pane、原始24 PID、owner/W&B ID/cwd/source 均未漂移，每任务8 apps/8 unique UUID，37--48C、UECC=0，首次 progress 后 runtime/kernel hard hit=0。W&B running step `2286/2348/2311`；latest fixed-BC 为 ball step2200 `0.24991420318605762`、bin step2300 `0.27610268952817263`、barrel step2300 `0.26957354082951945`，均4096/8 strata/exceedance0。宽正则的两处 pre-progress 命中只是同一 provenance JSON 行中 `NCCL_LIB_SHA256` 与 `HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC` 的跨字段假阳性，不是 runtime error。
- 10:54:44→11:05:28--30 UTC 的最新完整窗口继续通过：ball `4760→4996 (+236)`、bin `4888→5131 (+243)`、barrel `4826→5067 (+241)`。三条均 pane alive、8/8 workers、8 apps/8 unique UUID、UECC0，progress 后 runtime/new-Isaac-Error/kernel hard hit=0；W&B running step=`4996/5133/5071`，latest fixed MSE=`0.273468/0.281991/0.270406`，均4096/8 strata/exceedance0。
- 下一版 dual button 已把 label 与 contact envelope/root carry 明确解耦：dual entrypoint 强制 `contact_aware_button_window_mode=kinematic_lift`，由 source motion 的 `object_z-root_z`、阈值 `min+max(0.10m,0.35×range)` 与连续5帧抬升得到 `t1/t2`；root carry 仍为 `peak_height`，adaptive contact 仍为 union-v2。30/30 motion 严格 lift 验证通过，目标 ball/bin/barrel source window 分别为 `[59,232)`、`[58,226)`、`[60,231)`。
- 训练→ONNX patch→standalone inference 已加入 motion/transition digest 绑定的 source/materialized 整数 window contract，覆盖 static splice 重算、global prepend 映射、resume/policy-init 漂移拒绝；空闲 `.154` 的隔离测试合计565 passed，另有3组 launcher contract、bash syntax/py_compile 通过。formal-fresh dual launcher 同时绑定 authenticated snapshot 中 exact entrypoint path/SHA、95D 四组顺序，拒绝 resume/policy-init；modern stop heredoc 的 `$#` 提前展开 bug 已修并由动态回归覆盖。这些都不在当前 r28b immutable source 中，正式新跑仍需 Rule-90 v2、fresh source/video/run。

## r29 corrected ball29 single/dual（2026-07-18 UTC，当前运行中）

- 最终 immutable source：`src-b39f4d2476443c694c8abfaf09c9305857b0cb7ad0151054b7ea2c61ee4203ff`；archive SHA256=`437b4a5bed3eed4b79c0b37f08a6fa25683857bf2aeb01d9d35369ac12f94e40`。最终 focused regression 为 595 passed，critical launcher 2/2、bash syntax 8/8、pycompile 14/14、snapshot closure/diff 均通过。
- replay timeline 已修为完整 `319 source + 10 prepend = 329` 帧，末帧映射 source 318。single replay/manifest SHA256=`67e2d798...d233e8`/`09e92369...c4cbbc`；dual Rule-90 v2 MP4/manifest/binding=`84add64d...13b02`/`051215ca...7db54`/`38b1722b...38f73`。两条 fresh W&B 在训练前均通过 sole `vis/replay` 的 upload+remote verify。
- corrected single：node `10.99.0.117`，session `as_corrected_single_ball29_ws8_b39f_r29_20260718_100627`，ports `31211/31212`，W&B `zihanw22/carry-any/f46s6237`。actor=`root_contact_aware,drop_button,proprio_with_actions_no_linvel`；history 1，contact feature history 5。
- corrected dual formal-fresh：node `10.99.1.154`，session `as_corrected_dual_ball29_ws8_b39f_r29_20260718_100627`，ports `31221/31222`，W&B `zihanw22/carry-any/8hdketa1`。actor 是 exact ordered 95D `root_contact_aware,pickup_button,drop_button,proprio_with_actions_no_linvel`，全组 history 1；button=`kinematic_lift`，contact selector=`all_carry_regions_union v2`，root carry=`peak_height`。
- 两条均为 fresh seed42、8×64 env、MB16/1 epoch、Gloo、actor-only supervised BC、replay 512/96/0.5、PPO=0、target 40000、save/fixed-set interval=100，无 resume/policy-init。启动后分别达到至少 step `42/231`，每条 pane DEAD=0、8/8 exact b39f workers、8 apps/8 unique GPU UUID、UECC0，W&B=`running`。
- 首个 corrected 持续窗口 10:53:04→11:05:11 UTC 通过：single `129→400 (+271)`、dual `319→590 (+271)`；两条均 pane alive、8/8 ranks/apps/UUID、exact b39f cwd、UECC0、runtime/kernel hard hit=0，W&B running step=`404/596`，fixed MSE=`0.3408/0.3287`、guard exceedance0。每条86个 generic Isaac Error 均只在启动阶段，窗口内零新增。
- `.117` 首次实际 preflight 暴露 external ball29 asset closure 缺 URDF；该失败发生在 torchrun 前并被 launcher 精确回滚。随后仅补齐哈希验证的 base map/URDF（`8eb70e84...ede7`/`0a0cdd79...88ce`，mesh=`9734a65b...d5b9`），CHECK_ONLY 精确选中 1 motion 后，retry2 通过完整 8-worker+10秒稳定门。不得删除失败 log；它是 fail-closed 证据。
- debug replay 的 producer exit 0 不能单独证明录制成功：video recording thread 可能捕获异常且 0-frame encoder 会直接返回。launcher 必须比较启动前后的 MP4 fingerprint，要求本次恰好一个新增/变化的非 symlink、非空 MP4，并以 ffprobe 验证正尺寸视频流和至少一帧；无文件、旧文件、多个候选或损坏文件均须 fail closed，禁止打印成功完成。
- 未来任何 `batch_ne.sh launch/all` 都必须在 live W&B replay 验证、lifecycle intent、端口预留和 tmux 之前，让所有目标节点用认证 snapshot 实际生成并验证同一份 external-AS 闭包：solid selection/contact、single-slot motion/map/URDF/mesh 与全部 rank shard。所有节点的 byte/digest/path/world-size identity 必须完全一致；Rule-90 v2 还必须与 replay manifest 的 exact inputs 一致。这些值要封入 train control，并在真正的 solid/perception wrapper 中再比对，禁止出现“preflight 用 A、training 用 B”。`PREPARE_DATA=0` 时缺 map/NPZ/URDF/mesh/contact 必须在上述边界 fail closed，不得到 torchrun 前才暴露或静默从其他节点拷贝。
- 2026-07-18 用户明确要求停止监控；五条任务的周期只读 monitor 均不得继续或重建。禁止为了清理 W&B、修 corrected 任务而重启、attach 或修改任何现有训练进程。

## 工作区规则

- 工作区包含大量既有修改和本地数据；不要清理、回滚或暂存与当前任务无关的内容。
- 本次 Git 暂存范围只能新增 `agent.md` 与 `logs/as_distill_review_fix_20260711.log`；代码修改仍由维护者按计划组织提交。

## External-AS 正式 launch 门（2026-07-18）

- 正式 AS training 的 checkpoint 生成/上传间隔是硬契约：`SAVE_INTERVAL=1000`，即只在每1,000个 learning iteration 生成周期 PT/ONNX；`batch_ne.sh prepare/all/launch` 必须在 snapshot、W&B、SSH 和任何远端副作用前拒绝所有其他值。禁止为了 fixed-BC evaluation 的100步 cadence 把 checkpoint cadence 同步降到100；两者必须独立。短时隔离测试可以 mock 文件，但不得以非1,000配置冒充正式训练。
- `batch_ne.sh prepare/all/launch` 的 selected-GPU idle preflight 是强制门：先在任何 heavy asset/runtime、W&B 或 lifecycle 操作前执行 early probe；`launch` 在可选 health/contact 后、staging/tmux 前再次执行 pre-launch probe。`SKIP_NODE_HEALTH_CHECK=1` 只能跳过可选 health，禁止跳过这两个 idle gate。early probe 必须只用 stdlib Python + `nvidia-smi`，不得 import torch、创建目录/intent/端口或初始化 W&B。
- 正式 AS launch 必须在 W&B replay verify、intent、端口和 tmux 前完成全节点 external-AS 闭包：exact motion NPZ、object map、URDF 及递归 mesh、solid/contact、single-slot source/view、source-derived rank plan。所有节点的 path/digest/world-size identity 必须一致；这些值要封入 immutable train control，并由实际 solid/perception wrapper 在 worker entry 前重验。
- rank manifest 不能自证。publisher 与 current verifier 都必须从 immutable source closure 重新推导 exact plan；source/view/rank namespace 的 regular files/dirs 分别 seal 为 `0444/0555`，旧树迁移只允许在 namespace exchange 内临时 thaw，完成后必须 reseal。输出根及祖先要做 lexical no-symlink/non-directory 检查，禁止通过 symlink 重定向闭包。
- generic lifecycle 回归可以使用临时、内容完整的最小 one-motion external bank，以避免每个故障注入重复扫描真实大 bank；但必须保留独立真实 Rule-90 v2 contract，覆盖最终 motion/map/URDF/mesh 与 solid/contact/source/view/rank drift。dry-run idle probe 当前的可审计前缀是 `bounded-ssh`，断言不得再按旧 bare `ssh` 标签计数。
- 2026-07-18 最终证据：完整 launcher contract rc=0、`[PASS] launcher contracts`、2158.23 秒；真实 external-AS contract rc=0、52.81 秒；rank-plan 68 passed；bash syntax、Python compile、targeted diff 与 residue 检查通过。测试结束后不得遗留 launcher process、runtime-transfer temp 或 `.external-as-contract-test-*`。
- r28b ball/bin/barrel 三条正式 8-GPU 长跑继续受保护。15:57:09→16:08:17--19 UTC 分别推进 `+244/+250/+247` 到 `11625/11883/11763`；24/24 原 worker、8 apps/8 UUID/任务、UECC/runtime/kernel/W&B/fixed-set 均正常。禁止为了后续修复 attach、重启或改变它们。
- corrected single/dual 到 16:09:17 UTC 仍硬健康且分别推进 `+259/+261`；single fixed-set 曾升到 `0.4335`、最新为 `0.3773`（reference `0.3253`），dual 为 `0.2866`（reference `0.3224`）。这是截至停止监控时 single 的高波动泛化平台软预警而非进程故障；不得伪称 guard acceptance，当时 dual 是更稳定的 corrected 候选。
- 2026-07-18 W&B 存储清理严格限定为 `state != running` 且 runtime `<3h`：22 条已删短跑先按 exact ID 恢复、逐条重验状态/runtime，再以 `delete_artifacts=True` 级联删除；直接清掉约 `10.55 GB` run files，最终 22/22 ID 均不可见且近期剩余短时非 running run 为0。`ujovfciq/06hwut8r/hbhgoyhb/f46s6237/8hdketa1` 五条 running 未动。W&B 仍保留8个后端禁止用户删除的 system-managed history exports（合计约0.48 MB），等待 Cloud GC，不得伪报为已直接删除。
- 随后按用户明确要求清理上述五条 active run 的 W&B checkpoint backlog，但不触碰训练进程或本地文件：原有 1,078 个每100步上传的 `model_<step>.{pt,onnx}` 合计约 `94.67 GB`；保留每5,000步里程碑及操作时各 run 最新 PT+ONNX，删除1,052个旧文件共 `92,382,547,244` bytes。API 复核仅余26个模型文件、`2,283,220,103` bytes，replay/配置/指标均未动。两轮直接 run-file 清理累计约 `102.93 GB`；W&B 页面用量可能等待后端 GC 才回落。用户要求完成后结束，因此不得继续监控。
- 用户随后把6个 API 确认 `>10 GB` run 的 W&B checkpoint retention 明确为每1,000步：删除所有 `step % 1000 != 0` 的 `model_*.pt/.onnx` 共652个、`57,790,000,638` bytes；复核零违规文件，保留76个整千步模型文件共 `7,785,008,967` bytes。范围包含五条 active run 和39.64小时后 crashed 的 `bpnn852h`；replay/配置/指标/本地文件/训练进程未动。当前五条 active job 已加载的实际 save/upload interval 仍为100步，故这只是一次性清理，禁止伪称已永久改成1,000步或留下后台 monitor。

## W&B replay/evaluation 可见性交付硬约束（2026-07-18）

- pre-launch Rule-90 replay 的 summary upload 仍是正式训练前的身份门，但 **summary-only 不算最终可见交付**：原 training run 产生 history 后，必须由 rank 0 把同一份、同一 SHA256 的唯一 MP4 作为 history media 写入 `vis/replay`。只有 fresh W&B API 同时证明 `historyKeys` 含 `vis/replay`、恰好一条 media history row、summary 指向该 history-backed path，且远端文件非空，才算 W&B replay 可见。
- 正式 run 停止或完成前必须从该 run 自身保留的整千步 student checkpoint 录制 policy evaluation，并在 **同一个原 training run 的同一条 history row** 至少写入 `vis/evaluation_student` 与 `vis/evaluation_motion_generator_teacher`；禁止另建 eval project/run 冒充交付。`motion_generator_teacher` 必须是生成当前 input motion 的 exact checkpoint，而不是从 student provenance 猜出的蒸馏 label teacher；若两者身份不同且后者也需要展示，另写 `vis/evaluation_distill_label_teacher`。禁止继续使用含义不明的 `vis/evaluation_teacher`，已有错配内容必须标记 deprecated。所有 evaluation 均为 `NUM_ENVS=1` 单机器人，并记录角色、checkpoint ref/SHA、motion/asset SHA、randomization/noise 开关。旧 generator checkpoint 缺 modern timeline contract 时，只允许在 evaluation mode 下以 exact pinned SHA 与 actor/observation semantic identity 认证加载；不得把它当 training resume/policy-init。单环境不得改变训练 perception 语义：训练期 camera-warp reference batch 必须在最终 runtime env count 已知后显式绑定并通过 policy-identity validation，禁止用 CLI raw override 或关闭校验绕过。
- replay/evaluation MP4 在上传前必须用 ffprobe 验证存在视频流、正尺寸、正帧率和非零帧数；上传后必须以 fresh API 重新验证 history row、summary path、远端 size 与 checkpoint step。旧 summary-only replay 只能在新的 history-backed replay 验证成功后删除，避免破坏唯一可见副本。
- 2026-07-18 用户明确停止的五条 r28b/r29 训练均已按 exact lifecycle identity 关闭，W&B 已收口为 `finished` 且显式记录 `lifecycle/user_stopped=1`、`lifecycle/training_target_reached=0`。`ujovfciq/06hwut8r/hbhgoyhb/f46s6237/8hdketa1` 现各有且仅有一条 history-backed `vis/replay` 和一条 `vis/evaluation`；evaluation checkpoint 分别为 `18000/19000/18000/14000/14000`，五条视频均为 `640x360@50fps`、501帧、10.02秒。
- 停机后新产生的18个非整千步 PT checkpoint 已从上述五个 W&B run 精确删除，释放 `3,139,676,584` bytes；5个验证后失去引用的 summary-only replay 副本另释放 `7,253,939` bytes。fresh API 复核五条 run 的非整千步 `model_<step>.pt` 均为0，两个 media history key 与实际文件均仍有效。不得重建这些训练、evaluation 或周期 monitor。
- 2026-07-18 23:20 UTC 已按上述新契约补齐五条 run 的 teacher/student 单机器人配对结果。`ujovfciq/06hwut8r/hbhgoyhb/f46s6237/8hdketa1` 的新 history step 分别为 `18649/19106/18881/14366/14599`；每条均恰好一行同时含两个新 key，每个 key 的 history type count 均为1。十段视频全部为 `640x360@50fps`、501帧、10.02秒，人工抽帧确认每段仅一个机器人；fresh API 逐条核对 summary SHA、远端 file size、checkpoint step、`evaluation/num_envs=1`、`evaluation/robot_count=1`，且五条 run 仍为 `finished`、`user_stopped=1`、`training_target_reached=0`。原 `vis/replay` 未改动；旧 64-env `vis/evaluation` 为审计保留并显式标记 deprecated，不得再作为主 evaluation。

## Motion-generator teacher 身份门（2026-07-18）

- teacher rollout publication 必须把生成 motion 的 checkpoint ref、`saved_wandb_path` 与实际文件 SHA256 作为 `teacher_lineage` 写入 canonical publication payload；solid bank 必须把 canonical `motion_generator_teacher` 与 source rollout manifest file record 纳入 `source_identity/source_digest`，single-slot bank 必须继续纳入自己的 manifest/view digest。任一层缺失、部分字段、格式错误或上下游不一致均 fail closed。
- 正式 `batch_ne.sh prepare/all/launch` 必须在 W&B、launch intent、端口和 tmux 之前，从全节点相同的 immutable bank 解析 generator SHA；实际 distillation-label teacher 解包后必须重新 hash，并在 worker/entrypoint 前与 generator SHA 比对。默认 `REQUIRE_MOTION_GENERATOR_TEACHER_MATCH=1`；只有明确设计并记录的 cross-teacher 实验才可设0，且两个角色身份仍必须分别保留。legacy bank 无 lineage 时必须显式给出 `MOTION_GENERATOR_TEACHER_EXPECTED_SHA256`，禁止用当前默认 teacher 或文件名猜测。
- 当前 real-mesh rollout bank 的 generator 是 `wandb://zihanw22/carry-any/u8udzw0u/model_05000.pt`，恢复文件 SHA256=`80cb13e13590239d015ba0a29bdbae901b7785a9789d144745c7bd330059cd68`；此前默认蒸馏/所谓 teacher eval 的 `wandb://zihanw22/carry-any/bcleb5oi/model_67000.pt` SHA256=`1c441a7eea24fb28d67cc4b5edeb123b91d589ec095d383034f32210a87b6c5b` 是不同策略，不得再作为该 motion 的 generator teacher。正式训练选择前者，或把后者作为经过明确批准的 cross-teacher label policy；不得静默混用。
- generator-teacher motion validity evaluation 必须复现 rollout source 语义：exact motion/object asset、`NUM_ENVS=1`、关闭训练期 domain randomization、camera pose randomization、depth multiplicative noise/dropout 与 offline reward-only sidecar；额外 post-motion hold 必须与主 motion success 分开报告，不能用 hold 末端失稳反推生成动作失败。
- evaluation CLI 角色必须显式：`HOLOSOMA_EVAL_POLICY=checkpoint_actor` 运行传入的 exact checkpoint（student 或单独认证的 motion generator），`distill_label_teacher` 才从 student provenance 加载 DAgger label policy；禁止含义不明的 `HOLOSOMA_EVAL_POLICY=teacher`。

## r30 same-teacher 三条单 motion 正式重跑（2026-07-19 UTC，当前运行中）

- 本轮严格使用生成 input motion 的 exact u8 teacher，同时作为 distillation label teacher：`wandb://zihanw22/carry-any/u8udzw0u/model_05000.pt`，本地只读恢复文件 `/home/ubuntu/.cache/holosoma/checkpoints/motion_generators/80cb13e13590239d015ba0a29bdbae901b7785a9789d144745c7bd330059cd68/model_05000.pt`，SHA256=`80cb13e13590239d015ba0a29bdbae901b7785a9789d144745c7bd330059cd68`；`REQUIRE_MOTION_GENERATOR_TEACHER_MATCH=1`。禁止把旧 bcle label teacher 静默混入这三条 run。
- immutable source=`src-46291ccac7340a0bdd8e6c8d5fefd47a7e00fad507f1f0d642dfb1189726af3e`，archive SHA256=`e337a2595ff23bdfaa5a037c4174a5b8b76219c26713924b29dd637b9803415e`。三条均为 fresh seed42、单 motion、8 ranks x 64 env=512 env、Gloo、1 epoch/16 minibatches、actor-only supervised BC、bounded replay capacity=512/batch=96/fraction=0.5、PPO coefficient=0、actor LR=`1e-3`、target=40000，无 resume/policy-init；fixed-BC 每100步，周期 PT/ONNX checkpoint 严格每1000步。
- ball `unscale__any_ball_29`：node `10.99.1.21`，session `as_same_teacher_ball29_ws8_r30_20260719_012337`，ports `31411/31412`，W&B `zihanw22/carry-any/q9qn6xb9`。
- bin `unscale__any_bin_29`：node `10.99.1.60`，session `as_same_teacher_bin29_ws8_r30_20260719_012337`，ports `31421/31422`，W&B `zihanw22/carry-any/b72gh1wx`。
- barrel `scaledown__any_barrel_25`：node `10.99.1.122`，session `as_same_teacher_barrel25_ws8_r30_20260719_012337`，ports `31431/31432`，W&B `zihanw22/carry-any/5utvhw89`。
- 三份 Rule-90 replay 均由最终 source/teacher/one-motion bank 重新录制并人工抽帧确认：H264、1280x720、50 fps、419 frames、8.38 s、单机器人且 pickup/carry 连续；ball/bin/barrel MP4 SHA256 分别为 `02795f502af34cd77d1e643bb407e735a021478cceae1b6f69b6c6949baf667c`、`14d42b7b9e3c995a41ecabfd656b62dad3a82c129565190a92f006f1536caca1`、`5cbf07b380bf6e1a15384382bfd990c18936d8941ac267b0528628238aa22b42`。三条 run 在 GPU 分配前均完成唯一 pre-launch `vis/replay` 上传与 fresh API 验证；history-backed 最终可见性交付仍按上文约束执行。
- 第一次正式调用因新 snapshot 尚未安装到三节点而在 W&B/lifecycle/GPU 前 fail closed；仅执行 `PREPARE_DATA=0` 的 snapshot prepare/verify，没有启动训练。第二次在 mandatory external-AS exact one-motion closure 已通过后，被 optional raw-bank node-health 仍按30-motion源库检查的重复条件拒绝；launcher 随即精确回滚 run lifecycle/process/ports，worker 仍未启动。最终 retry 仅设 `SKIP_NODE_HEALTH_CHECK=1` 跳过该可选重复检查；两个 mandatory GPU-idle gates、external-AS closure、teacher equality、W&B replay、distributed provenance、final-worker 与10秒稳定门全部保留并通过。
- 2026-07-19 01:40 UTC 初始验收：ball/bin/barrel remote completed iteration=`38/46/40`，iteration time=`3.11/3.09/3.21s`；三台 exact tmux 均 alive，每台严格8 compute PIDs/8 unique GPUs。fresh W&B API state 均为 `running`，summary step=`39/46/40`，且三条均存在 `vis/replay`；三份实际 run config 再次确认 `algo.config.save_interval=1000`、fixed-BC interval=100、1 epoch/16 minibatches。本轮验收到此结束，不创建周期 monitor。

## r30 3k 单机器人配对 evaluation 交付（2026-07-19 UTC）

- 三条 run 均从各自保留的 `model_03000.pt` 评估；checkpoint 内部是 completed iteration 2999、next iteration 3000。student checkpoint SHA256 分别为 ball `68a040e3c1a445fef207b961a428325da4cf79f7f04adc18d0400d9756b153a2`、bin `ea05f677ebcc36f63bd56d04c1fba727a2b4dbe425bcebb9e031977104cbb282`、barrel `b4b880af5b40da8418c460a54513274797b42c67a24db1902668f18336941f67`。
- 每条 evaluation 都是 `NUM_ENVS=1`、单机器人、exact one-motion/object bank。student 角色为 `checkpoint_actor`；teacher 角色为 checkpoint provenance 认证的 `distill_label_teacher`，且其 SHA256 与 motion generator 完全相同，均为 u8 `80cb13e13590239d015ba0a29bdbae901b7785a9789d144745c7bd330059cd68`。没有再使用含义不明的 `vis/evaluation_teacher`。
- evaluation 使用 immutable source `src-46291ccac7340a0bdd8e6c8d5fefd47a7e00fad507f1f0d642dfb1189726af3e` 的逐任务可写 runtime clone；运行前后所有 manifest 中的1008个 source 文件均通过 SHA256。配置使用 canonical `randomization:disabled`，日志确认 Event Manager 为0 active terms、camera reset pose randomization=false；为保持 checkpoint perception contract，`camera_apply_sensor_noise` 配置值未被篡改，但 runtime std-mult/dropout state 均为 none，因此 depth multiplicative noise/dropout 实际未施加。
- 三份 source motion 均为319帧、50 Hz。原始录制保留了501帧用于完整审计，但后182帧已经超出输入 motion，不得混入主 tracking eval；另有最前4帧属于 renderer warm-up。最终上传严格取 source-video frames 4..318，共315帧、640x360、50 fps、H.264、6.30秒。六段视频均经 ffprobe、SHA256 和六时刻人工抽帧确认，只有一个机器人且 ball/bin/barrel 物体匹配。
- ball run `q9qn6xb9` 在同一 history row 3748 写入 `vis/evaluation_student` 与 `vis/evaluation_motion_generator_teacher`；远端文件分别为818,463/807,891 bytes，SHA256=`6b034fd48251d7991ab2e4740ddc5b53ff275b66f3faf0f8fced8e543da73828` / `fd0f9edf5bb9d9b2446c972a1f82b5a59b07360f6fcca2ea657a0a4dd7e3d58d`。
- bin run `b72gh1wx` 的配对 history row 为3818；远端文件分别为540,121/772,234 bytes，SHA256=`ca484913d02618916dcd713d62df93c2786f51546e1def78c3d460b789a5b882` / `a7ba356d7c5f604156565c7695ae8bd64da7864dd5f689a152e441352c7bf8a9`。
- barrel run `5utvhw89` 的配对 history row 为3784；远端文件分别为760,462/797,726 bytes，SHA256=`f89f9aaaa21f1bd88a20b7c655a1eb341ee74562dd79ae21007d8ca872dbbf29` / `695baeacc554e4e43ec795aaea5d53fddbce0d4028cc8a971cea464e372cd7a8`。
- W&B history row 的 `evaluation/checkpoint_step` 均严格为3000；history 的 `_step` 是训练已越过3k后 shared writer 获得的单调上传序号，不能倒写成3000。fresh API 对三条 run 都验证恰好一行同时含两个媒体 key、完整角色/asset/randomization metadata，并从远端重新流式下载六个文件逐字节复算 SHA256。最终 API state 仍全部为 `running`，未创建任何新 eval run，也未停止、attach 或修改正式训练进程。

## rollout30 的 64 卡 privileged teacher（2026-07-19 UTC，当前运行中）

- 正式 run 是 `zihanw22/carry-any/toxsjobi`，name=`priv_teacher_rollout30_ws64_u8init_dynmatch_srcfix_20260719_071125`。每节点 session 均为 `priv_teacher_r30_ws64_toxsjobi`，run root 为 `/home/ubuntu/FAR/holosoma_runs/formal_priv_teacher_rollout30_ws64_20260719_0635`；禁止重复启动或复用该 W&B identity。
- 拓扑为 8 nodes x 8 L40S = 64 ranks：master `10.99.0.141:31641`，其余节点为 `10.99.0.186`、`10.99.1.154`、`10.99.0.167`、`10.99.0.77`、`10.99.0.165`、`10.99.0.227`、`10.99.0.18`。每 rank 1024 env，全局 65,536 env；24 rollout steps、7 epochs、4 minibatches。
- 这是 fresh privileged PPO teacher，不是 student distillation，也不是 full-state resume。actor 从 motion generator `u8udzw0u/model_05000.pt` 仅做 policy-init，文件 SHA256=`80cb13e13590239d015ba0a29bdbae901b7785a9789d144745c7bd330059cd68`；critic、两 optimizer、RNG 与 iteration 均 fresh，从 iteration 0 开始。W&B 的 `provenance/training_regime=pure_rl`、`teacher_enabled=false` 表示没有外部 label-teacher 参与 PPO，并不表示本 run 不是要训练的 privileged teacher。
- 训练数据是 u8 rollout 的 exact 30 条 motion-object trajectory。64 个 rank-local shard 对 30 clips 做必要的重复覆盖，但 `distributed_loss_weight` 使用 inverse-cover 权重，精确恢复 uniform 30-clip objective；运行中每轮 `Train/distributed_loss_weight_sum=64.0000`。motion-shard semantic provenance SHA256=`8bb025b05d6028f3d85c133b8e10ba19ddb54e80b77e8f54ff1f123fed6d5175`。
- dynamics 使用与当前 student 匹配的 `g1_29dof_wbt_w_object_with_action_delay`；reward 是当前 rollout-motion tracking 加 exact offline contact guidance，不再把旧 original motion 当 reference reward。runtime default-pose prepend=0.2 s，contact interval 已启用相同的 prepend compensation；startup zero-timestep probability=0.2，adaptive timestep sampling 关闭，clip weighting=`uniform_clip`。
- 目标 40,000 learning iterations；周期 PT checkpoint 严格每 1000 iteration，training video 和 ONNX export 关闭。PPO checkpoint 判断是 `next_iteration % 1000 == 0`，因此不会生成 `model_00000.pt` 或非整千周期 checkpoint。
- 正式 source snapshot 为 `src-fa0be8d228aefac39e3f1fbd062c55997d9e3e886c3e88b76ef780ef7f312746`，archive SHA256=`4dd42c39329799d892c4c8892b765e26adf577cdb821c6bbc9b60bc963d3a1cf`；8 节点安装后均通过 manifest byte verification。run contract 位于 `outputs/formal_priv_teacher_rollout30_ws64_20260719_0635/run_contract.json`。
- Rule-90 replay 来自最终 immutable source 的第一条 `scaledown__any_ball_24`；单机器人 H.264、1280x720、50 fps、419 frames、8.38 s，SHA256=`0a8bd73f5a20852654eef8dcebbfdbe0e0c94495d5b813ea9afc9aee6f595ac2`。人工确认 approach/pickup/carry/return 连续、无默认 pose 替换或物体错配；fresh W&B API 已验证 exact summary key `vis/replay` 下只有这一段 MP4，远端 SHA/size 一致。旧的空 history prebind run `71qcdequ` 已验证无训练记录后删除，避免重复媒体存储。
- 正式启动前修复了两个 fail-closed launcher 缺陷：live pure-RL 没有携带 contact prepend compensation；rank-local shard manifest 没有传入 provenance 计算，导致 controller/worker digest 不一致。对应 regression tests 已加入并通过；前者相关组合共 142 tests passed，后者 launcher 定向文件 8 tests passed。
- 两次被拒绝的 prelaunch 均没有产生 simulator training step、checkpoint 或 W&B history：一次是 Tyro 的 logger subcommand 参数顺序冲突，一次是上述 shard provenance mismatch。两处根因修复、重新冻结 source 并重新录制 replay 后才启动正式 run；失败日志保留在 run root 的 `*.prelaunch_*_failure.log` 供审计。
- 2026-07-19 07:19 UTC 验收：8/8 tmux alive，每节点恰好8个 GPU compute apps，所有节点 fatal/Traceback/NCCL/OOM=0；W&B state=`running`、history 已到 step42。step0→42 的 reward 约 `1.066→12.367`，offline-contact reward `0.00874→0.23395`，吞吐 `277,341→366,168 steps/s`；BC/distill loss 恒0、PPO coefficient恒1。训练目录当时约1.2 MiB且尚无 checkpoint，符合1k cadence。
- 通信为 default/small Gloo + hierarchical gradient reduction：节点内 pinned NCCL，节点间8个 leader 走 CPU/Gloo；rank-visible GPU、CPU affinity 与 contiguous minibatches 开启。不得为了例行检查 attach/停止健康 job；健康检查应只读核对 exact session、每节点8 apps、progress 增长、W&B state 和 fatal scan。当前三条 r30 student run `q9qn6xb9/b72gh1wx/5utvhw89` 也不属于本 teacher launch，未被本轮修改。

## rollout30 的 64 卡 state-robust privileged teacher（2026-07-19 UTC，当前运行中）

- 本 run 取代上节 `toxsjobi`：旧 teacher 的8个 exact tmux 已在确认其全部 GPU worker/cwd 后精确停止；三条 r30 student `q9qn6xb9/b72gh1wx/5utvhw89` 未被 attach、停止或修改。新 W&B 是 `zihanw22/carry-any/aofybnlc`，name=`priv_teacher_rollout30_ws64_state_robust_u8init_noreset_20260719_163821`。
- 新 topology 仍为相同的8节点 x 8 L40S=64 ranks，master=`10.99.0.141:31651`，其余节点为 `10.99.0.186/10.99.1.154/10.99.0.167/10.99.0.77/10.99.0.165/10.99.0.227/10.99.0.18`；每节点 session=`priv_teacher_r30_ws64_aofybnlc`，run root=`/home/ubuntu/FAR/holosoma_runs/formal_priv_teacher_rollout30_ws64_state_robust_20260719_163821`。每 rank 1024 env、全局65,536 env，30条 rollout motion 的 uniform inverse-cover objective、24 rollout steps、7 epochs、4 minibatches保持不变。
- 这是 simulation robustness teacher，而不是 sim2real randomization。正式 preset=`g1_29dof_wbt_w_object_teacher_state_robust` 只有 motion-relative reset-state randomization 与 recovery push：joint position `+-0.20 rad`、joint velocity `+-0.35 rad/s`、root position `+-[0.08,0.08,0.025] m`、root rotation `+-[0.15,0.15,0.30] rad`、root linear velocity `+-[0.20,0.20,0.10] m/s`、root angular velocity `+-[0.25,0.25,0.35] rad/s`、object xy `+-0.08 m` 且 z=0；每4--8秒施加不超过 `[0.25,0.25,0.10,0.20,0.20,0.30]` 的 recovery push。明确禁止 action delay、PD/KD、RFI、motor strength、mass/COM/inertia、friction/material、joint sensor bias、camera/depth randomization 混入该 teacher preset。
- 修复了旧 joint reset randomization 实际不生效的问题：`BaseTask` 先执行 randomization reset，随后 `MotionCommand.reset` 又写入 reference motion state，导致旧 `randomize_dof_state` 被覆盖。现在 randomizer 只声明/校验 motion-relative reset contract，最终 joint/root/object 状态和新增 joint-velocity noise 由 `MotionCommand` 在最后一次 state write 中应用；对应 unit tests 覆盖 reset 顺序、范围和 forbidden terms。
- checkpoint 保持每1000 iteration生成，`reset_rollout_at_checkpoint=false`；正常保存不会再重置65,536个环境。iteration 0 的一次 canonical reset 仅用于 fresh training 初始化 rollout stream，不是 checkpoint side effect。actor 从 u8 SHA256=`80cb13e13590239d015ba0a29bdbae901b7785a9789d144745c7bd330059cd68` policy-init，critic/optimizer/RNG/iteration仍 fresh。
- immutable source=`src-a3d2416dc71f953488fa3c447fc110e9880ae1c8536d58ec4951e4f1d0d8738c`，archive SHA256=`73b90f6c880ee8c7b91556168eadc13465d7f18b11eda24d570cf96b8c5c381e`。相关 checkpoint/resume/randomization/Tyro tests 为529 passed，launcher/provenance/preflight tests 为229 passed；exact formal dry-run 和 Tyro CLI preflight 均通过。
- Rule-90 使用最终 source 的第一条 `scaledown__any_ball_24`，严格复现训练的0.2秒/10-frame prepend，而不是旧 capture 默认的2秒。最终唯一 `vis/replay` 是单机器人 H.264、1280x720、50fps、329 frames、6.58s，SHA256=`be87e675d03a3cd249a442e1d1206562c2079368348b6b7ea401d6308563c9a7`；人工 contact-sheet 审核及 W&B 远端字节级复验均通过。
- 2026-07-19 17:02 UTC live acceptance：8/8 session alive，每节点严格8个 compute apps，共64卡；cross-rank provenance 已验证 world_size=64，所有节点均打印 exact motion-relative randomization 参数，fatal/Traceback/NCCL/OOM=0。W&B state=`running` 且已有连续 history step0--10；step10 `Train/mean_reward=12.3345`、offline-contact reward=`0.19237`、throughput=`345,894 steps/s`，`ppo_coeff=1`、BC/distill loss=0，符合 privileged pure-RL teacher。此次为有界启动验收，没有创建周期 monitor。

## rollout30 的 64 卡 strong-push state-robust teacher（2026-07-19 UTC，当前运行中）

- 用户明确要求 teacher 的 push 至少覆盖 student 实际经历的强扰动。`g1_29dof_wbt_w_object_teacher_state_robust` 的 push 已从每 `4--8s`、上限 `[0.25,0.25,0.10,0.20,0.20,0.30]` 提升为与三条 r30 student 完全相同的每 `0.5--2.0s`、6D root velocity delta 上限 `[0.7,0.7,0.25,0.7,0.7,1.0]`。这不是额外拍脑袋扩大范围，而是 exact student disturbance coverage。
- teacher 仍只随机化 motion-relative reset state 与 external push；action delay、PD/KD、RFI、motor strength、joint bias、mass/COM/inertia、friction/material、camera pose、depth noise/dropout 继续明确缺席，避免把 sim2real nuisance 注入 privileged action label。reset-state 范围和 PPO/data/reward/objective 均未改变。
- 新 unit test 直接断言 teacher/student push interval 与 max velocity 完全相等，并继续验证 teacher preset 的 forbidden terms；定向回归 `tests/unit/test_teacher_state_randomization.py` 与 Tyro CLI 共 `14 passed`。最终 immutable source=`src-6a871a6c74d045b8ff1686002f6dcc8eacae438022151013e6630b3227a28eca`，archive SHA256=`fde43506f51194f90880d9e53b908867f4a9516cbd9ae66ff117b087aa44713c`，8个训练节点均通过完整 manifest byte verification。
- 新 Rule-90 replay 由该最终 source 在空闲节点重新录制第一条 `scaledown__any_ball_24`：`randomization:disabled`、0.2s/10-frame prepend、单机器人 H264 1280x720@50fps、329帧/6.58s，MP4 SHA256=`4288840a0784835247516079ca41911673e533c64f6d92723e84c25d990bdc1d`。人工 contact-sheet 审核确认 approach/grasp/carry/return 正常且无灰色默认 pose；新 run 的唯一 `vis/replay` 已完成远端字节级验证。
- 替换前先在旧 `aofybnlc` 仍运行时向全部节点部署并校验新 artifacts；随后逐节点确认64个 GPU PID 全部属于 exact session `priv_teacher_r30_ws64_aofybnlc`、旧 worker 的 W&B ID 为 `aofybnlc`，才停止该组。8节点均验证旧 session=0、GPU app=0；旧 run 已写入 `lifecycle/superseded_by=zihanw22/carry-any/li1gcc1v` 与 `training_target_reached=0`，并用无训练、无 media 的 exact-ID lifecycle finalize 收口为 W&B `finished`。三条受保护 student `q9qn6xb9/b72gh1wx/5utvhw89` 未被触碰。
- 新正式 run：W&B `zihanw22/carry-any/li1gcc1v`，name=`priv_teacher_rollout30_ws64_state_robust_student_push_u8init_noreset_20260719_174027`，每节点 session=`priv_teacher_r30_ws64_li1gcc1v`，master=`10.99.0.141:31661`，其余7节点不变；8x8 L40S=64 ranks、每rank 1024 env、全局65,536 env、fresh critic/optimizer/RNG/iteration，actor-only init 仍为 u8 SHA256 `80cb13e13590239d015ba0a29bdbae901b7785a9789d144745c7bd330059cd68`。
- live W&B config 已读回：randomization tree 只有 motion-relative reset、push state/schedule/application；push 精确为 `[0.7,0.7,0.25,0.7,0.7,1.0] @ [0.5,2.0]s`，`save_interval=1000`、`reset_rollout_at_checkpoint=false`、distill disabled/PPO coefficient=1。2026-07-19 17:59 UTC 8/8 session alive、每节点8 apps、world_size=64 provenance 每节点8/8、fatal/Traceback/NCCL/OOM=0；W&B running 到 step10，throughput=`344,690 steps/s`、mean reward=`10.9658`、mean episode length=`91.14`。这是执行与早期健康验收；最终 robust/action-label 质量仍须用整千 checkpoint 的无扰动与强 push 分层单机器人 eval 决定。

## 8k 新 privileged teacher 的 distill 绑定与三条单机器人 rollout（2026-07-20 UTC）

- 新 distillation-label teacher 固定为 `wandb://zihanw22/carry-any/li1gcc1v/model_08000.pt`，文件8,327,905 bytes，SHA256=`a6093a6fbfb84932517002323fab735aff4759214d3b56acd65e8db934929124`，checkpoint 为 completed iteration 7999 / next iteration 8000。controller 的只读 content-addressed copy 是 `/home/ubuntu/.cache/holosoma/checkpoints/distill_teachers/zihanw22_carry-any_li1gcc1v/by-sha256/a6093a6fbfb84932517002323fab735aff4759214d3b56acd65e8db934929124.pt`。
- input motion 的 generator 仍是 u8 `wandb://zihanw22/carry-any/u8udzw0u/model_05000.pt`、SHA256=`80cb13e13590239d015ba0a29bdbae901b7785a9789d144745c7bd330059cd68`；二者身份不同。下一次 **fresh** distill 必须显式传 `TEACHER_CHECKPOINT=wandb://zihanw22/carry-any/li1gcc1v/model_08000.pt`、`TEACHER_CHECKPOINT_EXPECTED_SHA256=a6093a6fbfb84932517002323fab735aff4759214d3b56acd65e8db934929124`、`MOTION_GENERATOR_TEACHER_EXPECTED_SHA256=80cb13e13590239d015ba0a29bdbae901b7785a9789d144745c7bd330059cd68`、`REQUIRE_MOTION_GENERATOR_TEACHER_MATCH=0`，并同时保留两个角色的 provenance。禁止把已经启动并把 teacher 常驻 GPU 的 `q9qn6xb9/b72gh1wx/5utvhw89` 原地热切换；这会造成不可审计的优化目标断点。
- 三条新 teacher rollout 使用 immutable source `src-6a871a6c74d045b8ff1686002f6dcc8eacae438022151013e6630b3227a28eca`、exact one-motion/object bank、`NUM_ENVS=1`、`HOLOSOMA_EVAL_POLICY=checkpoint_actor`、`randomization:disabled`、zero initial-pose noise、从 timestep zero 开始并关闭 auto/motion-end/clip-end reset。每段原始501帧；交付与旧 3k evaluation 相同地取 renderer frames4..318，共315帧、H.264 640x360@50fps、6.30秒。三份 source manifest、checkpoint、motion、map、URDF 与 mesh 均在运行前后复验；人工六时刻抽帧确认每段恰好一个机器人和正确 ball/bin/barrel。
- 一个通用 recorder 候选虽有501帧，却产生三份逐字节相同的黑底 HUD 视频，未上传。最终改用既有已验证的 headless Viser/Isaac recorder。另发现三个 eval clone 的 `data` 是指向主 data tree 的 symlink，最初三个 local-rank-0 URDF converter 会共享 `converted_rank0` 并撞写临时 USD；正式重录使用显式 cache clean 和不同 `HOLOSOMA_ORIGINAL_LOCAL_RANK`，结束后 eval node `10.99.0.116` 为0 GPU apps，未触碰训练节点/job。
- W&B 只增加中间的 `vis/evaluation_distill_label_teacher`：ball `q9qn6xb9` history step30518，399,771 bytes，SHA256=`b3349bb2589d0657f05365f7bc3fb231e0f8a4363edeb11fb34558e42addcb75`；bin `b72gh1wx` step30668，391,190 bytes，SHA256=`acba4e0f1095569d3fac41ed25387d4697c5cdb72a744461fbfbdfa16bb97ec1`；barrel `5utvhw89` history step30644（media filename step30640，因 primary writer 并发推进），388,415 bytes，SHA256=`954d20817633d79e59c65f737c4bd27a11df6e806ca5ab020a94f5bde2f97e7d`。fresh API 逐文件下载复算 SHA/ffprobe；三条 run 均仍为 `running`，没有新建 eval run。
- 用户要求每个 run 实际只保留 replay / teacher rollout / student rollout 三段。新 media 全部远端验证后，旧 `vis/evaluation_motion_generator_teacher` 的三份文件才事务性退休；它们的 exact bytes 已保存在本轮本地审计目录并写入 manifest。fresh API 证明每个 run 现在恰好3个相关 MP4，并将主 teacher key/role/SHA 与独立 generator SHA 写入 summary。
- 新旧 rollout 明显不相同（ball/bin/barrel 的 aligned-frame luminance SSIM 分别约0.706/0.707/0.675）。在旧 u8 的 authenticated rollout observations 上，新 teacher action L2 mean仍与旧 teacher 同量级（ball8.33 vs8.53、bin7.90 vs7.36、barrel8.51 vs7.74）；但单独 closed-loop policy-I/O run 在状态漂移后记录到主 motion 内 raw action L2 mean约35.83/39.46/38.28、max47.71/60.76/54.00。视频仍稳定可能部分来自 torque clipping，不能把这一点当作 raw MSE label 已合格。下一次 student fresh launch 前必须明确并验证 teacher action acceptance/projection contract；不得仅凭视频直接宣布 action labels scientifically accepted。
- 完整可复现实验、W&B transaction 与 action-label audit 位于未入 Git 的 `outputs/formal_teacher8k_eval3_20260720/delivery_manifest.json`；下一次 distill 的双 teacher 身份绑定位于相邻 `future_distill_binding.json`。

## 三 motion、teacher-only linvel 的 32+32 卡 fresh A/B（2026-07-20 UTC，已停止并切入 distill）

- 用户要求停止旧 teacher/distill job 后重新从零训练；已逐节点按 exact PID/cwd/token 停止旧 `li1gcc1v/q9qn6xb9/b72gh1wx/5utvhw89` 进程并将对应 W&B lifecycle 收口，未触碰无关用户进程。当前只保留下面两组正式 teacher A/B，禁止重复启动或用旧 checkpoint 热切换。
- 两组都只训练 `scaledown__any_barrel_25`、`unscale__any_ball_29`、`unscale__any_bin_29`，各为4节点 x 8 L40S=32 ranks、每rank 2048 env、全局65,536 env。original 组节点=`10.99.0.141/10.99.0.186/10.99.1.154/10.99.0.167`，master=`10.99.0.141:31701`；rollout 组节点=`10.99.0.77/10.99.0.165/10.99.0.227/10.99.0.18`，master=`10.99.0.77:31711`。
- original W&B=`zihanw22/carry-any/rukkpmdv`，name=`teacher_scratch_linvel_original3_ws32_20260720_062837`，session=`teacher_ab3_original_rukkpmdv`；rollout W&B=`zihanw22/carry-any/ppclmh15`，name=`teacher_scratch_linvel_rollout3_ws32_20260720_062837`，session=`teacher_ab3_rollout_ppclmh15`。共同 run root=`/home/ubuntu/FAR/holosoma_runs/formal_teacher_ab3_ws32_scratch_linvel_20260720_062837`。
- original 是 `/nfs/zzzihanw/ds_as_data/debug` 的真实原始 retarget motion，每条320帧；rollout 是 u8 generator `80cb13e...` 的真实 simulator rollout，每条319帧。两边 object-map SHA 均为 `02acd504...c5afc`、contact provenance 均为 `22fd1479...a14be`，rank分配和11/11/10 cover完全一致；共同数组已逐 clip 验证为数值不同，实验变量不是目录别名。
- observation preset=`g1-29dof-wbt-w-object-generalist-teacher-linvel`：teacher actor 在原175维 privileged observation 后追加 exact base-frame `base_lin_vel` 三维，runtime 注册shape=178；student groups 未修改。两组均为 pure PPO，actor/critic/optimizer/RNG/iteration全部 fresh，无 training resume、无 policy init；W&B 的 resume 只连接预先上传 replay 的同一 run ID。
- PPO 固定为24 steps/env、7 epochs、4 minibatches、40,000 updates、MLP `[512,256,128]`、初始 actor/critic LR=`1e-3`、各自 adaptive bounds=`[1e-5,1e-2]`、desired KL=`0.01`、entropy=`0.005`、gamma/lambda=`0.99/0.95`。action scale仍为0.25并按 effort-limit/P-gain缩放；checkpoint每1000 update，`reset_rollout_at_checkpoint=false`，无training video/ONNX。
- teacher randomization 只覆盖 simulation recovery：joint pos/vel `+-0.20 rad/+-0.35 rad/s`，root pos/rot/linvel/angvel分别为 `+-[0.08,0.08,0.025]`、`+-[0.15,0.15,0.30]`、`+-[0.20,0.20,0.10]`、`+-[0.25,0.25,0.35]`，object xy `+-0.08m`，每0.5--2.0秒施加强push `[0.7,0.7,0.25,0.7,0.7,1.0]`。没有action delay、PD/KD/RFI、motor、mass/friction或camera/depth sim2real randomization。
- immutable source=`src-a8879814e247e78b0e686fb8bdf94b985d008b2960aa187b482ef8de7b644399`，archive SHA=`dcbf88cd...35d7de2`；最终 worker SHA=`ceeacc7b...bc9b0`。8节点均通过 source/data/contact/URDF/mesh/runtime/NCCL/GPU/ECC 的 byte-level preflight，exact final CLI 另通过 Tyro scientific parse，episodic `motion_ends` 有效。
- 两份 Rule-90 都来自各自最终训练输入的 canonical 第一条 barrel clip、single robot、randomization disabled、精确0.2秒/10-frame prepend。original=`330 frames/6.60s/SHA 8fc998e7...4d270`；rollout=`329 frames/6.58s/SHA 472bba23...b7df`。contact sheet人工确认 grasp/carry/return 与物体对应正确。训练 history 建立后，同一字节分别写入 original history step97 与 rollout step178；summary 通过 exact live primary W&B service 绑定到这两个 history path，旧 summary-only 副本验证后删除。fresh API 证明每个 run 现在恰好一行 `vis/replay` history、一个 MP4，summary/history/path/SHA/size完全一致。
- 两次安全拒绝都发生在 simulator/model/optimizer/step0 之前：第一次发现绝对URDF依赖未同步，补齐并加入URDF/mesh显式SHA门；第二次发现 actor/critic初始LR同时由env和CLI重复传入，删除重复CLI行，仅保留launcher原生 `ACTOR_LR/CRITIC_LR=1e-3`，上下界仍显式。失败尝试未产生checkpoint或训练history。
- 2026-07-20 06:59 UTC bounded acceptance：original 四节点均到 iteration26，rollout 四节点均到iteration30；8/8 sessions alive、每节点8 compute apps、64卡总计、UECC=0、无exit file。启动期每节点有77条Isaac headless graphics/GPU-foundation `[Error]`（rank-visible下探测其他已占GPU/无viewport），但首个 `HOLOSOMA_PROGRESS` 后新增数为0，PhysX/CUDA/NCCL与训练持续正常；不得把它们隐瞒成零启动错误，也不得误判为training crash。fresh W&B API确认两条均 `running`、`vis/replay`存在、loss/reward/KL有限、`ppo_coeff=1`、BC/distill loss=0、policy-init=false、training-resume=false。本轮不创建周期monitor。
- 2026-07-20 07:17 UTC replay 收口后的最终只读复核：original 四节点一致到iteration291，rollout四节点一致到iteration334；仍为8/8 sessions、每节点8 apps、无exit file且首个progress后hard-error增量为0。两条W&B均保持`running`，history-backed replay单文件契约在多个后续training iteration后仍稳定；promotion没有停止、重启或改变optimizer/model状态。

## matched linvel teacher 的 32+32 卡三 motion distillation（2026-07-20 UTC，当前运行中）

- 两条 teacher A/B 已按 exact session 停止并将 W&B `rukkpmdv/ppclmh15` 收口为 `finished`；不从其尾部临时状态猜 checkpoint。original 固定使用 `rukkpmdv/model_09000.pt`（completed iteration 8999 / next 9000，SHA256=`7a342bb21830024495dfe3f95100b2d8c50de54398ed740cd158983f76a75a07`），rollout 固定使用 `ppclmh15/model_10000.pt`（completed iteration 9999 / next 10000，SHA256=`3700483de167cdac73c4ed495997e7071d7d9fb2b5283ab16194c63e8b561587`）。两份 checkpoint 均通过 strict actor state 与 metadata 门：`178 -> [512,256,128] -> 29`，actor observation 末项是 exact `base_lin_vel`。
- motion/teacher 一一对应：original teacher 只蒸馏 original retarget 的 barrel25/ball29/bin29；rollout teacher 只蒸馏 u8 simulator-rollout 的相同三条 motion。rollout bank 的 motion-generator 仍单独记录为 u8 SHA256=`80cb13e13590239d015ba0a29bdbae901b7785a9789d144745c7bd330059cd68`，它不是当前 label teacher，因此显式 `REQUIRE_MOTION_GENERATOR_TEACHER_MATCH=0` 并同时保存两个角色的 provenance。
- original student W&B=`zihanw22/carry-any/tigpqe1l`，name=`distill_linvel_original3_ws32_t9k_20260720_172020`，session=`distill_ab3_original_tigpqe1l`，节点=`10.99.0.141/10.99.0.186/10.99.1.154/10.99.0.167`，master=`10.99.0.141:31721`。rollout student W&B=`zihanw22/carry-any/ugqc8xn0`，name=`distill_linvel_rollout3_ws32_t10k_20260720_172020`，session=`distill_ab3_rollout_ugqc8xn0`，节点=`10.99.0.77/10.99.0.165/10.99.0.227/10.99.0.18`，master=`10.99.0.77:31731`。每组4节点 x 8 L40S=32 ranks、每rank 64 env、全局2048 env，fresh student，无 training resume/policy init，目标40,000 updates。
- student 并不接收 `base_lin_vel`：actor groups 精确为 `actor_obs_root_contact_aware + actor_obs_drop_button + actor_obs_proprio_with_actions_no_linvel`，合计94维，再由训练期 noisy D435i depth perception 注入。日志中的 `Registering key: actor_obs with shape: 94` 是 student rollout storage，不是把178维 teacher 降成94维；teacher 从 checkpoint actor config 独立构建，strict state-shape load 和 runtime `actor_obs->actor_obs` semantic validation 均已通过。
- 优化目标是 pure supervised DAgger：24 steps/env、1 epoch、16 minibatches，actor-only，BC weight=1，PPO start/target/current coefficient 均为0，fixed actor LR=`1e-3`，student std floor=`0.1`；每rank replay capacity=512、每update取96、current/replay各0.5。teacher mean action 显式 clip 到 `[-8,8]` 后作为 label，避免已发现的 raw label outlier 直接主导 MSE。student 仍使用 contact-aware sampling、offline contact guidance、depth noise/dropout 与 strong push；checkpoint严格每1000 update、`reset_rollout_at_checkpoint=false`、无 step0 checkpoint。
- 两份预训练 Rule-90 继续复用各自最终 bank 的 exact canonical barrel clip：original SHA256=`8fc998e7b9c5769ea68fb8dab2f0f80b8a29c147aacf9590597516851094d270`、rollout SHA256=`472bba238e80c2c841e8ba6b275ea7a390638159e4e19d7f464ed6405b62b7df`。两条新 run 各只有一个 MP4；同一文件分别在 history step25/34 写入唯一 `vis/replay` row，fresh API 已验证 history、summary、文件大小和 SHA 对应。
- launcher 两次 fail-closed 均在 GPU/simulator/optimizer/step0 前停止：先去掉 launcher 已从环境生成的重复 fixed-BC CLI 参数，再显式选择包含 `offline-contact-guidance` term 的 reward preset；最终 exact Tyro dry-run 两臂均 exit0。immutable source=`src-a8879814e247e78b0e686fb8bdf94b985d008b2960aa187b482ef8de7b644399`，worker SHA256=`59cb41fa8be88ccb899b305d26c4a74a0385d85151c1a24f179113ac64f86332`，run contract SHA256=`ec55853de00a09cf31f1cf76f21ee5bf9c8253b119362f93e0e079cbf2def93b`。
- 2026-07-20 17:52 UTC 有界验收：8/8 tmux session alive、每节点严格8个 compute apps，共64卡、UECC=0、无exit file，八份节点日志的 `Traceback/ChildFailedError/CUDA OOM/NCCL error/segfault` 计数全为0。original 节点到iteration138--139，rollout到156--157。fresh W&B 当时两条均为`running`；original `bc/distill=0.30303`、current/replay=`0.31139/0.29468`、reward=`4.73194`，rollout `bc/distill=0.31011`、current/replay=`0.31411/0.30610`、reward约`5.82`；两边 replay buffer=512、actor LR=`1e-3`、PPO loss/coefficient=0。该结果证明训练链实际更新且有限；不把早期 BC 数值当成最终 policy-quality 结论，本轮不创建周期 monitor。
- 可复现审计材料位于未入 Git 的 `outputs/formal_distill_ab3_ws32_linvel_teachers_20260720_172020/`。按用户的 staging 约束，本轮仍只能把 `agent.md` 与 `logs/as_distill_review_fix_20260711.log` 放入 Git index，run contract、worker、manifest、数据、checkpoint 与其他代码改动不得额外暂存。

## rollout 三 motion 的 64 卡 teacher-only fresh PPO（2026-07-20 UTC，误启动后已停止）

- 该 run `zihanw22/carry-any/crjedc0u`、name=`teacher_rl_linvel_rollout3_ws64_fresh_20260720_200722` 是对用户“student policy trained with pure RL”意图的错误实现：它虽然没有 distillation，但使用的是178维 privileged teacher interface，而不是带 depth 的 student interface。用户纠正后已立即停止，禁止恢复或重复启动。
- 拓扑为8节点 x 8 L40S=64 ranks：master=`10.99.0.24:31741`，其余节点=`10.99.0.39/10.99.0.54/10.99.0.61/10.99.0.180/10.99.0.183/10.99.0.201/10.99.0.244`。每rank 1024 env、全局65,536 env；这些是新选节点，当前两条受保护 distill `tigpqe1l/ugqc8xn0` 未被停止、attach 或修改。
- 输入严格是 rollout-distill arm 的同一三条 motion：`scaledown__any_barrel_25`、`unscale__any_ball_29`、`unscale__any_bin_29`。这是 fresh privileged PPO：`distill.enabled=false`、PPO coefficient=1、无 label teacher、无 BC/DAgger、无 policy init、无 training resume，actor/critic/optimizer/RNG/iteration 全从零开始。motion generator 的 u8 SHA `80cb13e...1422` 只作为数据 lineage，不加载进本 policy。
- teacher actor 是包含 base-frame `base_lin_vel` 的178维 observation，MLP=`[512,256,128]`、29D action。PPO 为24 steps/env、7 epochs、4 minibatches、actor/critic初始LR 1e-3及 adaptive `[1e-5,1e-2]`、desired KL 0.01、entropy 0.005、gamma/lambda 0.99/0.95；目标40,000 updates。checkpoint严格每1000 update，`reset_rollout_at_checkpoint=false`，不生成step0 checkpoint、training video或ONNX。
- randomization 仅为 teacher simulation robustness：joint pos/vel `+-0.20/+-0.35`，root pos/rot/linvel/angvel分别 `+-[0.08,0.08,0.025]`、`+-[0.15,0.15,0.30]`、`+-[0.20,0.20,0.10]`、`+-[0.25,0.25,0.35]`，object xy `+-0.08m`，push `[0.7,0.7,0.25,0.7,0.7,1.0] @ 0.5--2.0s`。action delay、PD/motor/mass/friction/camera/depth等sim2real项未混入。
- immutable source=`src-a8879814e247e78b0e686fb8bdf94b985d008b2960aa187b482ef8de7b644399`，最终 worker SHA256=`8ac6fe72cc248ece7b445d800e3096c8152f30c7d57c4b12930264d7d900d1cf`，run-contract SHA256=`897a17d2b590627144b72fa44098ca8ffa797346905d8033264f37cd61b939d4`，ws64 shard manifest SHA256=`311cd1a9507b2ee8a218ad36ecf7284a774bd752bd5af04cb248528cc302c6be`。64 shards 对三条 clip 的 cover 为21/22/21，所有节点 source/runtime/NCCL/data/contact/URDF/mesh byte gate 与最终111参数 Tyro scientific CLI 均通过。
- 启动前 fail-closed 找到并修正 source archive 未解包、contact validator PYTHONPATH、`.201` 缺三组 object mesh、全节点缺 source URDF依赖，以及 logger subcommand顺序；均未进入训练。首次 live startup 又发现 `.244` 没有 canonical `/data/logs_new`；失败发生在 iteration0 前且没有 checkpoint。最终在该节点按其余节点相同 owner/mode 创建目录并通过真实写入探针，attempt3 才被接受；失败 attempt 日志保留供审计。
- Rule-90 是最终 rollout bank 第一条 barrel clip：单机器人 H.264 1280x720@50fps、329帧/6.58s、1,322,033 bytes，SHA256=`472bba238e80c2c841e8ba6b275ea7a390638159e4e19d7f464ed6405b62b7df`。训练 history 建立后，同一字节在 step17 写入唯一 `vis/replay` row；summary 通过 live primary W&B service 绑定到该 history path，远端重下载复算SHA/size后删除 summary-only副本。最终run恰好一个MP4，`replay/history_backed=true`。
- 停止前该误启动 run 曾通过健康验收，但这不改变其 policy interface 错误。纠正时逐节点只停止 exact session `teacher_rl_rollout3_ws64_crjedc0u`，确认8个节点旧 session=0、旧训练进程=0、GPU app=0 后才复用这些卡；W&B lifecycle 已收口为 `finished`，`training_target_reached=0`，`stop_reason=wrong_target_teacher_training_replaced_by_depth_student_pure_rl`。两条受保护 distill `tigpqe1l/ugqc8xn0` 未被 attach、停止或修改。
- 审计材料位于未跟踪目录 `outputs/formal_teacher_rl_rollout3_ws64_fresh_20260720_200722/`。继续遵守用户 Git 范围：index 只能包含 `agent.md` 与 `logs/as_distill_review_fix_20260711.log`。

## rollout 三 motion 的 depth student pure-RL fresh PPO（2026-07-21 UTC，已停止并由 CORL79 替换）

- 正确替代 run 是 `zihanw22/carry-any/rvxqm5y5`，name=`student_pure_rl_depth_rollout3_ws64_fresh_20260720_235642`，每节点 session=`student_pure_rl_depth_ws64_rvxqm5y5`。它是带 D435i depth student interface 的纯 PPO RL：`distill.enabled=false`、`ppo_coeff=1`，没有 label teacher、BC、DAgger、resume 或 policy init；actor/critic/optimizer/RNG/iteration 全部 fresh。provenance 中 `teacher_enabled=false`；`teacher_sha256=bd1ae4...cb2` 是 domain-separated disabled sentinel，不是加载过的 checkpoint。
- 拓扑为8节点 x 8 L40S=64 ranks：master=`10.99.0.24:31751`，其余节点=`10.99.0.39/10.99.0.54/10.99.0.61/10.99.0.180/10.99.0.183/10.99.0.201/10.99.0.244`；每rank 64 env、全局4096 env。输入仍是 rollout-distill arm 的三条 `scaledown__any_barrel_25/unscale__any_ball_29/unscale__any_bin_29`，64 shards cover为21/22/21。
- student actor scalar groups 精确为 `actor_obs_root_contact_aware + actor_obs_drop_button + actor_obs_proprio_with_actions_no_linvel`，共94维，明确没有 `base_lin_vel`；D435i depth 由 `far_tracking_warp` 产生，raw `60x106`、processed `58x87`，CNN编码32维后 concat，所以 actor 第一层是126->512，后续512/256/128，输出29D。critic 继续使用377维 privileged state，depth 不注入 critic。
- depth 训练接口保持 student 的实际随机化：camera pose reset randomization、3--4帧 latency、sensor noise、edge noise/hole（p=0.2）、additive/depth-offset std=0.03m、robot mesh 与 object mesh 均启用；dynamics preset=`g1_29dof_wbt_w_object_with_action_delay`，strong push=`[0.7,0.7,0.25,0.7,0.7,1.0] @ [0.5,2.0]s`。运行日志确认 `sensor_noise=True`、`reset_pose_randomization=True` 且噪声/dropout state 已实际创建。
- PPO 为24 steps/env、7 epochs、4 minibatches，actor/critic LR初始1e-3、adaptive bounds `[1e-5,1e-2]`、desired KL=0.01、entropy=0.005、gamma/lambda=0.99/0.95、initial noise std=1.0，目标40,000 updates。motion sampling 使用 adaptive、固定start-at-zero p=0.2、uniform-T1 window 50/boost 7、contact button=`contact_interval`、carry=`peak_height`、sparse-root=`tracking_error`；episodic `bad_tracking/motion_ends/timeout` termination 有效。
- checkpoint严格每1000 update、`reset_rollout_at_checkpoint=false`、无step0 checkpoint；ONNX启用、training video关闭。immutable source=`src-a8879814e247e78b0e686fb8bdf94b985d008b2960aa187b482ef8de7b644399`、archive SHA=`dcbf88cd...35d7de2`；worker/run-contract/replay-manifest SHA256分别为 `c715d026...7a29c`、`a0dc0f91...945a2`、`e2b75331...3e3e`。
- Rule-90 使用 exact rollout bank 第一条 barrel motion 的已人工审核字节：单机器人 H.264 1280x720@50fps、329帧/6.58s、1,322,033 bytes，SHA256=`472bba238e80c2c841e8ba6b275ea7a390638159e4e19d7f464ed6405b62b7df`。正式 history step84 只有一行 `vis/replay`；summary 通过 live primary service 绑定到该 history path，远端重新下载复算 SHA/size 后删除 summary-only 副本。fresh API 确认 run 恰好一个MP4，`replay/history_backed=true`、`replay/history_step=84`。
- 启动 preflight 首先拒绝了 pure-RL provenance 中不应出现的 distill-only `student_motion_end_mode`，只从 provenance 移除该字段，实际 episodic CLI 未改变；第二次 rank0 只因 replay 的 absolute path 在新 master 不存在而拒绝，复制 exact reviewed bytes并复算SHA后8/8节点全部通过。两次均在 simulator/model/optimizer/history前停止，没有产生训练 step 或 checkpoint。
- 2026-07-21 00:15 UTC 有界验收：8/8 exact sessions alive，每节点恰好8个compute apps，共64卡，UECC=0、无exit file，八份日志的Traceback/ChildFailed/CUDA OOM/NCCL abort/segfault计数均0；rank0已到iteration118。每节点保留77条Isaac headless graphics startup `[Error]`，首个training progress之后新增为0。W&B为`running`、step115，`Perf/total_fps=23,468`、collection/learning=`2.327/1.861s`、mean reward=`2.1769`、actor/critic loss有限、`ppo_coeff=1`，BC/distill/current/replay BC loss全部为0。该结果证明正确 pure-RL depth student 路径正在更新；这是有界验收，不创建周期monitor。
- 审计材料位于未跟踪目录 `outputs/formal_student_pure_rl_depth_rollout3_ws64_fresh_20260720_235642/`。继续遵守用户 Git 范围：index 只能包含 `agent.md` 与 `logs/as_distill_review_fix_20260711.log`。
- 2026-07-21 06:05 UTC 用户要求改用 CORL79 后，逐节点只停止 exact session `student_pure_rl_depth_ws64_rvxqm5y5`；8节点均复核旧 session/进程/GPU app 为0。W&B `rvxqm5y5` 在 last step 4448 写入 `lifecycle/training_target_reached=0` 与 `lifecycle/stop_reason=user_requested_replace_with_corl79_pure_rl_1024_per_gpu` 后收口为 `finished`，禁止恢复。

## CORL79 depth student pure-RL、每卡1024环境（2026-07-21 UTC，当前运行中）

- 正式 run 是 `zihanw22/carry-any/3m8lkcxf`，name=`student_pure_rl_depth_corl79_ws64_e1024_20260721_060531`，每节点 session=`student_pure_rl_depth_corl79_e1024_3m8lkcxf`。节点仍为 `10.99.0.24/.39/.54/.61/.180/.183/.201/.244`，master=`10.99.0.24:31761`；8节点 x 8 L40S=64 ranks，精确 `1024 env/rank/GPU`、全局65,536 env。W&B fresh config 已读回 `training/num_envs_requested_total=65536`、`effective_total=65536`、`num_envs_per_rank=1024`、`world_size=64`，不能把 launcher raw `training.num_envs=1024` 误读为全局环境数。
- 数据是用户所称 CORL80 的实际79条 bank：`carryany_filter_scale_noscale_keep169_20260513_plus_box_teacher_rollout_success155_bcleb5oi58000_final0p5_primitiveproj_solid80_clean_box_bin_barrel_ball`，类别为 box25/bin15/barrel35/ball4。79/79 有 object mapping 与 contact window，wrist target coverage 为 left79/right78。immutable single-slot manifest SHA256=`910399359c1bf8d236ec446667b27902de0037c24c7dfcb40aa70a1bf6d0522d`；ws64 rank manifest SHA256=`2378e682cf8906854050731bc45a340a157c61be9eaeefb24fe31ef14ed4f470`，79条被64 rank 精确覆盖一次，每rank 1--2条。
- 这是从零开始的 noisy-depth student pure PPO，复用上一条正确 pure-RL 的 policy/reward/sampling/randomization 逻辑而只替换数据与环境规模：`distill.enabled=false`、PPO coefficient=1、teacher/BC/DAgger/policy-init/training-resume 全部关闭。actor 仍为 `root_contact_aware + drop_button + proprio_no_linvel` 加 D435i depth CNN，明确不含 `base_lin_vel`；critic 为 privileged state。PPO 是24 steps/env、7 epochs、4 minibatches、actor/critic LR 1e-3 adaptive、目标40,000 updates；checkpoint每1000、`reset_rollout_at_checkpoint=false`、无step0 checkpoint。
- 最终 immutable source=`src-a8879814e247e78b0e686fb8bdf94b985d008b2960aa187b482ef8de7b644399`，archive SHA256=`dcbf88cdc3c274799240b078a52fb7019a3bd0637e2339422fae6287f3357de2`。worker/run-contract/replay-manifest SHA256 分别为 `33d4317b316a72ec03c35bd0a492e1cbbf03d154524373c9f11fc8d27aa33523`、`86011f08b807f3f5c027060c0c3899d02d9922a423ff5dd1f7c49c151d46038c`、`bc8133f25922fb91ef076f64aa8d8b61f7a6571201b137cbcb9238439571e3fd`；未跟踪审计根为 `outputs/formal_student_pure_rl_depth_corl79_ws64_e1024_20260721_060531/`。
- Rule-90 使用最终 CORL79 source 的 canonical 第一条 `box_10`，采用训练一致的0.2s/10-frame prepend与2.0s/100-frame append，关闭 randomization，单机器人 H.264 1280x720@50fps、368帧/7.36秒、1,540,060 bytes，SHA256=`42ed3c334792cecd4edf14ad96dcd67f40c671885f2d32f86d0ce706deca2ea4`。contact sheet 已人工确认 approach/pickup/carry/return 连续且无灰色默认替换。训练 history 建立后，同一字节只在 step21 写入一条 `vis/replay`；fresh remote download SHA/size 完全一致，summary 已绑定 history path，summary-only 副本已删除，run 中只剩一个 replay MP4。
- 2026-07-21 首次有界验收：8/8 exact session alive、每节点8 compute apps、总计64 GPU apps、0 exit file；所有节点进度一致至少到 iteration13。rank log 明确显示每rank固定分配1024 env，模型参数在64 GPU同步校验通过，hierarchical gradient reduce 已启用。每卡峰值显存约27--29 GiB，未 OOM；八节点日志对 Traceback/ChildFailed/CUDA OOM/NCCL abort/error/nonfinite/NaN 的精确扫描均为0。W&B 为 `running`，provenance=`pure_rl`、teacher/resume=false、`Loss/ppo_coeff=1`、`Loss/replay_bc_loss=0`；早期 total FPS 约188,000。此为启动验收，不创建周期 monitor。

## no-contact CORL79 student model-12000 单机器人评估（2026-07-23 UTC）

- 原 run `zihanw22/carry-any/34qv1qqp` 在正式录制前后均为 `running`；最终复核时 history 已到 step12587，最新完整整千 checkpoint 仍是 `model_12000.pt`。exact checkpoint SHA256=`160b15676191d3a14ccb19ab38d408293b233376570a285ec8861efdeace13ed`，completed iteration=11999。没有暂停、停止或修改该训练 job。
- 评估使用 immutable source `src-a8879814e247e78b0e686fb8bdf94b985d008b2960aa187b482ef8de7b644399`、CORL79 canonical `box_10`、单机器人/单环境、seed42、timestep0、zero init-pose noise，关闭 adaptive/T1 sampling、physical randomization 以及 auto/motion-end/clip-end reset。motion/URDF/mesh SHA256 分别为 `61b7a9b...c33c45a`、`a7db8f4e...958de6`、`8cf29017...e7b32`。
- checkpoint 的 noisy-depth perception contract 未被关闭或用 raw CLI 绕过：`far_tracking_warp`、sensor/edge/hole noise active、hole p=0.2、additive/depth-offset std=0.03、latency=[3,4]。单环境录制要求保留训练 producer 的 hole reference batch=1024；`scripts/record_checkpoint_inference.py` 原先漏掉正式 `eval_agent` 已有的 `_bind_training_perception_reference_batch` 调用，导致权重加载前被 semantic identity gate 正确拒绝。录制入口已在 validate 后补上该绑定，harness SHA256=`c642da8f...ec770`；checkpoint/actor/observation/motion 均未改变。
- 非 headless renderer 在无 DISPLAY 时产生501帧黑底片段，人工审核后拒绝上传；按既有正式流程改用 Isaac headless rendering kit 与 tracking camera 重录。两次的501行 trajectory/action/object metrics 逐值完全一致，证明仅改变渲染。最终 MP4 为 H.264 yuv420p、640x360@50fps、501帧/10.02s、543,318 bytes，SHA256=`50a23c4e151a7ae5e6ec698b43c634fc7160e28955ef3c4d8a925193252b48fc`。
- contact sheet 确认机器人与 box 均可见、没有 reset/default-pose substitution。真实结果是不成功：机器人前期接近箱子，但最终未携带箱子；robot/object 净位移约2.283m/0.560m，robot-object 距离由0.768m增至1.973m。该失败行为按原样保留，没有剪掉或美化。
- W&B 原 run 只新增一个 `vis/evaluation_student` history media；fresh API 证明相关 history row=1、远端 MP4=1、run 仍为 `running`。远端重新下载的 size/SHA/ffprobe 与本地完全一致，57个同 row metadata 字段绑定 checkpoint/source/motion/assets/randomization/perception/video 与视觉审核结果。正式本地文件为 `/home/ubuntu/FAR/holosoma_runs/formal_student_34qv1qqp_latest_eval_20260723/videos/final/34qv1qqp_model_12000_student_box_10.mp4`。

## no-contact student model-12000 从已抱起状态初始化评估（2026-07-23 UTC）

- 使用与上一段完全相同的 `34qv1qqp/model_12000.pt`、`box_10`、单环境、seed42、noisy-depth contract、zero pose noise、disabled adaptive/T1/physical randomization 及 disabled reset；唯一任务差异是精确从 runtime timestep 110 初始化，即10帧 static prepend之后的原始 motion 第100帧。该帧 reference object z约0.828m，相比地面约0.110m已经稳定抬起。
- evaluation 原路径会无条件把 reset timestep 设为0；现为 `MotionCommand` 增加默认关闭的 exact forced-reset timestep，并在 `scripts/record_checkpoint_inference.py` 暴露 `--initial-motion-timestep`。强制值在读取 robot/joint/object reference state前应用，越界 fail closed，且不再被 start-at-zero mixture覆盖；训练默认值为 `None`，训练行为不变。recorder metrics同时逐步记录 effective motion timestep/clip index。
- rollout 从 runtime 110 reset；reset warm-up和第一个policy step后首条metric为timestep112，370条完整metrics最终冻结于467。首帧 student root tracking command为非零小误差、drop button为0，证明该实验不是静止 locomotion command，也不是只把物体单独抬高。
- 视觉审核确认机器人从首帧就双臂抱住正确箱子，持续完成主要carry，并在reference lowering/drop阶段放到地面。object/robot净位移分别为2.582m/2.315m；有效 source-reference 区间256帧的aligned object position error mean/median/max为0.1067/0.0999/0.2366m。结果支持“主要瓶颈是从地面建立抓取，而非已经抓住后的carry完全失效”。
- 最终MP4为H.264 yuv420p、640x360@50fps、370帧/7.40秒、481,148 bytes，SHA256=`28b0fdc452749076205aced17073ad7e490652522e36286cb571bb6bed2b6622`，路径为 `/home/ubuntu/FAR/holosoma_runs/formal_student_34qv1qqp_carry_init_eval_20260723/final/34qv1qqp_model_12000_student_box_10_carry_init_t110.mp4`。用户随后明确要求上传；原run只新增一个 `vis/evaluation_student_carry_init` history media（step13350），远端文件 `media/videos/vis/evaluation_student_carry_init_13350_28b0fdc452749076205a.mp4`。fresh API下载复核SHA/size/codec/尺寸/FPS/370帧/7.40秒全部一致，上传前后run均保持`running`，没有覆盖旧 `vis/evaluation_student`、没有创建新run或停止训练job。
- `py_compile` 通过，`tests/unit/test_evaluation_setup_order.py` 为6 passed。录制只使用controller空闲GPU4；退出后该卡回到约905MiB/0%利用率。Git index继续只能包含 `agent.md` 与 `logs/as_distill_review_fix_20260711.log`。

## no-contact student model-12000 已抱起 + persistent forward command（2026-07-23 UTC）

- 使用同一checkpoint、motion、timestep110初始化与noisy-depth/disabled-randomization评估设置，将actor sparse-root输入持续精确覆盖为robot-heading-frame relative pose command `[dx,dy,dyaw]=[0.5m,0,0]`，manual模式下drop button保持0。这不是`vx=0.5m/s`；持续相对位移误差会在机器人前进后仍保持0.5m。recorder新增默认关闭的 `--manual-forward-command-m`，训练与不带该参数的evaluation行为不变。
- policy-I/O debug前40行逐值证明actor最前四维始终精确为`[0.5,0,0,0]`。视频真实结果为初始约2秒抱箱快速前进，随后失稳：robot root低于0.5m发生在2.74s，object低于0.4m发生在2.80s，robot-object距离超过0.7m发生在3.04s；末端机器人与物体距离2.087m。失败没有裁剪。
- 最终MP4为H.264 yuv420p、640x360@50fps、370帧/7.40秒、530,183 bytes，SHA256=`4c56db4f844b5c337372f9ec33fc1a588522d11945f6a5c86e779e7eca54398e`。原run新增唯一key `vis/evaluation_student_carry_init_forward_0p5`；remote filename=`media/videos/vis/evaluation_student_carry_init_forward_0p5_13456_4c56db4f844b5c337372.mp4`，fresh history读取step13464（live primary writer并发推进导致media filename step与最终row step不同）。远端重下载的SHA/size/ffprobe全部一致，run保持`running`。
- 本地文件为 `/home/ubuntu/FAR/holosoma_runs/formal_student_34qv1qqp_forward_0p5_eval_20260723/final/34qv1qqp_model_12000_student_box_10_carry_init_t110_forward_0p5.mp4`。录制只用controller空闲GPU5，结束后回到约905MiB/0%；没有修改训练job。`py_compile`与evaluation setup 6 tests通过，index仍只允许agent/log。

## no-contact student 已抱起 forward-command 四档 sweep（2026-07-23 UTC）

- 用户认为0.5m过大后，使用同一 `34qv1qqp/model_12000`、`box_10`、runtime timestep110、drop=0、noisy-depth与disabled physical randomization，并行录制persistent heading-frame relative-pose `dx=0.05/0.10/0.15/0.20m` 四档；每档370帧/7.40秒。policy-I/O各40行分别只有float32 exact值`0.050000000745/0.100000001490/0.150000005960/0.200000002980`，dy/dyaw/drop全部为0。
- 视觉与trajectory结果：0.05档机器人未摔倒，但约1.68s开始把物体降低到接近地面，末端object z=0.217m、robot-object distance=0.509m；0.10/0.15/0.20三档均全程保持抱持，末端object z分别0.820/0.870/0.915m，robot-object distance分别0.306/0.289/0.320m。robot/object净位移依次为1.976/1.991m、3.829/3.478m、4.645/4.318m、7.286/7.163m。
- 四个视频同一history row step13523写入原run，keys分别为 `vis/evaluation_student_carry_init_forward_0p05/0p10/0p15/0p20`。远端大小/SHA256依次为：445,756 bytes/`22ddbe16fae25239864152438e92b8671632fb4a2e6c053f18b224666c4ea779`；535,229/`1f58e6e224004909b5490baf713f36b43c2075a2278077b9fe23404a615c42f3`；570,194/`e812527da0867c1acaec4d272fd493217c3414e25cb2e22b6bed34b7061e3181`；589,017/`b692dc9b2ad7cd8e26a695fecbbf4da1340f90f763b81fd3266094a1a6e9fd61`。fresh API逐个下载复核H.264/yuv420p/640x360/50fps/370帧/7.40秒和SHA完全一致，run保持`running`。
- 本地审计根分别为 `/home/ubuntu/FAR/holosoma_runs/formal_student_34qv1qqp_forward_0p05_eval_20260723`、`..._0p10_...`、`..._0p15_...`、`..._0p20_...`。四条只使用controller空闲GPU3/4/6/7和独立converter/cache，结束后全部回到约905MiB/0%，未修改训练job；index仍仅agent/log。

## no-contact student 8-object carry-init forward-command sweep（2026-07-23 UTC）

- 为检验结论是否只对 `box_10` 成立，在同一原 run `zihanw22/carry-any/34qv1qqp`、同一 `model_12000.pt`（SHA256=`160b15676191d3a14ccb19ab38d408293b233376570a285ec8861efdeace13ed`）和同一 CORL79 immutable source 上另选4类、每类2个 canonical clip：box=`box_20/box_28`，ball=`noscale__any_ball_3/noscale__any_ball_6`，barrel=`noscale__any_barrel_1/noscale__any_barrel_12`，bin=`noscale__any_bin_32/noscale__any_bin_34`。逐物体选择已稳定抱起的 source/runtime 初始化帧分别为 `93/103, 124/134, 105/115, 108/118, 144/154, 134/144, 134/144, 132/142`；没有把同一个 box 的高度硬套给其他物体。
- 每个物体分别录制 persistent robot-heading-frame relative-pose `dx=0.10/0.15/0.20m`，共24条独立 rollout；这仍不是速度命令。每条 policy-I/O 前40行均验证 dx 为相应 float32 exact 值，dy/dyaw/drop全为0；每条均有370行metrics、370帧、7.40秒，编码统一为H.264/yuv420p、640x360@50fps。checkpoint noisy-depth contract保留，adaptive/T1/physical randomization及自动reset关闭。
- 8份三档 contact sheet 已人工逐帧抽查：所有 clip 均显示其正确物体资产，没有 default-pose/asset substitution；`box_20/box_28` 的灰色外观就是其 canonical box asset，不是fallback。严格末端门（robot z>0.5m、object z>0.5m、robot-object distance<0.7m）通过19/24。`box_20`、`box_28`、`ball_3`、`barrel_1` 三档通过；`ball_6` 的0.10末端物体降低到0.474m但机器人仍站立，0.15/0.20通过；`barrel_12` 的0.10/0.15发生掉落而0.20通过；`bin_32` 的0.10降低到0.398m而0.15/0.20通过；`bin_34` 的0.10约6.6秒机器人摔倒而0.15/0.20通过。因而跨物体趋势是0.15/0.20明显比0.10稳定，但不能声称所有物体都成功。
- 24个视频以 `vis/evaluation_student_object_sweep/<category>/<clip>/forward_0p10|0p15|0p20` 写入原 run 的同一 history row step13830，没有新建eval run。fresh API得到恰好24个media key和24个远端MP4；逐一重下载后，SHA256、size、370帧与7.40秒均和本地完全相同。期望/远端核验清单的共同SHA256为 `1675420e9ded31cd1af1726aa38d61cfee485261f34457877bddb6d07d54f85b`，媒体总量13,133,288 bytes；复核时live run已继续推进到step13854且仍为`running`。
- 本地完整审计根为 `/home/ubuntu/FAR/holosoma_runs/formal_student_34qv1qqp_object_forward_sweep_20260723`，其中 `analysis.json` 保存24条命令、trajectory、结局、video SHA/size/ffprobe，`review/*_compare.png` 保存8份人工审核contact sheet，`remote_verify_13830` 是fresh远端回读。录制只使用controller当时空闲的GPU3--7及独立cache/converter rank；退出后这些卡回到约905MiB/0%，GPU0--2及正在运行的训练进程未被attach、停止或修改。Git index继续只能包含 `agent.md` 与 `logs/as_distill_review_fix_20260711.log`。

## no-contact student model-15000 从地面自主 pickup 评估（2026-07-23 UTC）

- 用户要求检查最新 student 能否自己抱起物体时，原 run `zihanw22/carry-any/34qv1qqp` 的最新完整整千 checkpoint 是 `model_15000.pt`；文件SHA256=`fb7e737163943a7d8f8d019b07ecd883d4a4c9d498397759bee3f8a31d5c6652`，checkpoint内部 `iter=iteration=14999`。使用与model-12000失败对照相同的canonical `box_10`、seed42、单环境、runtime timestep0、zero initial-pose noise、disabled adaptive/T1/physical randomization及disabled automatic/motion-end/clip-end reset。没有设置 `--initial-motion-timestep`，也没有设置 `--manual-forward-command-m`；因此这是从地面建立抓取，不是已抱起初始化或forward-command实验。物体URDF名义质量仍为0.1kg。
- 501步trajectory和人工contact sheet一致证明pickup成功：object z从0.1105m开始，在约2.00s后连续10帧高于初始0.30m，在2.52s后连续高于初始0.50m；峰值z=0.9177m（约3.48s）。物体高于初始0.30m期间robot-object最大距离0.5274m；到step300/约6s时物体已经水平移动2.4755m，机器人最低root z=0.6467m且全程未摔倒。约7--8s后的lowering与落地对应输入motion自身的release段，不是中途抓取失败。与相同设置下model-12000未能建立抓取相比，model-15000在这一个确定性clip上有实质改善；该单clip结果不能外推成79条数据的整体成功率。
- 完整未裁剪视频为H.264/yuv420p、640x360@50fps、501帧/10.02秒、654,740 bytes，SHA256=`3953be9b7e6a7e3722dd6e9659cba6e36c47fab7fa60ccfc246a9569235938f4`。原run新增唯一key `vis/evaluation_student_pickup_from_t0_model_15000`；media上传filename step15379，fresh history row为15386（live writer并发推进），remote重新下载后的SHA/size/ffprobe与本地完全一致。最终复核时run已到step15393且仍为`running`。
- 本地审计根为 `/home/ubuntu/FAR/holosoma_runs/formal_student_34qv1qqp_model15000_pickup_eval_20260723`。录制只使用当时空闲controller GPU3，501步结果落盘后Isaac Sim卡在退出清理，精确终止该独立eval进程；训练进程未被attach或修改，GPU3随后回到约905MiB/0%。Git index继续只能包含 `agent.md` 与 `logs/as_distill_review_fix_20260711.log`。

## CORL79 depth student sparse30 pure-RL、32卡（2026-07-23 UTC，当前运行中）

- 正式 run 是 `zihanw22/carry-any/ptabkuyq`，name=`student_pure_rl_depth_corl79_ws32_e1020_sparse30_no_contact_reward_20260723_215131`，exact session=`student_pure_rl_depth_sparse30_ws32_ptabkuyq`。节点为 `10.99.0.141/10.99.0.186/10.99.1.154/10.99.0.167`，master=`10.99.0.141:31782`；4节点 x 8 L40S=32 ranks，每rank/GPU 1020 env、全局32,640 env。W&B fresh config 已回读 `training/num_envs_per_rank=1020`、`training/num_envs_effective_total=32640`、`training/world_size=32`。
- sparse root command 精确采用 `t1_aligned_segment`：以每条 motion 的 carry-window `t1` 为锚点划分互不重叠的30个 motion-frame segment；每段只在段首计算一次 reference root 从段首到段末的 `[delta x, delta y, delta yaw]`，xy 在段首 root-heading frame 表达，然后30帧 sample-and-hold。`[t1,t2)` 外为0，最后不足30帧的残段为0，yaw zero threshold=0。CORL motion 为50fps，因此一个完整command段名义上是0.6秒；不是每个sim frame重算，也不是把30帧的值平均。
- 数据仍是同一 CORL solid-clean actual79 bank：box25/bin15/barrel35/ball4，single-slot manifest SHA256=`910399359c1bf8d236ec446667b27902de0037c24c7dfcb40aa70a1bf6d0522d`，contact window coverage=79/79。新 ws32 shard 对79条 clip 精确覆盖一次，每rank为2或3条；source digest=`e98a6f9e66c07e9e552388593f98f2c7eeff4bff7bc7f3112612d0fff7c5c4f8`，manifest SHA256=`6861cb9b62547c8d16f68d7759344805b9684a6335fe32923f90f8acd54d799c`。1020 是低于1024的最近合法环境数，同时被2和3整除，因而 fixed scientific env-to-clip distribution 可精确表示。
- 被替换的首次 attempt `zihanw22/carry-any/4sp43zax` 使用1024 env/rank；三-clip rank只能得到342/341/341而不能精确表示1/3分布，因此在iteration0、checkpoint和training history之前 fail closed。exact旧session已逐节点停止并确认GPU app=0；W&B已收口为`finished`，写入 `lifecycle/pre_iteration_failure=1`、`training_target_reached=0`、`stop_reason=ws32_env_count_1024_not_divisible_by_three_clip_shards`、`superseded_by=zihanw22/carry-any/ptabkuyq`，禁止恢复或把旧replay改绑到新run。
- policy 是从零开始的 noisy-depth student pure PPO：scalar actor groups 仍为 `actor_obs_root_contact_aware + actor_obs_drop_button + actor_obs_proprio_with_actions_no_linvel` 共94维，不含`base_lin_vel`；D435i depth CNN输出32维后actor实际第一层126维，MLP=`[512,256,128]`、29D action，critic保持privileged。`distill.enabled=false`、`ppo_coeff=1`，无teacher/BC/DAgger/training resume/policy init；offline-contact-guidance reward weight=0，但真实物体碰撞/接触、contact sidecar、button/carry window和T1 sampling均保留。
- 更新预算与参考 no-contact run `34qv1qqp` 对齐为40,000 updates、24 steps/env、7 epochs、4 minibatches，actor/critic初始LR=`1e-3`及adaptive bounds=`[1e-5,1e-2]`、desired KL=0.01、entropy=0.005、gamma/lambda=0.99/0.95。因world size减半且每rank从1024微调为1020，transition budget ratio为`32640/65536=0.498046875`，不是transition总量相等。checkpoint每1000 update、无step0 checkpoint、`reset_rollout_at_checkpoint=false`、training video关闭；当前 sparse inference contract未实现等价ONNX，所以`export_onnx=false`，只保留PT checkpoint。
- immutable source=`src-a8879814e247e78b0e686fb8bdf94b985d008b2960aa187b482ef8de7b644399`，archive SHA256=`dcbf88cdc3c274799240b078a52fb7019a3bd0637e2339422fae6287f3357de2`。worker/run-contract/replay-manifest SHA256分别为 `3fdd7833d1d7749a084e85f95b4d5e167d9ea331663c357a73db2a8eab5eb592`、`804b56ad8fd740e6282c8fc96159d814f4aa26706ccb18238cfbd578c71f3cec`、`2765b91f3bd4c07883581af4ff1ca01475ebc062f00bf920510636d441b8b1dc`；未跟踪审计根为 `outputs/formal_student_pure_rl_depth_corl79_ws32_e1020_sparse30_no_contact_reward_20260723_215131/`。
- 新 Rule-90 replay 由该run冻结后的最终 CORL79 canonical 第一条 `box_10` 重新录制，不复用失败attempt：单机器人 H.264 1280x720@50fps、368帧/7.36秒、1,533,187 bytes，SHA256=`43de32d93bbcd389000bfab335521ebbc6c33a094ffb2a0fbbcd4081238da761`。8帧contact sheet人工确认approach/pickup/carry/drop/return及物体对应正确。训练history建立后，同一字节只在step20写入一条`vis/replay`，remote重新下载复算SHA/size后才删除summary-only副本；live primary W&B service已把summary稳定绑定到 `media/videos/vis/replay_20_43de32d93bbcd389000b.mp4`，fresh API在后续90+个iteration后仍证明history row=1、MP4=1、run=`running`。
- 2026-07-23 22:19 UTC 有界验收：四节点进度一致到`HOLOSOMA_PROGRESS completed_iteration=111`，4/4 exact sessions alive、每节点8 compute apps、共32卡、0 exit file、UECC=0；所有节点日志对Traceback/ChildFailed/CUDA OOM/fixed-assignment/NCCL fatal/nonfinite的扫描均为0，每卡显存约24.8--28.9GiB。fresh W&B已同步到step110：mean reward=`2.00450`、KL=`0.01130`、total FPS=`124,735`，`ppo_coeff=1`、BC/distill loss=0、全数值有限。原64卡 no-contact run `34qv1qqp` 位于另一组节点，未被attach、停止或修改；本轮不创建周期monitor。

## model-15000 先零指令、稳定抱起后覆盖 forward command（2026-07-23 UTC）

- 用户要求的两阶段 actor 输入已实现为默认关闭的 evaluation-only 路径：初始化后先强制 raw command slice `[dx,dy,dyaw,drop]=[0,0,0,0]`；物体 world-z 相对初始化高度达到0.30m并连续保持10个control step后，再把该 slice 直接替换为 `[0.10|0.15|0.20,0,0,0]`。这里的 dx 是 persistent heading-frame relative-pose displacement，不是速度，也不与 source-motion command 相加；drop 始终被覆盖为0。训练默认不会进入此分支。
- `MotionCommand.configure_manual_forward_after_lift()` 在 reset/warm-up 后冻结物体初始z，并维护per-env连续计数、latched trigger和trigger episode step；`MotionCommand.step()` 在physics后的object snapshot更新完成、actor observation计算之前执行切换，保证触发后的下一次actor forward立即看到目标值。recorder新增 `--manual-forward-after-lift-command-m`、`--manual-forward-after-lift-rel-z-delta-m`（默认0.30）及 `--manual-forward-after-lift-consecutive-steps`（默认10）；它与即时 `--manual-forward-command-m` 互斥，并在整个rollout未触发时fail closed。
- 单元测试覆盖zero hold、阈值抖动清零、连续帧触发、replacement语义、drop=0和非法配置；`python -m py_compile scripts/record_checkpoint_inference.py src/holosoma/holosoma/managers/command/terms/wbt.py`通过，`pytest -q tests/unit/test_manual_forward_after_lift.py tests/unit/test_evaluation_setup_order.py`为11 passed。
- 使用原run `zihanw22/carry-any/34qv1qqp` 的exact `model_15000.pt`（SHA256=`fb7e737163943a7d8f8d019b07ecd883d4a4c9d498397759bee3f8a31d5c6652`）、canonical `box_10`、0.1kg物体、timestep0、seed42、单环境和上一轮相同的disabled adaptive/T1/physical-randomization/reset设置录制三条确定性501-step rollout。三条都在metric step129、sim time约2.65s触发（内部episode step131）；policy-I/O证明step0--129严格为全零，step130起分别只有float32 exact dx=`0.10000000149/0.15000000596/0.20000000298`，dy/dyaw/drop始终为0。
- 人工contact-sheet和trajectory一致：三档都从地面完成pickup，并在剩余约7.4s内持续抱物前进，全程无done；最低robot root z均为0.6453m，触发后最低object z均为0.5774m。末端object相对初始z分别仍高0.5072/0.5628/0.6605m；触发后robot/object水平净位移分别约4.88/5.00m、6.35/6.44m、9.19/9.27m。该结果只证明同一确定性box_10上的两阶段行为，不是CORL79整体成功率。
- 三个完整H.264、640x360@50fps、501帧/10.02秒视频在原run同一history row step15653，keys为 `vis/evaluation_student_zero_then_forward_after_pickup_model_15000_0p10`、`..._0p15`、`..._0p20`。远端size/SHA256依次为689,983/`8627d2a2405bab2c1a8ad30f2d752fc22be16b4d1458c5008be0f2a76836e149`、705,521/`fd2d0acdf949f54f9e69c029fcc35299d97696c40e7a7542eb54e70ba8eaa92f`、740,353/`f9da5b5f93477cee7611883e2261cb0df09a476ea227ac9ab1c406e73d4858fb`。fresh API只找到这3个matching media和1个包含全部3 key的history row；逐个重新下载后SHA/size/codec/尺寸/FPS/501帧/10.02秒与本地完全一致，run仍为`running`。
- 本地审计根为 `/home/ubuntu/FAR/holosoma_runs/formal_student_34qv1qqp_model15000_zero_then_forward_sweep_20260723`，`remote_verify_15653`保存远端回读。录制只使用空闲GPU3/4/5及独立converter rank151/152/153；501行metrics和视频均完整写入后，仅Isaac Sim退出清理触发外层150s timeout（exit124），没有数据或rollout失败。GPU3--5随后回到约905MiB/0%，GPU0--2训练未被attach、暂停或修改。Git index继续只能包含 `agent.md` 与 `logs/as_distill_review_fix_20260711.log`。

## model-15000 其余8个object的zero-then-forward完整sweep（2026-07-24 UTC）

- 沿用上一节完全相同的exact `34qv1qqp/model_15000.pt`、timestep0、seed42、单环境、0.30m/连续10步稳定抬起触发与replacement command语义，对之前选定的8个其他canonical clip全部录制三档，共24条：`box_20/box_28`、`noscale__any_ball_3/6`、`noscale__any_barrel_1/12`、`noscale__any_bin_32/34`。每条先给actor raw slice `[0,0,0,0]`，触发后分别保持 `[0.10|0.15|0.20,0,0,0]`；不是速度、不是source command增量，drop始终为0。
- canary暴露出短motion若沿用训练termination会在motion末端reset并把视频切段；正式24条显式设置 `HOLOSOMA_DISABLE_AUTO_RESET/MOTION_END_RESET/CLIP_END_RESET/BAD_TRACKING_RESET=1`。因此每条都连续501 step/501帧、10.02秒，motion到尾端后冻结且没有episode reset；24份metrics的 `any_done` 全为false。该设置仅存在于独立evaluation进程，不修改训练。
- 24/24都从地面成功达到稳定抬起trigger；逐clip trigger episode step在三档确定性重复中分别为：ball3=112、ball6=141、barrel1=123、barrel12=116、bin32=135、bin34=152、box20=117、box28=126。每条180行policy-I/O都验证trigger前严格全零、trigger后下一次actor forward严格为对应float32 dx，dy/dyaw/drop始终为0；24/24 command gate通过。
- 人工审核8份三档contact sheet确认每条使用正确ball/barrel/bin/box资产、没有default pose/asset替换，机器人均未摔倒。严格末端抱持门 `robot_z>0.5m && object_delta_z>0.3m && robot_object_distance<0.7m` 通过21/24：box20/28、ball3/6、barrel1三档全过；barrel12、bin32、bin34的0.10档后段仍在机器人附近但物体最终高度增量仅0.2034/0.1783/0.1247m，未满足“仍抬高0.30m”，其0.15与0.20档均通过。失败/降低片段未裁剪或隐藏。
- 24个最终H.264/yuv420p、640x360@50fps视频以 `vis/evaluation_student_zero_then_forward_other_objects_model_15000/<category>/<clip>/forward_0p10|0p15|0p20` 写入原run同一history row step16222；没有创建新run。媒体总量16,506,251 bytes。fresh API确认恰好24个matching remote MP4和一个含24 key的history row；逐一下载后24/24均为501帧/10.02秒，远端与本地SHA/size清单共同SHA256=`86b62e5f3d688262252b97c488afa63e72948fb407e876086c71d617d58618b3`。最终复核run仍为`running`并推进到step16245。
- 本地审计根为 `/home/ubuntu/FAR/holosoma_runs/formal_student_34qv1qqp_model15000_other_objects_zero_then_forward_final_20260724`；`analysis.json`保存24条command/trigger/trajectory/outcome/video记录，`review/`保存8份三档视觉审核，`remote_verify_16222/`保存远端回读。本轮首次并行录制为每个task重复创建Isaac/XDG cache并填满controller磁盘；只删除了本轮无效canary及临时cache，改为每GPU复用cache后完成缺失任务，未删除训练checkpoint、数据集或既有媒体。最终cache已清理、policy-I/O已gzip；GPU3--7回到约905MiB/0%，GPU0--2训练未被attach、暂停或修改。Git index继续只能包含 `agent.md` 与 `logs/as_distill_review_fix_20260711.log`。

## model-25000 其余8个object的zero-then-forward对照sweep（2026-07-24 UTC）

- 只把上一节的checkpoint替换为原run `zihanw22/carry-any/34qv1qqp` 的exact `model_25000.pt`，其他实验变量保持不变：SHA256=`145e8d37f503a9069624c085ccb97cf27caa5fad742986cad5e0ffac8dad9b00`，checkpoint内部 `iter=iteration=24999`；同样使用8个canonical clip、每个 `dx=0.10/0.15/0.20m`，共24条单环境、seed42、timestep0、disabled randomization/reset的501-step rollout。actor仍先接收 `[0,0,0,0]`，物体相对初始world-z达到0.30m并连续10个control step后才直接替换为 `[dx,0,0,0]`；不是速度，也不是source command增量。
- 24/24均有501行metrics和180行policy-I/O，24/24达到稳定抬起trigger、command gate通过、`any_done=false`。三档确定性重复的trigger episode step分别为：box20=92、box28=115、ball3=110、ball6=111、barrel1=110、barrel12=120、bin32=134、bin34=128；actor在trigger前严格全零、随后只出现对应float32 dx，dy/dyaw/drop始终为0。
- 8份三档contact sheet已逐一人工检查：正确物体、连续pickup/carry、没有default pose或asset substitution。严格末端门 `robot_z>0.5m && object_delta_z>0.3m && robot_object_distance<0.7m` 为24/24通过；全批次最小末端object height delta为0.3896m、最大末端人-物距离为0.4551m、最低robot root z为0.6326m。相同确定性设置下model-15000为21/24；此前未通过的barrel12/bin32/bin34的0.10档在model-25000分别从末端高度增量0.203/0.178/0.125m提高到0.402/0.394/0.549m。该对照支持这些确定性clip上的改善，不能外推成CORL79整体成功率。
- 24个最终视频均为H.264/yuv420p、640x360@50fps、501帧/10.02秒，总计16,417,039 bytes，写入原run namespace `vis/evaluation_student_zero_then_forward_other_objects_model_25000/<category>/<clip>/forward_0p10|0p15|0p20`。上传端media filename step为26517；live训练并发写入使fresh history最终将唯一含24 key的row排到step26521。fresh API确认run仍为`running`、恰好24个matching MP4和一个完整history row；24个远端文件逐一下载后SHA/size/501帧/10.02秒全部匹配，本地与远端SHA/size manifest共同SHA256=`158b83a65022d6388024a39b0caa0fc464b4841572e5755b966d81c240c38cab`。
- 本地审计根为 `/home/ubuntu/FAR/holosoma_runs/formal_student_34qv1qqp_model25000_other_objects_zero_then_forward_20260724`；`analysis.json`、`review/`、`wandb_remote_verification.json`与`remote_verify_26521/`分别保存trajectory审计、人工抽帧、远端核验清单与远端回读。录制复用每GPU一份临时cache并在结束后只清理该可再生cache，policy-I/O已gzip；GPU3--7回到约905MiB/0%，GPU0--2训练没有被attach、暂停或修改，最终fresh run仍为`running`并已推进到step26566。Git index继续只能包含 `agent.md` 与 `logs/as_distill_review_fix_20260711.log`。

## CORL79 object 偏心（COM/inertia）校准候选（2026-07-24 UTC）

- 用户要求先修正物体偏心。当前正式 student run `zihanw22/carry-any/34qv1qqp` 仍绑定旧的 primitive-projection bank；该 live run、其进程、checkpoint、W&B 和 GPU 均未被修改。物体 URDF 不能在已运行 simulator 中安全热替换，因此本次只生成供下一次 fresh training/evaluation 显式绑定的 immutable candidate，不能宣称旧 run 已获得校准。
- `scripts/build_mesh_physics_object_bank.py` 新增显式 `--mass-mode source_urdf`：visual mesh、collision mesh、mesh origin/scale、motion payload和每条源 URDF 的 nominal mass 保持不变，只根据实际 mesh 体积重新计算 inertial origin 的 COM 与关于该 COM 的惯量张量。原有 `category_priors` 仍是默认模式，避免改变既有调用方；`source_urdf` 与外部 mass-prior 参数同时出现时 fail closed。构建器也会跳过输入 bank 内旧的 derived `_scientific_*`、`_single_slot_*` 和 package manifests，防止新 bank 误继承 zero-COM 的旧 immutable view。
- 新 derived bank 为 `/home/ubuntu/FAR/holosoma/data/ds_as_data/carryany_filter_scale_noscale_keep169_20260513_plus_box_teacher_rollout_success155_bcleb5oi58000_final0p5_primitiveproj_solid80_clean_box_bin_barrel_ball_cominertia_preserve_mass_v1`。最终 exact single-slot view 为其子目录 `_scientific_corl79_single_slot/by-source/cd0b792ed609907c7d76b57ef8d9907cf43ccfc704c6e029c503e098ac93b0f3`；source digest=`b3fba0713952eb7e0baac3e9271dfaf530e997dde89b7276e3176a029d0a531c`，view digest=`cd0b792ed609907c7d76b57ef8d9907cf43ccfc704c6e029c503e098ac93b0f3`，single-slot manifest SHA256=`76d4774c218dfda0d8b95c19caf8a9f4d97b4903c34fb357d33c3584be06dfdd`。
- 全量静态验证为79/79 clip 精确覆盖，79个 motion payload逐个解析回同一源文件，79/79 visual/collision path、origin和scale逐字段不变，nominal mass 唯一值仍为0.1kg。54/79个 mesh 得到非零 COM 修正，最大绝对 xyz 偏移为 `[0.00879963362, 0.0276728878, 0.0115003112]m`；其余25个原本已经以几何体积中心为原点。77个 watertight mesh 使用直接 volume integral，2个非 watertight `bin85` 使用记录在 manifest 中的 convex-hull fallback。79/79惯量矩阵均正定，最小特征值为`0.000223353727124`。
- `scripts/verify_object_bank_geometry.py` 对79条全部通过：missing/invalid、bad extents、size mismatch、box grounding bad 均为0；box首帧底面范围为 `[-0.00016669,+0.00071592]m`。没有 broken symlink，也没有继承旧 derived view。root object-map、mesh-physics manifest、validation report和全部生成URDF集合的SHA256分别为 `6ae1b93c76c9807ade4a5ad71571f30f189101e7c600673cf620b5ce897061d3`、`f5e5590312341af4e8c2cd61dc072e1d09291609f1c98ccd097048bd6f72e93b`、`48fe93c81c4d22ece03e693e9da843d95c15b85f60249be73a2291131998ebf7`、`9f3495e29f1617920fed88cb4783eceb98326b1579c8a83874a54ab2d15bcda6`。
- 新单元测试 `tests/unit/test_mesh_physics_object_bank.py` 用一个人为偏心的 watertight box 验证：质量精确保留、COM/惯量被重新计算、visual/collision完全不变且旧derived views不会传播；结果为1 passed，builder与test的`py_compile`通过。该代码和测试保持未暂存；Git index仍只允许 `agent.md` 与 `logs/as_distill_review_fix_20260711.log`。
- 本轮没有占用训练GPU、没有启动 simulator canary、没有写 W&B。下一次正式训练必须显式绑定上述 exact immutable view，重新生成对应 rank shards 并执行 Rule-90 replay provenance gate；运行时 object mass/friction随机化仍按训练配置执行，本次没有擅自改变其范围。

## CORL79 类别质量 + 耦合惯量随机化候选（2026-07-24 UTC）

- 在上一节 COM-only 校准的基础上，用户进一步指定 nominal object mass：
  `ball=0.5kg`、`bin=1.0kg`、`barrel=1.5kg`、`box=1.0kg`。这些值现在是
  `scripts/build_mesh_physics_object_bank.py` 的 exact category-prior
  `default=min=max`，不是在类别区间内再次随机取 nominal mass。`other` fallback
  仍为 `default=1.0kg,min=0.1kg,max=5.0kg`，但当前 CORL79 中不存在
  `other` clip。
- student object dynamics preset 的 startup scale 改为
  `Uniform(0.33,3.0)`。同一个每环境采样值同时乘 nominal mass 和完整 inertia
  tensor，COM 不变；因此这是固定几何下的 density scale，而不是 mass-only
  perturbation，也没有叠加旧的独立 inertia-axis randomizer。四类最终质量支持为：
  ball=`[0.165,1.5]kg`、bin=`[0.33,3.0]kg`、
  barrel=`[0.495,4.5]kg`、box=`[0.33,3.0]kg`。
- 为避免污染或静默改变上一节 immutable candidate，新建 derived bank：
  `/home/ubuntu/FAR/holosoma/data/ds_as_data/carryany_filter_scale_noscale_keep169_20260513_plus_box_teacher_rollout_success155_bcleb5oi58000_final0p5_primitiveproj_solid80_clean_box_bin_barrel_ball_cominertia_categorymass_v2`。
  正式 single-slot view 为其子目录
  `_scientific_corl79_single_slot/by-source/c9e02244ac1e3c870564f70837a963b03a337430bb1b4a58dc50610868df8027`；
  source digest=`89e1f7d099a741b03a6153654e1821e52deaf023eac8072927e965415b766fac`，
  view digest=`c9e02244ac1e3c870564f70837a963b03a337430bb1b4a58dc50610868df8027`。
  root map、immutable map 和 immutable manifest SHA256 分别为
  `5943790f5ed8153793c9050af552920d058a1489ab8b89186c4a0ec352615da8`、
  `cddc45058f70751d1d4d033b8138ab3a4a33d78bd973ab5c038124f05ca1af9b`
  和 `db4fcf5945cce3c83034791f33c0af4b73352ce2dd60077277baa5ec084047b9`。
- 全量验证覆盖79/79：类别计数为 ball4/barrel35/bin15/box25；79个 root
  motion 和79个 immutable motion 均逐文件匹配源 SHA；79/79
  visual/collision mesh、origin、scale 与源 URDF 一致。54个 mesh 保留非零
  mesh-derived COM，最大绝对 xyz 为
  `[0.00879963362,0.0276728878,0.0115003112]m`；77个 watertight mesh
  使用直接积分，2个 non-watertight mesh 使用声明的 convex-hull fallback。
  79/79 inertia matrix 正定，最小特征值为 `0.00223353727124`；159个
  immutable manifest file record 的 size/SHA 全部匹配，broken symlink=0。
- `scripts/verify_object_bank_geometry.py` 对79条全部通过：
  missing/invalid=0、bad extents=0、object-size mismatch=0、box grounding
  bad=0，box首帧底面范围为 `[-0.00016669,+0.00071592]m`。
  `py_compile` 通过，相关 unit tests 为12 passed，包含 exact category mass
  contract 和 `Uniform(0.33,3.0)` coupled mass/full-inertia contract。
- 该修改只对未来显式使用上述新 immutable view 且选择 object dynamics
  randomization preset 的 fresh process 生效；已启动 simulator 不会被热更新。
  本轮没有 attach/停止/修改正在运行的 training job，没有启动 GPU simulator，
  也没有写 W&B。下一次正式训练仍必须重新生成 rank shards 并执行 Rule-90
  replay gate。代码、测试和数据保持未暂存；Git index 只允许 `agent.md` 与
  `logs/as_distill_review_fix_20260711.log`。

## Object static/dynamic friction 耦合随机化（2026-07-24 UTC）

- object material 不再独立采样 static/dynamic，也不采用
  `make_consistent=True` 的 `dynamic=min(static,dynamic)` 截断。新契约为：
  `static ~ Uniform(0.1,0.7)`、`ratio ~ Uniform(0.7,0.99)`、
  `dynamic=static*ratio`；所以 dynamic 的严格支持为 `[0.07,0.693]`，
  每一个 bucket 都满足 `0.7 <= dynamic/static <= 0.99 < 1`，不会在
  ratio=1 处产生点质量。object restitution 保持独立
  `Uniform(0.0,1.0)`；robot material 路径及其原有独立 static/dynamic
  ranges 未改变。
- 实现继续复用 IsaacLab 的64个 startup material buckets 和原来的三个
  uniform RNG draws：把 IsaacLab 第二列的 sampling range 配置为 ratio
  range，在 material assignment 前原位转换为
  `dynamic=static*ratio`。因此没有为了耦合逻辑额外消耗 RNG，也没有改成
  每个 environment 创建一个 PhysX material；后续仍是每个
  environment/shape 随机选择一个64-bucket material。
- 该路径 fail closed：range 长度、finite、顺序、static/restitution
  下界、ratio `(0,1]` 和 material bucket shape/value 均在写入前验证；
  不允许同时配置 legacy `dynamic_friction_range` 与新
  `dynamic_friction_ratio_range`。写入后重新读取实际 PhysX material
  buffer，逐值验证 observed `dynamic/static` 仍在 `[0.7,0.99]`，否则
  startup 直接报错；正常路径会打印实际 observed ratio min/max。
- `g1_29dof_wbt_randomization_w_object` 和
  `g1_29dof_wbt_randomization_w_object_with_action_delay` 的 config
  回读均精确为 static `[0.1,0.7]`、ratio `[0.7,0.99]`、restitution
  `[0.0,1.0]`，且不再包含 independent dynamic range。`py_compile`、
  whitespace check 和 targeted tests 均通过；相关测试合计18 passed，
  覆盖分布支持/均值、非法range、IsaacLab bucket替换、assignment调用及
  PhysX readback contract。
- 该修改只影响未来 fresh simulator startup；不会热更新已经创建的 PhysX
  materials。本轮没有 attach、暂停、停止或修改任何 live training job，
  没有启动 GPU simulator，也没有写 W&B。代码和测试保持未暂存；Git index
  仍只包含 `agent.md` 与 `logs/as_distill_review_fix_20260711.log`。

## 新 object physics 的 CORL79 depth-student pure-RL、32卡（2026-07-24 UTC，当前运行中）

- 正式 run 为 `zihanw22/carry-any/998gj8c5`，name=`student_pure_rl_depth_corl79_newphysics_ws32_e1020_20260724_231107`，exact tmux session=`hs_998gj8c5`。节点/机器rank为 `10.99.1.134/0`、`10.99.0.116/1`、`10.99.0.117/2`、`10.99.0.77/3`，master=`10.99.1.134:31821`；4 nodes × 8 L40S = 32 ranks，每rank/GPU 1020 env、全局32,640 env。分配前和worker内两次确认四台均8/8卡空闲、UECC=0；其他节点和已有job没有被attach、停止或修改。
- immutable source=`src-f77271b1e9e87aca0140c0f75ed8db892e1f9bab492f77c101295bf520041383`，对应git commit=`6259df92ee6280183a92d0673f9521ec63fff8d5`、source archive SHA256=`961bc0f1b626b81bbf6a030610d994b23cce7cb56063f51b3485af490ad9b80d`。启动前focused suite为263 passed，覆盖mesh physics bank、object material randomization、adaptive sampler/rank shards及train CLI preflight。四节点逐一复核相同source/runtime/NCCL/data hashes、无broken shard symlink、79/79 contact-window coverage和episodic termination CLI。
- 训练输入是CORL79 exact new-physics single-slot view：box25/bin15/barrel35/ball4，source/view digest分别为 `89e1f7d099a741b03a6153654e1821e52deaf023eac8072927e965415b766fac` / `c9e02244ac1e3c870564f70837a963b03a337430bb1b4a58dc50610868df8027`；view manifest SHA256=`db4fcf5945cce3c83034791f33c0af4b73352ce2dd60077277baa5ec084047b9`。ws32 shard对79条clip精确覆盖一次，每rank为2或3条；source digest=`26ee28d11f8cc819d1f21db3902b382271cb76ab297854e31c042a04b9f541f3`、manifest SHA256=`87c51dac3c4662534d0f368cc2abada0b7532ac89d908c8d5aedca26cb85f5c9`。1020同时被2和3整除，fixed scientific env-to-clip分配可精确表示。
- 新物理在live W&B config中已回读生效：object startup同一scale乘mass和完整inertia，范围 `[0.33,3.0]`；static friction=`[0.1,0.7]`，dynamic/static ratio=`[0.7,0.99]`，restitution=`[0,1]`。immutable URDF的nominal mass为ball=0.5kg、bin=1.0kg、barrel=1.5kg、box=1.0kg，并保留mesh-derived COM/inertia。mesh-physics manifest SHA256=`39505d8244d93f4b6edf51b07cea9758c617e1df98dc3299c21e1f5308d8a039`。
- policy保持和 `34qv1qqp` 对照一致的noisy-depth student pure PPO语义：scalar actor为94D、depth CNN output32D、MLP=`[512,256,128]`、29D action；`distill.enabled=false`，无teacher/BC/DAgger/resume/policy-init，offline-contact-guidance reward weight=0。sampling也刻意不混入未批准的新方案：uniform clip + adaptive sampler默认kernel1/lambda0.8/uniform0.1/alpha0.001，20%从timestep0开始，T1 half-width50/density boost7，freeze=0。40,000 updates、24 steps、7 epochs、4 minibatches；checkpoint每1000 update、无step0 checkpoint、正常save不reset rollout，ONNX/training video关闭。
- Rule-90 使用最终view canonical第一条 `box_10`：single robot、randomization disabled、H.264 1280x720@50fps、368帧/7.36秒、1,542,374 bytes，SHA256=`d8eb25f717e1bd811819988979be83b005686110a3b77e0c86cd446731ae7251`。人工contact sheet确认approach/pickup/carry/drop/return连续且无default-pose/错物体。训练history建立后，同一字节只在step16写入一条 `vis/replay`；通过live primary service把summary绑定到history path，fresh API重新下载复算SHA/size后删除summary-only副本。最终run只有1条history row和1个replay MP4。
- run contract/replay manifest/worker SHA256分别为 `8eab9d6382edd357ad67c3e614a0ec24b71b46b296ee745d873b72fa13fc0d34`、`c3697d6bb163d19b0a7b4011969d2c47c363ddd11a49e2801f10f1980adfbdda`、`1dc31439e8701d2ac55d89ed368650f8642143aede024c1ed045cab2dc150b81`；未跟踪审计根为 `/home/ubuntu/FAR/holosoma_runs/formal_student_pure_rl_depth_corl79_newphysics_ws32_e1020_20260724_231107/`。
- 2026-07-24 23:35 UTC bounded acceptance：四节点进度一致到 `HOLOSOMA_PROGRESS completed_iteration=29`，4/4 sessions alive、每节点8 compute apps、共32卡、0 exit file、UECC=0；Traceback/CUDA OOM/ChildFailed/DistBackend/NCCL fatal/nonfinite扫描均为0，每卡显存约24.4--29.0GiB。fresh W&B state=`running`，已同步到training iteration约27：mean reward=`0.66543`、KL=`0.01360`、total FPS=`106,725`、collection/learning time约`5.42/1.92s`，`ppo_coeff=1`且BC/distill loss=0，所有验收数值有限。本轮不创建周期monitor。

## 复现 34qv1qqp 配置的新 physics + naive adaptive CORL79、64卡（2026-07-25 UTC，当前运行中）

- 对照 run `zihanw22/carry-any/34qv1qqp` 已正常完成40,000 updates；本次没有
  resume 或改写旧 run，而是在它释放的同一组8节点上 fresh 启动
  `zihanw22/carry-any/lc4kly4n`，name=
  `student_pure_rl_depth_corl79_newphysics_naive_as_ws64_e1024_20260725_185338`。
  exact tmux session=`hs_lc4kly4n`；节点rank依次为
  `10.99.0.24/.39/.54/.61/.180/.183/.201/.244`，master=
  `10.99.0.24:31831`。8 nodes × 8 L40S=64 ranks，每rank/GPU 1024 env，
  全局65,536 env。分发前、worker内及启动瞬间三次idle gate均通过；
  另一条 `998gj8c5` 及其他job没有被attach、停止或修改。
- policy/optimization 与 `34qv1qqp` 保持同一 depth-student pure-PPO
  scientific contract：94D scalar actor groups 加32D noisy-depth encoder，
  MLP=`[512,256,128]`、29D action、privileged critic；`distill=false`，
  无teacher/BC/DAgger/resume/policy-init。40,000 updates、24 steps/env、
  7 epochs、4 minibatches、actor/critic初始LR 1e-3 adaptive、
  desired KL=0.01、entropy=0.005。checkpoint严格每1000 update，skip step0，
  正常save不reset rollout；为精确复现34q，PT与ONNX均每1000 update生成，
  training video关闭。
- 数据是同一CORL79实际79条原始motion（box25/bin15/barrel35/ball4），但
  显式切到新 object-physics immutable view：
  source/view digest=`89e1f7d099a741b03a6153654e1821e52deaf023eac8072927e965415b766fac` /
  `c9e02244ac1e3c870564f70837a963b03a337430bb1b4a58dc50610868df8027`，
  manifest SHA256=`db4fcf5945cce3c83034791f33c0af4b73352ce2dd60077277baa5ec084047b9`。
  ws64 shards 对79条clip精确覆盖一次，每rank 1或2条，source digest=
  `3512ec2587a0dbd1a3ca0e6de533c098ab25c66ae43d17b6a4f59f90f3456cef`，
  manifest SHA256=`c32f9d8009740a814665916638b2a5bb115732cd7d47fc0bcbc5c17e69ae8e3a`。
  live W&B config已回读 object mass/full-inertia coupled scale=`[0.33,3]`、
  static friction=`[0.1,0.7]`、dynamic/static ratio=`[0.7,0.99]`、
  restitution=`[0,1]`；nominal masses为ball=0.5、bin=1.0、barrel=1.5、
  box=1.0kg并使用mesh-derived COM/inertia。
- “naive adaptive” 的精确定义是跨clip保持 `uniform_clip`，clip内保留
  显式 `P(t=0)=0.2` 以训练自主pickup；其余80%不再叠加T1/contact-window
  density boost，而是
  `0.8 * (0.9 * normalized per-clip failure/exposure + 0.1 * uniform_nonzero)`。
  没有累计失败前nonzero branch退化为uniform。live config精确为
  adaptive=`true`、T1 enabled=`false`、half-width=`0`、density boost=`1`、
  freeze0=`0`；sampler默认kernel/lambda/uniform-floor/EMA为
  `1/0.8/0.1/0.001`。contact sidecar仍服务button/carry window与统计，
  但不再重加权reset概率；offline contact reward weight仍为0。
- immutable source=`src-f77271b1e9e87aca0140c0f75ed8db892e1f9bab492f77c101295bf520041383`
  （git `6259df92ee6280183a92d0673f9521ec63fff8d5`）；正式启动前focused
  suite为263 passed，8/8节点逐一验证相同source/runtime/NCCL、79/79
  contact coverage、object bank/shard hashes、无broken symlink及episodic
  train CLI。run contract/replay manifest/worker SHA256分别为
  `e511711eea5bbdd6a67634f9f44a8d860d6ba4f861d6600efec4ba755e6fe524`、
  `693995f568c024c41005915cd81f01316a394cc52c9a638e7852b2e16e63d168`、
  `93de190e7b6f0b518fbafa03ce17a8209448e4d5321def04e1b8b281f1339415`；
  未跟踪审计根为
  `/home/ubuntu/FAR/holosoma_runs/formal_student_pure_rl_depth_corl79_newphysics_naive_as_ws64_e1024_20260725_185338/`。
- Rule-90 从最终view重新录制canonical第一条 `box_10`：single robot、
  randomization disabled、H.264 1280x720@50fps、368帧/7.36秒、
  1,536,911 bytes，SHA256=
  `5a6255dca7ec0dc338c1958698833e4633256fecc8f74fdefdbd343a1f59eda4`。
  人工抽帧确认approach/pickup/carry/drop/return连续且无default pose/错物体。
  训练history建立后，同一字节只在step9写入一条`vis/replay`；summary经live
  primary service绑定到该history path，fresh API下载复算SHA/size后删除
  summary-only副本。远端最终恰好一条history row和一个MP4。
- 2026-07-25 19:14 UTC bounded acceptance：rank0已到
  `HOLOSOMA_PROGRESS completed_iteration=28`；8/8 sessions alive、每节点
  8 compute apps、共64卡、UECC=0、exit file=0，Traceback/ChildFailed/
  CUDA OOM/NCCL fatal/segfault/killed扫描均为0。fresh W&B为`running`；
  step27 total FPS=`185,128`、collection/learning=`6.069/2.427s`、
  mean reward=`0.67172`、KL=`0.01438`，actor/critic loss有限，
  `ppo_coeff=1`且BC/distill/current/replay BC loss均为0。本轮只做有界
  launch验收，不创建周期monitor。

## Manual forward 世界航向闭环、policy interface 不变（2026-07-25 UTC）

- 用户要求 forward 必须沿前方推进而不是随机器人偏航转圈，同时保持已有
  policy interface。根因确认是旧 manual adapter 永久把
  `[dx,dy,dyaw]=[command,0,0]` 当作当前机器人 heading-frame 指令；机器人
  一旦因接触不对称产生偏航，command frame 会随机器人旋转，而且
  `dyaw=0` 不会恢复触发时的世界航向。
- 工作区实现了 world-heading-locked lookahead adapter。立即 forward 和
  stable-lift 后 forward 都在激活时锁存 robot root world yaw；之后每个
  control step 仍只通过原有三槽输入更新
  `[command*cos(e_yaw), command*sin(e_yaw), e_yaw]`，其中
  `e_yaw=wrap(yaw_anchor-yaw_robot)`。外部参数仍是
  `manual_forward_command_m` / `manual_forward_after_lift_command_m`；
  scalar actor仍为94D、depth encoder仍为32D、29D action，checkpoint与
  ONNX接口均未变化。metrics新增heading anchor、effective actor command、
  along-track/cross-track displacement，便于审计。
- focused回归覆盖command manager、manual forward、evaluation setup、
  observation history和generalist sparse-root launcher，共 `88 passed`；
  compileall与`git diff --check`通过。环境未安装ruff，因此没有伪称完成ruff。
- 真实Isaac Sim A/B复用了已知最严重反例：exact
  `34qv1qqp/model_25000.pt`（SHA256
  `145e8d37f503a9069624c085ccb97cf27caa5fad742986cad5e0ffac8dad9b00`）、
  `noscale__any_barrel_12`、单环境、seed42、randomization disabled、
  stable lift后`command=0.15m`。共同forward窗口step118--325内，旧常量
  body-frame command的root yaw净变化/range为
  `231.84°/231.84°`；新闭环为`12.21°/25.29°`。新逻辑沿锁存前向推进
  `3.409m`、cross-track `-1.163m`，物体末帧z=`0.721m`且
  robot-object XY距离=`0.390m`，确认直到该episode结束仍抱住物体。
  抽帧人工确认approach/pickup/carry连续、无转圈和错物体。
- 诊断根为
  `/home/ubuntu/FAR/holosoma_runs/diagnostic_heading_locked_forward_barrel12_model25000_20260725/`。
  当前代码的episodic motion-end在step326 reset，因此结论严格只使用共同
  第一episode，不把reset后的174帧混入比较。诊断仅占用空闲GPU7；完成后
  GPU3--7均回到约905MiB/0%。没有attach/暂停/修改GPU0--2现有job，没有
  W&B写入、正式训练或push。代码与测试保持未暂存，Git index仍只包含
  `agent.md` 和 `logs/as_distill_review_fix_20260711.log`。

## 34qv1qqp终点checkpoint手动forward对比视频（2026-07-25 UTC）

- 从fresh W&B API确认`zihanw22/carry-any/34qv1qqp`已finished，使用其终点
  `model_40000.pt`，checkpoint内completed iteration=`39999`，本地文件
  SHA256=`ec7a2de8d0f5a9e7a74ac8b1e70d50f11c346de24670a2eaa73bdd560dae52fe`。
  没有用此前仅用于早期诊断的`model_25000.pt`冒充终点模型。
- recorder新增evaluation-only A/B开关
  `manual_forward_heading_lock`。本节录制时用它生成world-heading-lock一侧；
  后续部署语义复核已把当前默认恢复为constant robot-frame
  `[0.15,0,0]`，world-heading lock只允许通过显式
  `--manual-forward-heading-lock`进入诊断。policy输入、checkpoint和action
  interface没有变化。
  新增legacy A/B回归后focused suite为`90 passed`，compileall及
  `git diff --check`通过。
- 两侧使用完全相同的`noscale__any_barrel_12`、单环境、seed42、timestep
  0、randomization disabled、initial-pose noise 0、stable lift 0.30m后
  command 0.15。forward开始前robot/object state逐元素最大差均为0。共同
  第一episode的forward窗口为step117--325，step326为motion-end reset：
  旧逻辑root yaw净变化/range=`202.12°/202.12°`，along/cross=
  `1.359m/1.491m`；新逻辑为`11.85°/25.06°`，along/cross=
  `3.692m/-1.240m`。末帧两侧物体仍抬起，旧/新object z分别为
  `0.701m/0.670m`，robot-object XY距离分别为`0.395m/0.404m`。
- 交付视频为单路H.264、1280x360、50fps、325帧、6.50秒、1,407,777
  bytes；去掉两侧相同的前2帧renderer warm-up噪声后再严格逐帧并排。
  SHA256=`0027b4e2858960063beea1421c2972381afd6d29cf3522c93cb3ce6916311074`。
  人工检查6帧contact sheet确认标签、approach、pickup、carry和左右差异
  可见；完整decode与ffprobe均通过。
- 本地审计根为
  `/home/ubuntu/FAR/holosoma_runs/34qv1qqp_terminal40000_heading_lock_comparison_20260725/`。
  只短暂使用空闲GPU6/7；完成后两卡均回到约905MiB/0%，GPU0--2现有任务
  未被attach、暂停或修改。没有W&B写入。代码与测试保持未暂存，Git index
  仍只包含`agent.md`和`logs/as_distill_review_fix_20260711.log`。

## HIL式task-sparse直线行走reward审计（2026-07-25 UTC）

- 用户澄清目标是让policy通过训练reward学会直线行走，而不是部署时依赖
  simulator world-yaw command adapter。HIL heading task的原式是
  `w_vel*exp(-alpha*(v_target-d_hat^T v)^2) + w_face*(f_hat^T q_hat)`；
  它是每步计算的低维task reward，不是只有成功时非零的时间稀疏reward。
  RSS follow-up所谓sparse goal reward同样包含连续position/orientation error
  和离散reach bonus；这里的“sparse”是相对于全身逐body reference tracking。
- exact `34qv1qqp/model_40000.pt` config证明当前所谓pure RL只表示
  `distill=false`、PPO从零优化；其reward仍启用root/body linear/angular、
  relative pose及object reference tracking，总正权重上限为7，并保留
  bad-tracking termination。command mode还是`tracking_error`，机器人偏航后
  训练输入会自动出现reference-relative `dyaw`修正；这与部署固定
  `[0.15,0,0,0]`不同。因此该run不是HIL式goal/task-sparse训练。
- 正确实验合同应从已学会pickup/carry的tracking checkpoint warm start，
  使用并行tracking/task两类环境：tracking环境保留dense imitation以维持动作
  质量；task环境在稳定抱起后给与部署一致的sample-and-hold raw command，
  移除全身reference tracking，改用目标方向速度、lateral velocity、
  initial-heading facing、cross-track、yaw-rate、object-hold及waypoint/reach
  bonus。actor仍只看现有4槽command和proprio/depth；task indicator只能给
  privileged critic。训练world pose只用于reward，不进入部署接口。
- 必须明确可观测性边界：当前feed-forward actor只有body angular velocity，
  固定`dyaw=0`不包含累计heading error。因此reward能显著抑制自身产生的
  yaw rate和横向漂移，但在任意外部旋转后严格恢复原世界航向并非可保证；
  HIL本身给actor target heading和target facing。若要求这种强恢复，必须在
  同样3个root-command槽中提供动态relative heading error，或采用能积分
  yaw-rate的recurrent/history policy。
- 为避免后续评估再次混淆，`manual_forward_heading_lock`及core
  `heading_lock`当前默认均改为false；raw `[dx,dy,dyaw,drop]`保持不变才是
  deployment-faithful路径，world-heading adapter只作显式opt-in诊断。
  focused回归仍为`90 passed`，compileall和`git diff --check`通过。此审计
  没有启动训练、修改W&B或占用GPU。

## Hybrid Stage-2实现、真实canary与32卡资源门禁（2026-07-26 UTC）

- 已实现独立preset `g1-29dof-wbt-w-object-hybrid-stage2`。actor接口保持
  `94D scalar + 32D depth -> 29D action`，输入group、MLP
  `512/256/128`、normalizer和`model_40000.pt`完全兼容；task indicator只
  进入privileged critic。50% tracking env始终使用dense reference tracking；
  50% task env在pickup前同样tracking，pickup后actor只收到deployment-faithful
  raw `[0.15,0,0,0]`，dense tracking切换为forward/lateral/heading/
  cross-track/yaw-rate/object-pose-hold低维task reward。
- pickup时锁存robot world pose和robot-relative object pose。task phase允许
  有意的平面reference偏离；仍以gravity/tilt failure和robot-relative object
  position error大于0.35m终止。普通action/contact/limits safety penalty保留。
- 首轮1024-env full-bank canary被既有科学门禁正确拒绝：79不能整除1024。
  试图把env数改为1032又被policy-init门禁拒绝，因为终点checkpoint绑定的
  depth-hole reference batch为1024。最终扩展immutable rank-shard builder，
  新增显式`--environments-per-rank`约束；32-rank exact-once分配成为
  `8 ranks x 4 clips + 23 x 2 + 1 x 1 = 79`，所有local clip count都整除1024。
  manifest SHA256=`2eb12afba67bea4858db5bb1fb819aaa919ac2f8066bf593d9a28057045cdfd9`，
  source digest=`cf926c0ec4c3df2dc203d4ed0e9b569c6cf921b1f591d0ec4c018e121628fc54`。
- 修复了另一处隐蔽偏置：原先global alternating task mask在2-clip
  round-robin下会把一条clip全部分到task、另一条全部分到tracking。现在在
  fixed clip内独立做deterministic exact split；不消费训练RNG。rank-shard、
  provenance和hybrid定向回归共`209 passed`，额外hybrid/manual-forward
  focused为`14 passed`。
- 最终immutable source为
  `src-40c3f3f9697c21e331d7d6a067e76a4c967dd627b0cdd21b38e7999ab6871c7f`，
  archive SHA256=`a7d7735d806068dd84f8ca00cba32f74fb5b5e56aa72102ae1a28d596b03f61e`。
  在完全空闲`10.99.0.97`上完成真实IsaacSim canary：1024 env、rank0的4条
  clip各256 env、每clip内50/50 hybrid split，actor-only加载exact
  `34qv1qqp/model_40000.pt` SHA256
  `ec7a2de8d0f5a9e7a74ac8b1e70d50f11c346de24670a2eaa73bdd560dae52fe`，
  fresh critic/optimizers/iteration。两次PPO update均完成，约
  `4612 -> 5795 steps/s`，所有七项task reward出现非零统计，正常发布
  `model_00002.pt`和`HOLOSOMA_RUN_COMPLETE`，随后GPU进程清零。
- 预启动合同位于
  `/home/ubuntu/FAR/holosoma_runs/formal_student_hybrid_stage2_corl79_ws32_e1024_init34q40000_20260726_061000/experiment_contract_prelaunch.json`，
  SHA256=`412ccc237ded0b232d05a2a57ea335a60100b3af986c846fa68f642b4e11c00f`。
  最新可访问节点复核只有`10.99.0.97`这一组完整8卡空闲；其余8卡节点均有
  compute apps。遵守“不要动已经跑上的”，没有停止、attach、迁移或超卖
  任何现有job。因此正式4 nodes x 8 GPUs尚未创建W&B run、尚未做Rule-90
  replay、尚未启动worker，避免产生空run或部分DDP。代码/测试保持未暂存，
  Git index仍严格只有`agent.md`和本log。

## Hybrid Stage-2正式32卡启动交接（2026-07-26 21:59 UTC）

- 后续资源复核发现四组完整空闲且UECC为0的8xL40S节点：
  rank0/master `10.99.1.21`、rank1 `10.99.0.141`、rank2
  `10.99.0.186`、rank3 `10.99.1.122`。启动前分别完成一次完整worker
  dry-run和一次启动瞬间空闲门禁；四台均为8/8 GPU空闲、端口32131空闲。
  没有停止、attach、迁移或超卖任何既有job。
- 正式W&B run为
  `zihanw22/carry-any/wgcsw25u`：
  <https://wandb.ai/zihanw22/carry-any/runs/wgcsw25u>。run name为
  `student_hybrid_stage2_corl79_init34q40000_ws32_e1024_20260726_214121`。
  Rule-90在worker启动前已完成：canonical首条`box_10` reference replay为
  单机器人H.264 1280x720、50fps、368帧、7.36秒、1,543,015 bytes；
  视频SHA256=
  `892a480a96dc7bac4305c179ac7173dd5505cc7d65ad10da2e4c730abc4c213e`。
  完整decode、ffprobe和人工多帧检查通过，W&B fresh API复核
  `vis/replay`恰好指向这一份MP4。replay manifest SHA256=
  `06fa6dfe298bff71e757e6976e3c44f2150ca82ea9d1f5a0bcd5bea3fa0fbeb8`。
- immutable正式合同改为launch root下`run_contract.json`，SHA256=
  `4b40e0e773f43e43ec430327ef83e9cdb163f147c467fa4f09279f477e219e21`；
  worker SHA256=
  `20681487cb19192c0007f678863558f4b783c1e38b7e388377109df6f06282a6`。
  四台均逐文件验证source、runtime、NCCL、motion/object/contact、32-rank
  shard、contract、replay manifest和policy-init checkpoint哈希，且完整
  CLI preflight通过。
- 正式配置为4 nodes x 8 ranks、每rank 1024 env、global 32768 env；
  CORL79按immutable compatible shard exact-once分配。PPO为24 steps/env、
  7 epochs、4 minibatches、40000 updates，checkpoint每1000 updates，
  `reset_rollout_at_checkpoint=False`。distillation关闭且不加载teacher。
  actor-only从exact `34qv1qqp/model_40000.pt`初始化；正式日志确认忽略旧
  iteration=39999、critic、optimizers、critic normalizers和env state，
  actor normalizer未恢复，fresh iteration从0开始。
- 运行时每个rank确认`task_envs=512/1024`、`tracking_envs=512`、
  `stratified_by_fixed_clip=True`、raw forward command=0.15m。启动验收到
  completed iteration 5；总吞吐由iteration 0的61,724 steps/s升至
  iteration 2的91,131 steps/s、iteration 4的108,955 steps/s。W&B fresh
  API在step2看到全部七项hybrid task reward为非零，并确认world size=32、
  每rank 1024 env、global 32768 env及`vis/replay`仍存在。四台各8个GPU
  compute app持续存活，tmux均为up，无Traceback/OOM/ChildFailed/NCCL
  fatal。正式run已成功进入长期训练，不再做持续读取式监控。

## 原始30条real-mesh bank物理修正（2026-07-27 UTC）

- “原始30条”固定绑定到
  `/home/ubuntu/FAR/holosoma/data/ds_as_data/debug39_realmesh_rollout_u8udzw0u_model05000_retake4gpu_20260706_0205_target`；
  source object-map SHA256为
  `9af9da48f24d0c0f076de7147d80f21786e13bbeabc58b6a138d56f979f30f51`。
  它是30条经real-mesh rollout筛选通过的原始reference/retarget motion，
  source map仍指向`*_original.npz` lineage；不是后来
  `prism_debug30_convexhull_allmesh`的convex-hull teacher bank，也不是
  student/teacher policy rollout motion。旧bank未被覆盖或修改。
- 新的可编辑derivative为
  `/home/ubuntu/FAR/holosoma/data/ds_as_data/debug30_original_realmesh_rollout_u8udzw0u_model05000_cominertia_categorymass_v2`。
  30份motion与source逐文件SHA256一致，所有NPZ key/array也逐项一致；
  clip集合严格为ball 5、barrel 1、bin 24。30份visual/collision均继续引用
  原始`object_mesh_yup.obj`，没有把几何替换成convex hull。
- 每份URDF的inertial origin和完整inertia tensor由实际scaled collision mesh
  重新积分，所有30份COM均非零，COM norm范围
  `[0.0041181771, 0.0315724909] m`，最小inertia eigenvalue为
  `0.00578296018 kg m^2`且30/30正定。28份mesh直接用封闭体积积分；
  `scaledown_any_bin_27`和`unscale_any_bin_27`非watertight，只在估算
  COM/inertia时使用记录在案的convex-hull fallback，实际render/collision
  mesh仍是原real mesh。
- nominal mass按当前类别合同写入：ball=`0.5 kg`（5份）、
  bin=`1.0 kg`（24份）、barrel=`1.5 kg`（1份；box若加入为`1.0 kg`）。
  `_clip_object_urdf_map.json`、`_mesh_physics_manifest.json`和
  `_mesh_physics_report.csv` SHA256依次为
  `fad0b688ad65e8c811cc961938e26a566a8cf48b6c28d7fbe930ca88f002479a`、
  `f1dec081766327fbd68694d821a9a201f46ee4e670a550685c180e0c21ddc61b`、
  `7df22426db8d6b695b99bc863fecd0e6ee4b484f24cd57cf49c3cc7abd9ef681`。
- 正式只读single-slot view为
  `/home/ubuntu/FAR/holosoma/data/ds_as_data/debug30_original_realmesh_cominertia_categorymass_v2_scientific_single_slot__src_1f3966245545689a2e14909dda31d6673790bdf888168362449a0ac1902f3510`；
  source digest=
  `1f3966245545689a2e14909dda31d6673790bdf888168362449a0ac1902f3510`，
  manifest SHA256=
  `e2c53df92e5af02048de9b0e062ea6dde03db9b1ee493dfc42591da606dd53b7`。
  view内30份motion为regular read-only副本且与source/derivative三方SHA256
  一致；0 symlink、0 writable file；object/contact闭包完整，
  `contact_export_from_teacher_realmesh_rollout`含1140个被manifest哈希的
  regular files；transition语义保持`global_multi_clip_runtime`、source
  clip count 30。
- `verify_object_bank_geometry.py`完成30/30训练路径检查：
  missing=0、bad extents=0、object-size mismatch=0。immutable gate首次发现
  mesh-physics builder把`realmesh_rollout_manifest.json`也做成软链接；
  builder现对这份provenance snapshot强制普通文件复制，bulk motion/mesh
  仍可symlink。相关mesh-physics加immutable回归共`25 passed`。
- URDF只固化确定的nominal mass/COM/inertia；训练时的不确定性继续由当前
  `object_state_dr_at_setup`显式采样：static friction
  `Uniform(0.1,0.7)`，dynamic/static ratio `Uniform(0.7,0.99)`，
  restitution `Uniform(0,1)`，mass与inertia共享同一个
  `Uniform(0.33,3.0)` scale。这样不会把随机分布错误写成单个URDF常数。
  本次没有启动或修改训练、没有占GPU、没有写W&B。builder与测试修改保持
  未暂存；Git index仍只允许`agent.md`和本log。

## 原始30条作为reference的64卡pure-RL正式训练（2026-07-27 21:55 UTC）

- 用户澄清后的数据边界已固化：既有CORL79数据、实验目录和W&B身份全部
  保留且未停止、覆盖、合并或改写；本实验是另一条fresh pure-PPO run。
  新的30条只作为environment中的reference motion，不是teacher rollout，
  不提供action label，也不启用distillation、BC、DAgger、teacher、
  policy init或training resume。
- 正式W&B run为`zihanw22/carry-any/6urn4jvc`：
  <https://wandb.ai/zihanw22/carry-any/runs/6urn4jvc>；run name为
  `student_pure_rl_depth_original30_reference_newphysics_naive_as_ws64_e1024_20260727_213011`。
  资源变化前产生的`7cyvqnws`只有Rule-90 prebind、0条training history；
  在确认没有训练后已删除，避免留下空垃圾run。
- 最终拓扑为8 nodes × 8 L40S = 64 ranks：rank0/master
  `10.99.1.134:31841`，其余node rank依次为
  `10.99.0.116`、`10.99.0.117`、`10.99.0.167`、`10.99.1.154`、
  `10.99.0.77`、`10.99.0.97`、`10.99.1.69`；每rank/GPU 1024 env，
  global 65,536 env，session统一为`hs_6urn4jvc`。第一次选出的
  `10.99.0.18/.165/.227`在preflight前被pointworld评测占用，idle gate
  fail-closed拒绝启动；没有停止、attach或超卖这些任务。最终8台在完整
  dry-run后又通过启动瞬间的8卡空闲、UECC=0、合同哈希和端口门禁。
- `10.99.0.77`当时root盘满但`/data`有6.5 TB；只用`pip cache purge`
  删除11,498.9 MB可再生下载缓存，没有删除checkpoint、run或源码。
  30条数据落在`/data`并通过原绝对路径symlink访问。对
  `10.99.0.77/.97`和`10.99.1.69`补齐`/data/nfs -> /nfs`，保证移动到
  data volume后URDF相对mesh URI仍解析到同一只读NFS资产。所有相关
  motion/object/contact/source/runtime/NCCL哈希在worker内重新验证。
- 输入固定为immutable view
  `debug30_original_realmesh_cominertia_categorymass_v2_scientific_single_slot__src_1f3966245545689a2e14909dda31d6673790bdf888168362449a0ac1902f3510`，
  30条类别为ball 5、barrel 1、bin 24。由于64 ranks > 30 clips，
  shard为每rank一条：26条覆盖2个rank，4条覆盖3个rank；inverse-cover
  distributed loss weight令每条clip的全局总权重均为64/30，保持
  uniform-clip objective。rank-shard manifest SHA256=
  `e1952bb3fc05b8690c41f59a31dd303225d06fe1a5473a87e4d65d7eed99bb0c`，
  source digest=
  `bdd1011b2704b8e83134070e6d2b9d605496e7c8005a83e9fee6a2e0b9e602ab`。
- Rule-90在训练启动前对canonical第一条
  `scaledown__any_ball_24`做了policy-free reference replay：
  randomization disabled、单机器人、timestep 0、reset noise 0。
  成片为H.264 1280×720、50 fps、329帧、6.58秒、1,326,443 bytes，
  视频SHA256=
  `dfa432c01bce6f40afe755e764a360b5ab599a0c57ce1967ec6aa28d1e2ced35`。
  ffprobe、完整decode和人工8帧检查通过。训练history出现后，视频已在
  history step 15写为唯一`vis/replay`；primary summary已绑定到同一
  history media，prebind duplicate已删除。fresh API确认恰好1个replay
  history row和1个MP4。replay manifest SHA256=
  `76b487dd9e5921ac2134be8684ef83c4e7f91e9c28363c1ebca98836feb895eb`。
- immutable run contract SHA256=
  `63a9ca99a879b4862a7377fb477e97b422fcfe0f6d890491641f7c5f1e0b9ed9`；
  worker SHA256=
  `ab9ee6f8f662c503a297ca79c008ddc82f1e6c245641a812ed6bbead9d3c8d84`；
  sealed source仍为
  `src-f77271b1e9e87aca0140c0f75ed8db892e1f9bab492f77c101295bf520041383`。
- 训练语义与CORL79 pure-RL baseline保持一致：depth student actor
  scalar 94D + 32D depth encoding、MLP `[512,256,128]`、29D action、
  privileged critic；24 rollout steps、7 epochs、4 minibatches、
  adaptive LR 1e-3、desired KL 0.01、entropy 0.005、40,000 updates。
  naive adaptive sampling为uniform clip、显式`P(t=0)=20%`、无T1 boost；
  offline contact guidance reward weight=0。物理随机为mass+inertia共享
  `Uniform(0.33,3.0)`、static friction`Uniform(0.1,0.7)`、
  dynamic/static ratio`Uniform(0.7,0.99)`、restitution`Uniform(0,1)`。
  checkpoint PT+ONNX每1000 updates，跳过step 0且
  `reset_rollout_at_checkpoint=False`。
- 有界启动验收完成到iteration 22：iteration 0为116,605 steps/s，
  iteration 5为151,968 steps/s，iteration 20为167,245 steps/s，
  iteration 22为154,052 steps/s；mean reward由-0.25增至0.51。
  8台均session up、每台8个GPU compute app、无exit，且
  Traceback/OOM/NCCL/Gloo/segfault fatal均为0。fresh W&B API读回
  `training_regime=pure_rl`，teacher/policy-init/training-resume均false，
  world size 64、每rank 1024 env、global 65,536 env，所有最新数值有限。
  正式run已进入长期训练；验收结束后未建立常驻读取式监控。

## 原始30条reference的16卡teacher正式训练（2026-07-27 23:15 UTC）

- 按用户最终口径启动了新的fresh teacher PPO run：
  `zihanw22/carry-any/mlgjus6q`
  （<https://wandb.ai/zihanw22/carry-any/runs/mlgjus6q>），run name为
  `teacher_priv_linvel_original30_reference_newphysics_ws16_e4096_20260727_223824`。
  这是teacher policy本身从零训练；没有external teacher、
  distillation、BC/DAgger、policy init或training resume。
- 拓扑为2 nodes × 8 L40S = 16 ranks：rank0/master
  `10.99.0.18:31981`，node rank1为`10.99.0.227`，统一tmux session
  `hs_mlgjus6q`；每rank/GPU 4096 env，global 65,536 env。
  `scene.env_spacing=5.0 m`用于避免超大并行场景中跨environment的无意义
  broad-phase/contact pair膨胀，不改变单个environment内的任务物理。
- 数据固定为immutable single-slot view
  `debug30_original_realmesh_cominertia_categorymass_v2_scientific_single_slot__src_1f3966245545689a2e14909dda31d6673790bdf888168362449a0ac1902f3510`，
  共30条（ball 5、bin 24、barrel 1）。view/source digest=
  `1f3966245545689a2e14909dda31d6673790bdf888168362449a0ac1902f3510`；
  object-map SHA256=
  `412e86f38c3376b456d5f97e58876f6189eff1345da92e9903b9752030c9e742`。
  16个rank shard为rank 0--13各2条、rank 14--15各1条，并使用
  distributed loss weight保持全局uniform-clip objective；shard manifest
  SHA256=`fc68090f7052cf7acfb8aa58e585119d42d73abe8af6d97690e797ceab864f83`。
- 用户在正式启动前明确取消contact reward。最终生效配置中
  `offline_contact_guidance.weight=0`，且arms、palms、torso和左右wrist
  的全部正向body-contact reward weight也均为0。物理接触仍正常模拟，
  contact指标仍仅作诊断；安全惩罚`undesired_contacts=-0.5`和
  `lower_body_undesired_contacts=-0.1`保留，不属于正向contact reward。
- teacher随机化使用`g1_29dof_wbt_w_object_teacher_state_robust`并由合同
  证明逐项覆盖student分布：机器人材质static `[0.25,1.7]`、dynamic
  `[0.25,1.3]`、restitution `[0,0.6]`；base COM
  x/y/z为`±0.065/±0.095/±0.12 m`；link mass scale `[0.85,1.25]`、
  base added mass `[-1.5,3.5] kg`；物体static friction `[0.08,0.8]`、
  dynamic/static ratio `[0.65,0.99]`、restitution `[0,1]`、
  mass与完整inertia共享scale `[0.30,3.25]`。action delay、PD gain、
  torque RFI和joint-zero bias在本teacher与目标student中均关闭。
  randomization contract SHA256=
  `908c0db6a8fcf3c644e8a7fa7370af635c1da056e468b9805ac2dd1f4c1ffeaa`。
- PPO合同为40,000 updates、24 steps/env、7 epochs、4 minibatches，
  actor/critic MLP均为`[512,256,128]`，actor 178D（含精确robot-base-frame
  base linear velocity）、privileged critic 310D；adaptive LR 1e-3、
  desired KL 0.01、entropy 0.005、checkpoint每1000 updates。
  immutable run contract SHA256=
  `c69eb3707007da061fe91b431b482f421a657efdac8760f39afb383b61fe0d02`；
  sealed source为
  `src-a49d2e042eb15bd580dd3e2ca256668370b97646423d434a1968d131d8e03f39`。
- 正式启动前先通过单卡4096-env两次更新canary，再通过8卡、每卡4096-env、
  contact reward=0的两次更新canary。早先一个误开single-node
  hierarchical reduction的canary和一个尚未应用用户reward覆盖的canary
  均在正式训练前停止并保留失败/作废日志，没有混入正式run。
- Rule-90使用canonical第一条`scaledown__any_ball_24`完成policy-free
  reference replay：randomization disabled、timestep 0、initial-pose
  noise 0、单机器人和正确real-mesh物体。H.264 1280×720、50 fps、
  329帧、6.58秒、1,324,959 bytes；视频SHA256=
  `7eb72ee002fbd5ea4a8ea74d720654a8c8f12bf0a14cf6a90cdbda62d9838baf`，
  replay manifest SHA256=
  `1e750d3f0eaf2bf0bb4a69bb2c5f8bdd5c2b54de7e885528375e5d7b48de1e3b`。
  ffprobe、完整decode和人工8帧检查通过；fresh W&B API确认history step
  25恰好1个`vis/replay` row和1个MP4，primary summary绑定同一媒体，
  下载后SHA256一致，prebind duplicate已删除。
- 有界启动验收到`completed_iteration=47`：最近一次iteration计算吞吐
  234,810 steps/s、mean reward 0.86、distributed loss weight sum 16。
  两台均session up、各8个GPU compute app、无exit、UECC sum=0。
  逐日志排除Isaac Sim启动时每rank一次的
  `PhysXFoundation: Couldn't get driver version`已知warning后，
  Traceback、OOM、NCCL/Gloo fatal、segfault和non-finite均为0。
  fresh W&B API读回run为`running`、16 ranks、4096 env/rank、
  global 65,536 env，所有正向contact reward weight为0。正式run已进入
  长期训练；验收后未建立常驻读取式监控。
- 2026-07-29 19:24 UTC 用户明确要求暂停该run。停止前两节点一致推进到
  `completed_iteration=31388`；只向exact session `hs_mlgjus6q`发送中断，
  随后确认`10.99.0.18/.227`上该session、绑定run ID/launch token的进程、
  GPU compute app及31981 listener均为0，未触碰`6urn4jvc`或其他job。
  可恢复锚点固定为W&B与本地字节一致的`model_31000.pt`：size=7,413,361，
  SHA256=`455324485299286f8d6ac55a63991cc997d658aececb1f05c2752c6d5927393c`，
  `saved_iteration=30999/next_iteration=31000`。actor、critic、两optimizer、
  normalizer及16-rank env/RNG state均通过finite/deep-load检查；checkpoint
  的recovery rollout contract也绑定`next_iteration=31000`。31,000之后的
  389个completed updates没有周期checkpoint，未来不得从31,388猜恢复状态；
  exact continuation必须从31,000建立新的W&B lineage。
- fresh W&B API已确认`mlgjus6q`为`finished`而非伪装完成：
  `lifecycle/user_paused=1`、`user_stopped=1`、
  `training_target_reached=0`、`stop_reason=user_requested_pause`、
  `resumable_from_checkpoint=1`，并记录上述checkpoint SHA、remote验证、
  31,388尾部及`resume_should_use_new_wandb_run=1`。这些字段同时存在于
  history step 31,387和summary；远端`model_31000.pt`仍可访问。

## 6urn原始30/旧CORL79分层单机器人评估（2026-07-29 21:04 UTC）

- 用户要求用`zihanw22/carry-any/6urn4jvc/model_25000.pt`替换先前
  `wgcsw25u`完成同规格zoom-out轨迹对比，并进一步要求先评估原来的4条，
  再从该run实际绑定的原始30条bank按同标准选择。exact checkpoint为
  completed iteration 24,999 / next iteration 25,000，SHA256=
  `c9e35fc51f1902a82a89d8911dbe084084b2684253a0a3f2c1ba10fd5a70858d`；
  本地使用content-addressed只读副本，没有修改训练checkpoint。
- checkpoint认证的30个object mesh与旧4条CORL79 mesh交集为0。因此旧
  `box_10/noscale__any_ball_3/noscale__any_barrel_12/
  noscale__any_bin_32`被明确标为OOD，不得伪称in-distribution。为此增加
  默认关闭、仅限single-env `checkpoint_actor` evaluation的object-geometry
  opt-in；它只允许选定live object不属于checkpoint training support，
  camera source、robot meshes、observation、normalizer、actor及provenance
  仍严格认证，并逐条写入OOD audit。训练、resume和policy-init不能启用。
- 原始30条bank固定为view digest
  `1f3966245545689a2e14909dda31d6673790bdf888168362449a0ac1902f3510`，
  类别只有ball 5、barrel 1、bin 24，没有box。按已有代表选择标准各取
  `unscale__any_ball_29`、`scaledown__any_barrel_25`、
  `unscale__any_bin_29`一条；这3条均走默认严格geometry membership，
  没有启用OOD开关。
- 7条rollout均为1 env/1 robot、seed 42、timestep 0、501 steps@50Hz，
  physical randomization disabled、initial-pose noise 0、adaptive/T1
  sampling关闭，auto/motion-end/clip-end reset关闭。object world-z相对
  初始值升高0.30m并连续10步后，command切换为robot-heading-frame
  relative pose `[0.15,0,0,0]`；这是持续相对位移，不是速度。相机offset
  `[3,3,1.6]`、target offset `[0,0,0.35]`，成片动态绘制G1 root与object
  的world-XY轨迹。
- 旧4条OOD结果为pickup 2/4、strict end carry 1/4：box和barrel未稳定
  抬起；ball在1.81s触发并完整搬运，object XY displacement 5.151m；
  bin在1.97s触发但中途掉落，post-trigger carry fraction 0.5197。原始30
  的3条in-bank代表均pickup且strict end carry通过，触发时间依次为
  1.75/1.83/1.81s，object XY displacement为4.830/5.011/5.083m。
  这是固定seed下的代表clip诊断，不能外推成整个bank的成功率。
- 本地审计根为
  `/home/ubuntu/FAR/holosoma_runs/policy_comparison_sw39999_34q40000_6urn25000_20260729_203536`。
  recording/analysis/trajectory manifest SHA256依次为
  `449e08c7bd942e6afe625c9d939684046da776c92c42f54f8e24abf76d716cde`、
  `416f433131536a68baa3276dd02dd42368516ea8cd36ac93bf018abc952a8104`、
  `bf3e8185421a6a8476819c314bc681ff760ebd00bbe3b7d5536a9b683b05f258`。
  全部27个raw/annotated/final MP4均fresh ffprobe为恰好一个H.264流、正尺寸、
  50fps、501帧、10.02s；5张跨时间contact sheet人工确认robot/object/
  trajectory连续。5个最终1920x560视频也已逐字节复制到
  `/home/ubuntu/FAR/_check_vis`。
- 原training run `6urn4jvc`只通过secondary writer新增同一条history row
  step 25,344的5个media：4个OOD三策略对比及1个原始30三类别汇总；
  checkpoint metadata严格为25,000。Artifact
  `policy-comparison-zoomout-trajectories-6urn4jvc-model25000-sw39999-34q40000:v0`
  已核对为20个MP4、5张contact sheet和3份manifest。fresh API逐个下载
  5个history MP4复算SHA/size并ffprobe，summary path与history path一致，
  Artifact中的trajectory manifest SHA一致；最终run仍为`running`，
  summary已推进到step 25,350，没有新建run或改变训练lifecycle。
- 轨迹编码第一次因controller root盘写满而中断；仅执行
  `python -m pip cache purge`清理可重建下载缓存，释放约5.9GB并恢复
  约5.5GB可用空间，没有删除实验、checkpoint或结果。最终py_compile、
  两组focused regression分别`10 passed`和`55 passed`，工作树与index的
  `git diff --check`均通过；未暂存或覆盖其他既有用户修改。

## mlgjus6q-30K teacher的16卡depth distillation正式训练（2026-07-30 05:05 UTC）

- 正式W&B run为`zihanw22/carry-any/n09wopdy`：
  <https://wandb.ai/zihanw22/carry-any/runs/n09wopdy>；run name为
  `distill_depth_dual_original30_mlgjus6q30000_ws16_e2048_chunkfinite_20260730_034057`。
  这是fresh student，没有training resume或policy init；截至本记录已
  `completed_iteration=64`且仍在运行，后续不得误停、重启或改用旧失败run。
- 拓扑为`10.99.0.18`和`10.99.0.227`各8张L40S，共16 ranks；
  session统一为`hs_n09wopdy`，每rank/GPU 2048 env、全局32,768 env。
  每次update为24 steps/env，即全局786,432 samples；1 epoch、
  64 minibatches，每rank每个minibatch 768 samples。启动控制器exit 0，
  两节点均通过`distributed_provenance=8/8`和`final_workers=8/8`，
  验收时各8个worker和8个GPU compute app仍存活。
- label teacher固定为
  `wandb://zihanw22/carry-any/mlgjus6q/model_30000.pt`，本地精确字节
  SHA256=`997f30c471e71199a2392c4593411ddeb29e90ecc5df3920d85f9e59b9ecb2cc`。
  teacher为178D privileged actor、MLP `[512,256,128]`、29D action；
  student为95D scalar input加D435i depth、MLP
  `[2048,1024,512,256,128]`。actor/critic adaptive LR均从`1e-3`
  开始，desired KL 0.01，noise std 0.01，entropy 0。
- DAgger/PPO schedule严格为iteration 0时PPO coeff 0、BC weight 1；
  每700 updates增加0.1，4900时到PPO 0.7/BC 0.3。验收日志中
  iteration 0到64均为`ppo_dagger_coeff=0.0000`、
  `ppo_dagger_bc_weight=1.0000`，没有错误地从0.1开始。
  总目标40,000 updates，checkpoint和ONNX每1000 updates保存，跳过step 0。
- 输入bank为原始30条immutable realmesh view：ball 5、barrel 1、bin 24。
  single-slot source/view digest分别为
  `d35955490ddf3b03063bfb20cb65c88234589b856244d37467740c67458436c5`和
  `234f215d4c52ac1dc6f67b9ba3968678fd3596c0812a253eb1ffaf5d76e20ada`；
  `ws16` shard source digest为
  `27aee938d8447ff537f83bfca80263e9968118487bd228bba936ab943aea0b23`，
  manifest SHA256=
  `dd392594c400f22b2e288468df04a87161bc15cda6cf14df038e6c659ab3a5c5`。
  每环境只有一个object slot；每rank约2条clip，fixed env-to-clip
  assignment和inverse-cover loss correction保持全局uniform-clip
  objective，没有PyTorch/DataLoader二次shard。
- collision使用`convex_decomposition`，depth/raycast明确使用
  `perception.object_geometry_mode=mesh`的真实visual mesh。
  object filtered contact sensors关闭；offline contact guidance及所有
  arms/palms/torso正向contact reward均为0，contact sidecar只用于
  adaptive timestep和pickup/drop window。start-at-t0从0.2在
  iteration 2500到39999线性升到1.0，freeze概率始终0。
- 第一次作废run`ic8tl4jh`因PhysX found/lost pair容量不足而报告missed
  interactions；第二次作废run`4x0qhgy2`已修正PhysX容量并完成iteration 0，
  但完整depth rollout的`torch.isfinite()`一次产生约238 MiB bool临时量，
  在iteration 1前因只剩约126 MiB显存而OOM。两者在W&B均已标记
  `formal-invalid`和`superseded-by-n09wopdy`，不得resume。
- 当前source仅把rollout finite诊断改为逐timestep穷举：
  每个已填充buffer仍逐元素执行同一`isfinite`谓词，仍合并所有buffer并做
  同一跨rank MAX verdict；只把depth bool临时量峰值从约238 MiB降到约
  9.9 MiB，不改变sample、loss、gradient、optimizer、schedule或异常语义。
  新回归验证最后一个timestep中的NaN仍抛错且`isfinite`恰好按每个step调用；
  teacher nonfinite、全部floating rollout buffer及chunked检查共`3 passed`。
- 新run iteration 0为47,882 steps/s，collection 12.443 s、learning
  3.981 s、总16.42 s；旧`4x0qhgy2` iteration 0为41,337 steps/s，
  collection 14.952 s、learning 4.073 s、总19.02 s，即同起点吞吐约
  提高15.8%、iteration time降低13.7%。warmup后iteration 53到58约
  89,180到91,968 steps/s、8.55到8.82 s/iter；这部分同时包含runtime
  warmup，不能全部归因于finite-check改动。两节点全日志中Traceback、
  OOM、NCCL/Gloo fatal、non-finite、segfault和PhysX missed-interaction
  均为0。
- Rule-90使用最终bank canonical第一条`scaledown__any_ball_24`的全新
  policy-free reference replay；H.264 1280x720@50fps、329帧、6.58秒、
  2,017,963 bytes，视频SHA256=
  `40c72dd7a334b466182ad72697353d34fff8d807bc271ac8cda6f98c7ccdcbe6`。
  ffprobe、完整decode和人工contact-sheet检查通过；replay manifest
  SHA256=`442ed879835593c8623dcab4a8aadd637531c1b0ccf36561ae8a71eec756bf29`，
  immutable run contract SHA256=
  `995dfcdf7432cd113e9950ce9ce9f86028a1be50c6a8dca701c36d8fafc47db5`。
  训练history建立后，同一字节写入唯一history step 39的`vis/replay`；
  primary summary绑定该history path，远端下载复算SHA/size后删除prebind
  duplicate。fresh API确认run仍`running`、恰好1个replay row和1个MP4。
- sealed source为
  `src-7ca8a5d3ab710e83054a5439eabf988f94c98463248c3718f47189d170eeb609`，
  source archive SHA256=
  `18274e35a8aa34ebb178cdc2e0ed9fd7f2d0b1139fed6b08587e8ade556e52f4`。
  完整controller、Rule-90、contract、dry-run和lifecycle审计根为
  `/data/holosoma-distill-launches/formal_distill_original30_mlgjus6q30000_ws16_e2048_physx512m_chunkfinite_20260730_034057`。

## 原始30三个相同初始化的三checkpoint策略对比（2026-07-30 05:49 UTC）

- 用户要求复用
  `original30_representatives__6urn25000__ball_barrel_bin__zoomout_xy_trajectories.mp4`
  中的三个初始化，对`swl41n4x/model_39999`、
  `34qv1qqp/model_40000`和`6urn4jvc/model_25000`做逐物体同规格对比。
  exact checkpoint SHA256依次为
  `7f62166a326423704841f87446fd953d6ea7d466a97872cb020f835f36db3081`、
  `ec7a2de8d0f5a9e7a74ac8b1e70d50f11c346de24670a2eaa73bdd560dae52fe`、
  `c9e35fc51f1902a82a89d8911dbe084084b2684253a0a3f2c1ba10fd5a70858d`。
  SW旧checkpoint没有authenticated geometry-support证据；34q的79个
  authenticated training mesh与这三个object均不相交，故显式标为OOD；
  三个object均属于6urn认证的原始30 support。
- exact初始化为`unscale__any_ball_29`、`scaledown__any_barrel_25`、
  `unscale__any_bin_29`，nominal mass依次为0.5/1.5/1.0 kg；对应URDF
  SHA256依次为
  `15b51ea323e3746fdc10a41afa3ef9b90a3c9237a382a1092be7e358461004cb`、
  `b83f9562bbd82ba92ecb6e7ad62cf4e182dff769156c32d5704f747710a82982`、
  `fbd84cbcbceb322d389af6611b06dcb7e7381a9ad1149d9d81a32de0999c90b7`。
  object map SHA256为`29672bc4...0c92d`、`1573aebb...ace78`、
  `14dc6534...9190`；immutable view digest仍为
  `1f3966245545689a2e14909dda31d6673790bdf888168362449a0ac1902f3510`。
- 9条rollout均为1 env/1 robot、seed42、randomization disabled、
  initial-pose noise 0、forced timestep0、501 steps@50Hz，关闭
  adaptive/T1 sampling及auto/motion-end/clip-end reset。lift后持续使用
  heading-frame relative pose`[0.15,0,0,0]`，相机offset
  `[3,3,1.6]`、target`[0,0,0.35]`。SW checkpoint曾继承legacy
  `default_pose_prepend=False`，即使forced reset为0，首个记录行仍从
  motion timestep2开始；该attempt fail closed保留后，显式统一0.2秒
  prepend和2.0秒append重新录制。最终每个clip三策略的完整501-step
  motion-timestep序列逐元素相同，开头为
  `[0,0,0,0,0,0,0,0,0,1,2,3,4,5,6]`，末尾五项均为318，
  timeline SHA256=
  `775d4bdca62b67b69eb629e0cb677099f8a6226db2f5a53c3b0ca6f5808e8074`。
- ball的SW/34q/6urn pickup时间为3.31/2.11/1.75秒，post-trigger object
  XY displacement为3.444/1.691/4.830米，carry fraction均1；barrel为
  2.63/2.19/1.83秒与2.346/0.482/5.011米，carry fraction均1；bin为
  1.83/2.15/1.81秒与3.585/5.715/5.083米，其中SW bin的carry fraction
  只有0.586，另外两列为1。9/9均pickup且末帧strict carry通过，但不能
  用末帧通过隐藏SW bin中途失持，也不能忽略34q ball/barrel净位移较小。
  这只是固定seed的三个诊断clip，不能外推总体robustness。
- 本地审计根为
  `/home/ubuntu/FAR/holosoma_runs/original30_representatives_policy_comparison_sw39999_34q40000_6urn25000_20260730_050347`；
  recording/analysis/video manifest SHA256依次为
  `5f73be0f12a64fe42f20150ea332281f9dcc95745da482068981a962dc4c1984`、
  `10fce394ca15075c6af33cec4d80a7002a495b3db9ec35c6656a165b317e56d6`、
  `707aebd09cd1733d5cb75052a7cbff29371e8e7d74cf0e847ab94287387b011b`。
  9个annotated panel、3个1920x560 triptych和1个1920x1680 3×3总览均为
  H.264 50fps、501帧、10.02秒；四张contact sheet人工确认robot/object/
  trajectory连续。四个最终对比视频已逐字节复制到
  `/home/ubuntu/FAR/_check_vis`，文件名均以`original30_`开头。
- 原training run `zihanw22/carry-any/6urn4jvc`只新增同一history row
  step29,881的4个media，namespace为
  `vis/original30_same_initialization_policy_comparison/`，没有新建run。
  secondary writer上传后通过rank0 exact live primary service把summary
  稳定绑定到该history path；fresh API跨step29,964到29,973再次确认四个
  summary/history path一致，并逐个下载复算SHA/size/ffprobe。Artifact
  `original30-same-initialization-policy-comparison-sw39999-34q40000-6urn25000:v0`
  精确包含13个MP4、4张contact sheet和3份manifest，共20个文件；
  Artifact内三份manifest SHA均与本地一致。复核后run仍为`running`，
  没有finish、暂停或修改正式训练进程；完整远端结果写入审计根的
  `wandb_upload_verification.json`。
- 最终再次跨训练推进到summary step29,993复核：matching history row=1、
  matching Artifact=1、四个primary summary path仍逐项等于history path，
  run仍为`running`。相关source及审计脚本全部通过`py_compile`；
  OOD/perception/checkpoint测试55项、recorder/manual-command测试16项通过，
  `git diff --check`和`git diff --cached --check`均通过。所有本轮evaluation
  进程已退出，未留下monitor。

## 原始30三个相同初始化的35K替换对比（2026-07-30 17:45 UTC）

- 用户要求在上一节完全相同的三个初始化和录制协议下，用
  `6urn4jvc/model_35000`替换第三列`model_25000`；前两列仍为
  `swl41n4x/model_39999`和`34qv1qqp/model_40000`。35K checkpoint的
  SHA256为
  `5501ec25c1079053c6de829ab0e1c7931740f12b5f250f92670592a37b4cfbc4`，
  文件内`iter=34999`、`iteration=34999`、`next_iter=35000`，其
  motion-transition contract SHA256仍为
  `ab843902d4d69a5d28e0b9bbdf7e8dacfb21e0b22f3ab89173ea1ff5c7f83442`，
  并与25K同属经过认证的原始30 mesh support。
- exact初始化仍为`unscale__any_ball_29`（0.5 kg）、
  `scaledown__any_barrel_25`（1.5 kg）和`unscale__any_bin_29`
  （1.0 kg）；URDF、object map和immutable source view均未变化，view
  digest仍为
  `1f3966245545689a2e14909dda31d6673790bdf888168362449a0ac1902f3510`。
  只新运行三条35K rollout；六条已验证的SW/34q录制逐字节复用。
- 协议仍为1 env/1 robot、seed42、randomization disabled、
  initial-pose noise 0、forced timestep0、501 steps@50Hz、显式0.2秒
  prepend和2.0秒append，关闭adaptive/T1及所有reset。三策略每条clip的
  完整motion-timestep序列逐元素相同，timeline SHA256仍为
  `775d4bdca62b67b69eb629e0cb677099f8a6226db2f5a53c3b0ca6f5808e8074`。
- 35K在ball/barrel/bin上的pickup时间依次为1.79/1.87/1.89秒，
  post-trigger object XY displacement依次为3.954/4.283/5.521米，
  carry fraction均为1且末帧strict carry均通过。相比25K的
  4.830/5.011/5.083米，35K在ball和barrel上的净位移较短、bin较长，
  pickup约晚0.04到0.08秒。三个固定seed诊断clip不足以判定35K总体比
  25K更robust或更差。
- 本地审计根为
  `/home/ubuntu/FAR/holosoma_runs/original30_representatives_policy_comparison_sw39999_34q40000_6urn35000_20260730_173019`；
  recording/analysis/video manifest SHA256依次为
  `b227826eb81403d0764b41795ebb109dc2461fc1a2df5896e94018e51243ce5c`、
  `a22192c2e2054724c8b953893f9f58090bb3380f4cff83b10d29fddd3817e2a3`、
  `6e22270e2c74798f4d1c6f700f959cbe56d0a737d4e75ff17c6fd9823ce6f7f0`。
  9个annotated panel、3个1920x560 triptych和1个1920x1680 3×3总览均为
  H.264 50fps、501帧、10.02秒；ffprobe、完整decode及四张contact sheet
  人工检查通过。四个最终视频已逐字节复制到
  `/home/ubuntu/FAR/_check_vis`，文件名均包含`6urn35000`。
- 原training run `zihanw22/carry-any/6urn4jvc`只新增一个history row
  step36,275的4个media，namespace为
  `vis/original30_same_initialization_policy_comparison_35k/`。
  Artifact
  `original30-same-initialization-policy-comparison-sw39999-34q40000-6urn35000:v0`
  精确包含13个MP4、4张contact sheet和3份manifest，共20个文件。
  fresh API逐个下载复算四个视频SHA/size/ffprobe，并确认summary/history
  path、checkpoint metadata、Artifact inventory和三份manifest SHA均
  精确一致；最终跨训练推进到summary step36,298再次确认matching history
  row=1、matching Artifact=1、run仍为`running`，没有finish、暂停或修改
  正式训练进程。
- 全部相关shell和Python脚本通过语法检查；recorder/manual-command测试
  16项、OOD/perception/checkpoint测试55项通过，`git diff --check`和
  `git diff --cached --check`均通过。所有本轮evaluation进程已退出，
  GPU 3/4/5恢复空闲，未留下monitor。

## consecutive_steps=0与n09最新checkpoint四策略对比（2026-07-30 19:55 UTC）

- 用户要求把lift后的debounce改为`consecutive_steps=0`，并在完全相同的
  三个初始化下比较`swl41n4x/model_39999`、`34qv1qqp/model_40000`、
  `6urn4jvc/model_35000`和`n09wopdy`的最新checkpoint。实现契约为：
  0只关闭debounce，仍必须等object world-z相对初始值首次达到+0.30米才
  触发，不能在episode开始时直接绕过阈值；触发前命令为`[0,0,0,0]`，
  首个threshold-qualified control step开始使用robot-heading-frame
  relative pose`[0.15,0,0,0]`，不启用world-heading lock。对应实现和
  单元测试位于`MotionCommand._update_manual_forward_after_lift`及
  `tests/unit/test_manual_forward_after_lift.py`。
- `n09wopdy`远端文件扫描只有`model_01000.pt`至`model_05000.pt`，因此
  “最新”明确选择`model_05000.pt`，不是35K；选择时该run状态为`failed`。
  checkpoint SHA256为
  `d0593e0fc8a97fef18d017826ef2e139d2becd473652ee82fead1bd3e78901ac`，
  文件内`iter=4999`、`iteration=4999`、`next_iter=5000`。它与
  `6urn4jvc/model_35000`的30个authenticated training meshes集合完全
  相同，motion-transition contract SHA256也同为
  `ab8439024b6e619f59638fc967734433ab08cec607698433c2641b75b7163442`；
  但n09 actor因pickup button多一个输入槽（首层127对35K的126），录制时
  使用其checkpoint-native config，不能把两者当成training-age matched
  的等价checkpoint。
- exact初始化仍为`unscale__any_ball_29`（0.5 kg）、
  `scaledown__any_barrel_25`（1.5 kg）和`unscale__any_bin_29`
  （1.0 kg）；四策略逐clip共用同一motion、URDF、object map和immutable
  source view，view digest为
  `1f3966245545689a2e14909dda31d6673790bdf888168362449a0ac1902f3510`。
  协议为1 env/1 robot、seed42、randomization disabled、initial-pose
  noise 0、forced timestep0、501 steps@50Hz、0.2秒prepend、2.0秒append，
  关闭adaptive/T1及全部reset。每个clip四策略的501-step motion-timestep
  序列逐元素相同，timeline SHA256为
  `775d4bdca62b67b69eb629e0cb677099f8a6226db2f5a53c3b0ca6f5808e8074`。
- ball的SW/34q/35K首次触发时间为3.13/1.97/1.61秒，post-trigger object
  XY displacement为3.425/0.963/3.985米；barrel为2.45/1.99/1.69秒与
  2.422/4.823/4.417米；bin为1.65/1.93/1.71秒与
  1.056/8.035/5.635米。三策略9/9均pickup、carry fraction均1且末帧
  strict carry通过。n09三条均未达到+0.30米阈值，最大object z增量仅为
  0.010/0.067/0.055米，故命令在全部501步保持0、pickup为0/3；其分析器
  的post-trigger指标为0只是“没有trigger”的定义，不能表述为robot或
  object完全没有运动。固定seed的三个诊断clip只能说明本样本上35K为3/3、
  n09 5K为0/3，不能外推总体robustness。
- 本地审计根为
  `/home/ubuntu/FAR/holosoma_runs/original30_representatives_policy_comparison_sw39999_34q40000_6urn35000_consecutive0_20260730_193235`；
  recording/analysis/video/media-validation manifest SHA256依次为
  `9ab3190c8bac4187af19a614a538bde1b8d988dd285d5fc3007dbd6c3bc4faee`、
  `214013780fea00679e63c7c3093067be45cd137c864f2ead84e7aef949bfa15e`、
  `5cbef09d43cba91884940477e46854b5e44ad41de3d9dd9fa8a6bc086f6bff1a`、
  `0c499e6e9873acd989ffeab7b0593581bbc60f505f833541715eeb5db1aa9cbb`。
  12个raw、12个annotated、3个2560x560四列对比和1个2560x1680的3×4
  总览，共28个H.264 MP4，全部50fps、501帧、10.02秒，逐个ffprobe及
  完整decode通过；ball/barrel/bin/总览contact sheet均人工确认object、
  robot和XY轨迹连续。四个最终视频已逐字节复制到
  `/home/ubuntu/FAR/_check_vis`。
- 原training run`zihanw22/carry-any/6urn4jvc`只新增一个history row
  step37,420的4个media，namespace为
  `vis/original30_same_initialization_policy_comparison_35k_consecutive0_n09/`。
  Artifact
  `original30-same-init-comparison-sw39999-34q40000-6urn35000-n09w05000-consecutive0:v0`
  的公开inventory精确包含16个MP4、4张contact sheet及4份manifest，
  共24个文件。fresh API逐个下载复算四个交付视频的SHA/size/ffprobe，
  并确认summary/history path、checkpoint metadata、Artifact inventory
  和manifest SHA全部一致；复核后`6urn4jvc`仍为`running`。
  `n09wopdy`仅被读取，没有写入或改变其W&B lifecycle。
- targeted recorder/evaluation/perception/checkpoint测试共115项通过；
  审计脚本通过shell语法或`py_compile`检查。所有本轮evaluation进程已
  退出，GPU 3/4/5/6/7恢复空闲，未留下monitor。

## consecutive_steps=0五策略对比新增n09 label teacher（2026-07-30 20:19 UTC）

- 用户要求在上一节四列对比后新增第五列，且必须是“训练第四列policy时
  使用的teacher policy”。第四列`n09wopdy/model_05000` checkpoint的
  finalized provenance把唯一label teacher固定为
  `wandb://zihanw22/carry-any/mlgjus6q/model_30000.pt`，teacher SHA256=
  `997f30c471e71199a2392c4593411ddeb29e90ecc5df3920d85f9e59b9ecb2cc`，
  文件内`iter=29999/iteration=29999/next_iter=30000`。该teacher是从零
  训练的178D privileged pure-RL actor、29D action，没有external teacher、
  resume或policy init。
- teacher rollout不是把`mlgjus6q`文件名猜作另一个普通checkpoint actor。
  evaluation入口仍加载精确的`n09wopdy/model_05000`作为身份认证
  checkpoint，并显式使用`HOLOSOMA_EVAL_POLICY=distill_label_teacher`；
  由n09 source config中的`policy_to_clone`和
  `training_provenance.teacher_sha256`strict-load上述teacher。每条录制日志
  都确认选择该SHA，policy-I/O抽样为178D raw/normalized privileged obs、
  29D finite action、无perception input，student actor不参与动作生成。
  本列运行underlying teacher的deterministic raw`act_inference`，不是把
  动作事后变成训练BC loss使用的clipped label tensor。
- ball/barrel/bin仍使用与前四列逐字节相同的motion、URDF、object map和
  real mesh，并保持1 env/1 robot、seed42、randomization disabled、
  initial-pose noise 0、forced timestep0、501 steps@50Hz、0.2秒prepend、
  2.0秒append、关闭adaptive/T1/reset。`consecutive_steps=0`仍只关闭
  debounce：object world-z首次达到+0.30米才把命令从`[0,0,0,0]`切为
  robot-heading-frame relative pose`[0.15,0,0,0]`，不启用heading lock。
  五策略逐clip的完整timeline逐元素相同，SHA256仍为
  `775d4bdca62b67b69eb629e0cb677099f8a6226db2f5a53c3b0ca6f5808e8074`。
- teacher在ball/barrel/bin上的首次触发时间为1.61/1.63/1.61秒，
  post-trigger object XY displacement为2.101/2.961/1.785米。三条均完成
  pickup，但carry fraction只有0.389/0.376/0.375，末帧strict carry均失败；
  即teacher是3/3 pickup、0/3 end carry。contact sheet和完整视频明确显示
  其pickup后失持，不能只用pickup成功隐藏。相同样本中n09 5K student为
  0/3 pickup，而6urn 35K为3/3 pickup且3/3 end carry；这些仍只是固定seed
  的三个诊断clip，不能外推总体robustness。
- 本地审计根为
  `/home/ubuntu/FAR/holosoma_runs/original30_representatives_policy_comparison_sw39999_34q40000_6urn35000_n09teacher_consecutive0_20260730_200238`；
  recording/analysis/video/media-validation manifest SHA256依次为
  `d18166940765ca61cb5490d6d74cea47f3c3161bc416b8ff63803e15c1aa8c90`、
  `07a6f758a20f976ba54481e315b33e8f7f8c3f5838c973d0055723763017d59c`、
  `394d755c60498edeba25d2e30a9f12c273bccfd37efd632f68041b0f4c362f72`、
  `0ef2c591241e6d3e254debc4869bf794e92389372458b03a6daf2cd4f84ea4d0`。
  15个raw、15个annotated、3个3200×560五列比较和1个3200×1680的
  3×5总览，共34个H.264 MP4，全部50fps、501帧、10.02秒；逐文件
  ffprobe和完整decode通过，四张contact sheet人工确认正确ball/barrel/bin、
  robot/object/trajectory连续，无default-pose或错物体替换。四个最终视频
  已逐字节复制到`/home/ubuntu/FAR/_check_vis`。
- 原training run`zihanw22/carry-any/6urn4jvc`只新增history step37,624
  的4个media，namespace为
  `vis/original30_same_initialization_policy_comparison_35k_consecutive0_n09_teacher/`。
  Artifact
  `original30-same-init-comparison-sw39999-34q40000-6urn35000-n09w05000-n09teacher-mlgjus6q30000-consecutive0:v0`
  的公开inventory精确为19个MP4、4张contact sheet和4份manifest，共27个
  文件。fresh API逐个下载四个交付视频并复算SHA/size/ffprobe，也核对
  primary summary/history path、teacher身份、Artifact inventory和四份
  manifest SHA；随后跨训练推进至summary step37,665再次确认matching
  history row=1、matching Artifact=1、summary仍绑定同一媒体且run为
  `running`。n09和
  mlgjus6q run均只读，没有改其W&B lifecycle；远端临时summary绑定文件已
  删除。
- teacher evaluation role的定向回归4项、manual-command/evaluation/
  perception/checkpoint回归115项通过；本轮所有审计shell/Python脚本通过
  语法检查，repo worktree/index的`git diff --check`均通过。
  所有本轮evaluation、编码和上传进程已退出，GPU 3/4/5恢复空闲，未留下
  monitor。

## n09 label teacher的raw与clipped-BC-target轨迹对比（2026-07-30 21:25 UTC）

- 用户确认`n09wopdy`训练使用clipped teacher label后，要求按上一批相同
  画法展示G1与物体轨迹。本轮没有把raw teacher视频改名冒充训练label；
  evaluation新增显式角色
  `HOLOSOMA_EVAL_POLICY=distill_label_teacher_bc_target`。入口仍由
  `n09wopdy/model_05000` checkpoint认证唯一teacher
  `mlgjus6q/model_30000.pt`、SHA256=
  `997f30c471e71199a2392c4593411ddeb29e90ecc5df3920d85f9e59b9ecb2cc`，
  并从checkpoint自身的source config读回
  `clip_teacher_actions=true/clip_actions_threshold=8.0`，随后在
  `env.step`前逐维执行`clamp(action,-8,8)`。该模式只允许evaluation，
  不改变正式训练或普通checkpoint actor语义。
- ball/barrel/bin继续逐字节复用上一批相同motion、URDF、object map、
  real mesh和初始模拟状态；仍为1 env、seed42、randomization disabled、
  zero initial-pose noise、forced timestep0、501 steps@50Hz及
  `consecutive_steps=0`。六策略逐clip的完整motion-timestep sequence
  仍相同，timeline SHA256=
  `775d4bdca62b67b69eb629e0cb677099f8a6226db2f5a53c3b0ca6f5808e8074`。
- clipped三条共43,587个action分量全部finite且位于`[-8,8]`；ball/barrel/bin
  分别有66/193/93个分量精确落在clip边界，证明变换实际生效。raw与clipped
  从exact相同初态开始，并在metric step429/100/421首次分叉；raw整段
  max-abs分别为9.860/28.422/16.941，clipped均为8.0。两种版本在这三个
  固定诊断clip上都是3/3 pickup、0/3 strict-end-carry，不能据此宣称clipping
  改善任务成功率。
- 轨迹仍定义为以G1在t0的world XY为原点：G1 root为cyan、object为orange，
  每个object的六列共用完全相同XY bounds。生成了每object六列、每object
  raw-vs-clipped两列、3×6总览和3×2总览；八个交付comparison已逐字节
  复制到`/home/ubuntu/FAR/_check_vis`。本地审计根为
  `/home/ubuntu/FAR/holosoma_runs/original30_representatives_policy_comparison_teacher_raw_vs_clipped_consecutive0_20260730_211551`。
- recording/analysis/video/media-validation manifest SHA256依次为
  `c138cc8faa685d59cbbe51d15a89b2e7c3925f46e5ef6e1bada30ec7fdf66eaa`、
  `cd776cebec907c266ce82c83b0cfec02f03c5970590f1fcb3e57f8be686c81e1`、
  `a288245c0bebddc2a900de32c49251fd020adc2890edf0bfa9566b7eae2c8980`、
  `9633e0c218a4044645c72fdc3aae12b25fb0a491154c4a004d802ba8d3a532a8`。
  新root内29个MP4均为H.264/yuv420p、50fps、501帧、10.02秒，逐文件
  ffprobe与完整decode通过；八张contact sheet人工确认单机器人、正确
  ball/barrel/bin、连续动作与动态XY轨迹，无default-pose或错物体替换。
  本轮为本地交付，没有写入W&B或改变任何run lifecycle。录制、编码和验证
  进程均退出，GPU 3/4/5恢复空闲。teacher evaluation与setup-order完整
  回归83项通过，`py_compile`和`git diff --check`通过。

## kdw7jhze最新完整pure-BC checkpoint的三物体录制（2026-07-31 05:10 UTC）

- 用户要求录制`zihanw22/carry-any/kdw7jhze`的最新BC。开始选择和全部录制
  校验完成后均重新用fresh W&B API扫描；截至2026-07-31 05:09:35 UTC，
  run仍为`running`、summary step为1964，但远端唯一且因此数值上最新的
  完整checkpoint仍是`model_01000.pt`。不能把在线summary step 1964冒充
  可下载checkpoint。1K文件为127,235,764 bytes，MD5与W&B一致，SHA256=
  `8103c08bc05b310653bf9895a50a95cf8f6f075f8d6f0867ac02ce74dac9c038`；
  文件内`iter=999/iteration=999/next_iter=1000`，341个tensor、
  31,758,432个值全部finite。
- 该run是fresh pure behavior cloning student：PPO start/target coeff均为0，
  BC loss coeff为1，DAgger replay关闭，teacher action mix为0，
  `take_teacher_actions=false`；训练label来自
  `mlgjus6q/model_30000`、teacher SHA256=
  `997f30c471e71199a2392c4593411ddeb29e90ecc5df3920d85f9e59b9ecb2cc`
  并按source contract clamp到`[-8,8]`。本轮使用
  `HOLOSOMA_EVAL_POLICY=checkpoint_actor`评估1K student自身，不是teacher
  rollout，也不是把clipped label直接送入环境。checkpoint认证无resume、
  无policy init；其30个object geometry identity集合与已验证的
  `6urn4jvc/model_35000`完全相同，robot mesh binding也相同，motion-
  transition contract SHA256同为
  `ab8439024b6e619f59638fc967734433ab08cec607698433c2641b75b7163442`。
- exact初始化继续使用`unscale__any_ball_29`（0.5 kg）、
  `scaledown__any_barrel_25`（1.5 kg）和`unscale__any_bin_29`（1.0 kg）；
  motion、URDF、object map、real mesh、seed和timestep均与上一批六列逐字节
  对齐。协议仍为1 env/1 G1、seed42、randomization disabled、initial-pose
  noise 0、forced timestep0、501 steps@50Hz、0.2秒prepend、2.0秒append，
  关闭adaptive/T1及全部reset；`consecutive_steps=0`只取消debounce，
  仍需object相对初始world-z达到+0.30米才启用`[0.15,0,0,0]`。七列逐clip
  的501-step motion timeline逐元素相同，SHA256=
  `775d4bdca62b67b69eb629e0cb677099f8a6226db2f5a53c3b0ca6f5808e8074`。
- 1K BC在ball/barrel/bin上均未达到pickup阈值，最大object相对z增量分别为
  0.0385/0.0672/0.0613米，因此三条forward command全程保持0，结果为
  0/3 pickup、0/3 strict-end-carry。三条student action的max-abs分别为
  8.094/8.644/8.642；这不违背“训练label clamp到8”，因为BC student是
  连续回归器，`checkpoint_actor`推理输出本身没有再做label clamp。该结果
  只说明训练极早期1K在三个固定seed诊断clip上尚未学会pickup，不能外推
  最终40K性能或总体robustness。
- 轨迹仍以G1在t0的world XY为原点，G1 root为cyan、object为orange，每个
  object的七列共用同一XY bounds。生成了BC三物体3×1、n09 5K student与
  kdw 1K pure-BC的3×2直比、完整七策略3×7、三个逐object七列以及三条raw/
  三条单列trajectory视频；12个主要文件均逐字节复制并复核到
  `/home/ubuntu/FAR/_check_vis`。
- 本地审计根为
  `/home/ubuntu/FAR/holosoma_runs/original30_representatives_policy_comparison_latestbc_kdw7jhze_model01000_consecutive0_20260731_045729`；
  checkpoint-selection/recording/analysis/video/media-validation/
  W&B-latest-recheck/delivery manifest SHA256依次为
  `162845cdac8582cd4c2c4baf94111dd780493caf851fb998482d0c28033253cc`、
  `12f6b374111cd1ded2a3a14bc1ead0eacb0f3dd7770abf4cd280992bd931ca40`、
  `738f15332023306bcadf3c5447b63c66005db9b0fb8f6c87e0f9a87f0123fee8`、
  `cfebb419886519c6dd53c2248bc18c51e36d0e0ee813f170e4eaa4d7c3b8d5b9`、
  `120a7c1043b7510e15c6dac1b2df1a4f7edbb0a3cf54ed633d69cbe6a86e0a17`、
  `79b2bcfb207a53f6c1788ad3e431ee159382272aed15a97b7dec323355a29f85`、
  `a4dcd3d17a12fb44c6ee0b2d245bcbc7297474b8bdf7af34325babf6404925d3`。
  新root内30个MP4全部为单一H.264/yuv420p stream、50fps、501帧、
  10.02秒，逐个ffprobe与完整decode通过；3×7和3×2 contact sheet人工
  确认正确ball/barrel/bin、单G1、连续动作和动态XY轨迹，无default-pose或
  错物体替换。所有脚本通过`py_compile`或`bash -n`，录制、编码和校验
  进程均退出，GPU 3/4/5恢复空闲。本轮只读W&B，没有上传media、修改summary
  或改变任何run lifecycle。

## 6urn4jvc 35K direct-RL的0.5米command三物体对照（2026-07-31 05:35 UTC）

- 用户所说“之前的直接RL checkpoint”绑定为
  `zihanw22/carry-any/6urn4jvc/model_35000.pt`，而不是privileged teacher
  `mlgjus6q`。该checkpoint为同original-30 geometry support的depth-student
  pure-RL actor，`teacher=false/distill.enabled=false`，无resume或policy
  init；文件SHA256=
  `5501ec25c1079053c6de829ab0e1c7931740f12b5f250f92670592a37b4cfbc4`，
  checkpoint内`saved iteration=34999/next iteration=35000`。
- exact初始化仍是`unscale__any_ball_29`（0.5 kg）、
  `scaledown__any_barrel_25`（1.5 kg）和`unscale__any_bin_29`（1.0 kg），
  逐字节复用原0.15基线的motion、URDF、object map、real mesh、seed42和
  timestep0。协议为1 env/1 G1、randomization disabled、initial-pose
  noise 0、501 steps@50Hz、0.2秒prepend、2.0秒append，关闭adaptive/T1
  及reset；`consecutive_steps=0`只取消debounce，object相对初始world-z
  首次达到+0.30米后才启用`[0.5,0,0,0]`。这里0.5是robot-heading-frame
  的恒定相对pose target（米），不是0.5 m/s速度。
- ball/barrel/bin分别在metric step 77/81/82（1.61/1.69/1.71秒）触发，
  与0.15基线完全一致。直到各自触发行，robot/object simulator state、
  action、motion clip index和motion timestep均逐元素相同；首个action
  分叉分别只出现在下一行78/82/83，因此本次确定性样本唯一干预是command
  从0.15改为0.5。
- 0.15基线为3/3 pickup、3/3 strict-end-carry、平均post-trigger carry
  fraction 1.0；0.5仍为3/3 pickup，但仅0/3 strict-end-carry、平均carry
  fraction 0.1347，并出现明显旋转、跌倒和丢物。三物体0.5的post-trigger
  carry fraction分别为0.1061/0.1619/0.1360；这说明该固定checkpoint和
  三个固定初始化下0.5已落入明显不稳定区，不应外推为完整分布成功率。
- 轨迹继续以G1在t0的world XY为原点，G1 root为cyan、object为orange；
  已生成0.15-vs-0.5的3×2总览、0.5三物体3×1、逐物体两列、逐物体0.5
  trajectory及raw rollout，并安全复制11个主要文件到
  `/home/ubuntu/FAR/_check_vis`。本地审计根为
  `/home/ubuntu/FAR/holosoma_runs/original30_representatives_6urn35000_command050_consecutive0_20260731_052735`。
- recording/analysis/video/media-validation/delivery manifest SHA256依次为
  `4330b4d3b22d4e30e3542dd96ee69ae9e0bd14a54d9a2df84858edce6ea61062`、
  `102181a9778d2f6dd0be65d1a19057c87d0836b61e7fd485471e005da4a3c79a`、
  `7ffdcd12417fd9ef5e8d9b538bdc9c61ae1d85738434f0e5f2c8bd8ef03dcbee`、
  `6762f562c246b4c72318d1f83232b4b81fa81ee4a55b1a79f7230e4a57356b0c`、
  `55db8e93d8fb96352fa496521f47e4f044793ce9cced1556a0a8b64bae996d9e`。
  14个MP4均为单一H.264/yuv420p stream、50fps、501帧、10.02秒，
  ffprobe和完整decode通过；contact sheet人工确认正确ball/barrel/bin、
  单G1、连续动作及动态XY轨迹。本轮没有写入W&B或改变任何run lifecycle。

## [无效且已停止] original30错误forward-task的32卡pure-RL（2026-07-31 06:49 UTC）

- 纠错：该run虽然是fresh pure RL而不是hybrid/distillation，但它错误地把
  用户的“只替换policy command input”实现成了post-lift forward-task
  reward和特殊termination；因此它不是用户要求的“command input改变、
  reward仍全程reference tracking”实验，禁止resume或作为该实验的结果。
  exact session已只在四个目标节点停止；W&B现为`finished`，并写入
  `training_target_reached=0`、
  `stop_reason=invalid_contract_forward_rewards_instead_of_tracking_rewards`和
  `superseded_by=zihanw22/carry-any/xm0hda83`。
- 这个错误实验的正式run为
  `zihanw22/carry-any/4q7mibrm`，name=
  `student_pure_rl_forward_after_lift_original30_ws32_e1024_20260731_060702`，
  URL=`https://wandb.ai/zihanw22/carry-any/runs/4q7mibrm`。四节点分别为
  `10.99.1.134/10.99.1.154/10.99.0.116/10.99.0.117`，每节点8张L40S、
  每rank 1024 env，共32 ranks/32768 env；exact tmux均为
  `hs_4q7mibrm`。不得把contact数据目录名中的teacher provenance或
  disabled-teacher sentinel误报成加载了teacher。
- 算法合同为actor/critic/optimizer/RNG/iteration全部fresh：
  `training_regime=pure_rl`、`distill.enabled=false`、teacher=false、
  policy-init=false、training-resume=false、PPO coefficient恒为1，
  BC/distill/replay-BC均为0。W&B `resume=must`只重开训练前预绑定的fresh
  run identity。actor为depth perception encoder加
  `[512,256,128]` MLP，actor/critic LR=`1e-3`；24 steps、7 epochs、
  4 minibatches、target 40000 updates、checkpoint/ONNX每1000 update，
  upload未关闭。
- command对所有env相同：实际pickup latch前固定`[0,0,0]`，latch后固定
  robot body-frame`[0.5,0,0]`，不计算reference-to-robot target offset，
  不使用动态反馈。pre-lift保留reference tracking以学习approach/pickup；
  post-lift使用`vx=0.5`、`vy=0`、heading、cross-track、yaw-rate和
  robot-relative object pose hold reward，offline contact guidance权重为0。
  post-lift termination放开reference平移/动作偏离，但保留倾倒与丢物安全门。
- exact 30-motion immutable view为
  `data/ds_as_data/debug30_original_realmesh_cominertia_categorymass_v2_scientific_single_slot__src_1f3966245545689a2e14909dda31d6673790bdf888168362449a0ac1902f3510`，
  含5 ball、1 barrel、24 bin，single-view/object-map/transition digest分别为
  `1f396624...2f3510`、`412e86f3...c9e742`、`ab843902...3442`。
  30 clips小于32 ranks时，rank sharder现在使用确定性duplicate-fill和
  inverse-cover loss weight：28条覆盖一次、2条覆盖两次，每rank恰好1条，
  全局每clip质量仍精确为1/30；rank source/manifest SHA256分别为
  `b2ec7b56...c359b`和`d5a95e87...2caf`。
- immutable source=
  `src-6267fe626038c76a5662189c8a963e8413b460c44cfeeec43e0c980d4ab5bfed`，
  archive SHA256=`18654248...ce18`。run contract/worker SHA256分别为
  `00b39647...29ed`、`aeb653cb...c4de`；本地审计根为
  `/home/ubuntu/FAR/holosoma_runs/formal_student_pure_rl_forward_after_lift_original30_ws32_e1024_20260731_054647`。
- Rule-90使用最终source/view canonical第一条
  `scaledown__any_ball_24`重新做policy-free reference replay：
  randomization disabled、timestep0、单机器人、real mesh，
  H.264 1280x720@50fps、329帧/6.58秒、1,326,970 bytes，SHA256=
  `e440f4cb...4655`。ffprobe、完整decode和九帧人工检查均通过。训练history
  建立后，同一字节在step39写入唯一`vis/replay` row；live primary summary
  已绑定该history path，远端重下载复算SHA/size后才删除prebind副本。
  fresh API确认run为running、history row=1、MP4=1。
- exact 4-node/32-GPU canary先完成2个update：总timesteps 1,572,864，
  iteration time 12.61/10.07秒，七个pure-RL reward均非零，
  `ppo_coeff=1/distill_loss=0`。正式run截至06:52四节点严格同步达到
  iteration104；W&B step104的`Perf/total_fps=127,534`、collection/
  learning time=`4.122/2.044s`、mean reward=`3.3831`，七个pure-RL
  reward均finite。四节点各8个compute app，首个progress后的
  Traceback/ChildFailed/CUDA OOM/NCCL fatal/segfault/nonfinite均为0。
  这是启动与早期数值健康验收，不是最终policy-quality结论。
- sharder完整回归70项、pure/hybrid定向回归8项、相关unit 119项、PPO
  493项、perception 46项及launcher定向合同均通过；正式启动后又重跑
  pure-RL/rank-shard/perception/USD-cache高风险组合132项全部通过，
  worktree/index的`git diff --check`均通过。巨型aggregate launcher曾因
  重复启动测试实例而主动终止，不能伪称该aggregate完成；动态目标合同、
  exact canary和formal live run均已覆盖本次路径。

## kdw pure-BC 7K与无效4q task-RL 6K的三物体0.5输入对照（2026-07-31 17:10 UTC）

- fresh W&B API在录制前选择每条run数值上最新的完整checkpoint：
  `zihanw22/carry-any/kdw7jhze/model_07000.pt`（checkpoint内
  `iter=iteration=6999/next_iter=7000`，127,235,828 bytes，SHA256=
  `9dc042736aaef675f3b442c549c2a76db4f23593dd5ed12fa1dee8a772a39525`）
  与`zihanw22/carry-any/4q7mibrm/model_06000.pt`（5999/6000，
  10,220,193 bytes，SHA256=
  `80921bb9534391893d737b907fc7c48eeceb19c2a959f30772d6d0811d282b8e`）。
  17:09 UTC录制与编码结束后fresh recheck仍分别只有1K--7K和1K--6K，
  两条run均为`running`；不能用其在线summary step 7501/6403冒充checkpoint。
- kdw 7K仍是fresh pure-BC depth student：PPO coeff=0、BC coeff=1、无
  replay、teacher action mix=0，训练label来自`mlgjus6q/model_30000`并
  clip到`[-8,8]`；evaluation只运行checkpoint student actor。4q 6K是fresh
  pure-RL depth student：`distill=false/teacher=false`、无resume或policy
  init，但它是上节已标记无效的forward-task合同：pickup前actor command
  为`[0,0,0]`、pickup后为`[0.5,0,0]`，同时替换了post-lift reward和
  termination。该三维actor slot的代码语义是`[dx,dy,dyaw]`，不能再标成
  `vx=0.5 m/s`。两checkpoint各自认证的30-mesh geometry identity集合完全
  相同，motion-transition SHA256同为
  `ab8439024b6e619f59638fc967734433ab08cec607698433c2641b75b7163442`。
- 为和上一轮保持可控，本次对两个actor使用同一个evaluation-only外部门：
  ball/barrel/bin均从exact timestep0开始，object world-z首次达到初始值
  `+0.30 m`且`consecutive_steps=0`后，把actor command slots置为
  `[0.5,0,0]`。两个checkpoint收到的都是同一组三维sparse-root actor
  slots；4q的数值与其native训练command一致，但它训练时额外使用了错误的
  forward-task objective，本轮触发门则是统一的外部world-z门。
- exact初始化继续为`unscale__any_ball_29`（0.5 kg）、
  `scaledown__any_barrel_25`（1.5 kg）、`unscale__any_bin_29`（1.0 kg）；
  motion/URDF/object map/real mesh、seed42、task ID、timestep0、501 steps@
  50Hz、camera、zero pose noise、disabled randomization/reset均与上一轮一致。
  三checkpoint逐clip的501-step motion timeline完全相同，SHA256均为
  `775d4bdca62b67b69eb629e0cb677099f8a6226db2f5a53c3b0ca6f5808e8074`。
- kdw 7K与4q 6K在这三个固定诊断clip上均为0/3 pickup、0/3
  strict-end-carry；最大object相对z增量分别为kdw
  `0.0112/0.0763/0.0487 m`、4q `0.0332/0.0465/0.0673 m`，所以六条
  rollout的0.5 command均从未激活。这批结果只能说明早期checkpoint的
  autonomous approach/pickup尚未通过，不能当成post-lift 0.5行走结论。
  作为上下文加入的6urn 35K旧列为3/3 pickup但0/3 end-carry。
- 六条新rollout的87,174个action标量全部finite。4q三条max-abs不超过
  10.348；kdw ball在跌倒后的step266开始出现百万级unclipped actor输出，
  step300达到2,594,334.25（此时command仍为0且物体未抬起）；kdw barrel/
  bin max-abs为25.025/8.272。该值是checkpoint actor的实际有限输出，不是
  teacher-label clip结果；它是单个固定clip的明显不稳定诊断，不能外推整条
  run的总体分布。
- 生成了新两checkpoint的3×2、连同6urn旧列的3×3、两个3×1、六个逐物体
  对照、九个动态G1/object XY panel及六条raw rollout。本地审计根为
  `/home/ubuntu/FAR/holosoma_runs/original30_representatives_kdw7000_vs_4q6000_command050_consecutive0_20260731_165214`；
  25个final MP4全部为单一H.264/yuv420p、50fps、501帧、10.02秒，ffprobe
  与完整decode通过；两张六时刻contact sheet人工确认正确物体、单G1、
  连续动作、动态轨迹且无default-pose/错物体替换。16个主要视频已逐字节
  复制并复核到`/home/ubuntu/FAR/_check_vis`。
- checkpoint-selection/recording/analysis/video/media-validation/
  W&B-latest-recheck/delivery manifest SHA256依次为
  `8d297208add3a75d1874bb794a775f4408827e971a2200760447e4e652e25059`、
  `73665a705ab42e96682b5f29b790dff428a8048bcbeb976b24bb8231ad42f7b5`、
  `4dd646a8d92799e9b713d8ed2f591f92b3b4649c46f345673fc92b8667907a9a`、
  `4d52b0fb3e2f98b9707925bbd6a286b054941212354419a82139fb680c995848`、
  `8afa014c1ffe9d7c7084771b5de474ba3a8d5e3ae478f6f2f052c36c8313e1ee`、
  `e348a5ab3f827990a14b3af930ff8bdac288423d658052f91e53027146449f80`、
  `78a470bafcd582f691a79d9175dda757c0e1888f38f0fde77136f801d70577c1`。
  录制wrapper遇到Isaac完成输出后shell先退出而五个simulator child仍留在
  cleanup的问题；只按本轮exact process group清理，未触碰训练或其他进程。
  最终GPU3/4/5均回到约906 MiB/0%，没有W&B写入或lifecycle修改。

## original30仅替换policy command input的32卡tracking pure-RL（2026-07-31 18:06 UTC）

- 用户最终确认的唯一干预是actor的三维sparse-root command input：runtime
  pickup/lift latch前固定`[0,0,0]`，latch后sample-and-hold为
  `[0.5,0,0]`；其代码语义仍是`[dx,dy,dyaw]`（m、m、rad），不是velocity
  reward或0.5 m/s。reference tracking、motion timeline、critic输入、drop
  button和generalist `BadTracking` termination全程保持原行为。只有此前已
  明确要求关闭的`offline_contact_guidance.weight=0`；物理contact、
  undesired-contact penalties以及全部原tracking reward继续存在。
- 正确正式run为`zihanw22/carry-any/xm0hda83`，name=
  `student_pure_rl_tracking_original30_cmd000_to050_ws32_e1024_20260731_171611`，
  URL=`https://wandb.ai/zihanw22/carry-any/runs/xm0hda83`。节点为
  `10.99.1.134/10.99.1.154/10.99.0.116/10.99.0.117`，master=
  `10.99.1.134:32201`，exact session均为`hs_xm0hda83`；4 nodes x 8 L40S
  =32 ranks，每rank/GPU 1024 env，全局32768 env。没有停止或修改其他job。
- 算法是fresh pure PPO：actor/critic/optimizer/RNG/iteration全部fresh，
  `training_regime=pure_rl`、`distill.enabled=false`、teacher/policy-init/
  training-resume均false、`ppo_coeff=1`，BC/distill/replay-BC恒为0。actor为
  94D scalar加32D depth encoding、MLP `[512,256,128]`、29D action；
  actor/critic LR均为`1e-3`，24 rollout steps、7 epochs、4 minibatches、
  target 40000 updates，PT+ONNX每1000 update且upload未关闭。
- reward preset为
  `g1_29dof_wbt_reward_w_object_generalist_tracking_no_contact`：六项body
  tracking、两项object tracking及原regularization/rest terms，与原
  generalist offline-contact tracking tree仅有offline contact权重归零的
  差异。不存在forward/lateral/heading/cross-track/yaw-rate/object-hold
  task reward，也不存在post-lift tracking mask或特殊task termination。
- exact immutable 30-motion view为
  `debug30_original_realmesh_cominertia_categorymass_v2_scientific_single_slot__src_1f3966245545689a2e14909dda31d6673790bdf888168362449a0ac1902f3510`，
  含5 ball、1 barrel、24 bin；object-map SHA256=
  `412e86f38c3376b456d5f97e58876f6189eff1345da92e9903b9752030c9e742`，
  transition digest=
  `ab8439024b6e619f59638fc967734433ab08cec607698433c2641b75b7163442`。
  32个rank shard均非空：28条一次、`scaledown__any_bin_16`和
  `unscale__any_bin_16`各两次，通过inverse-cover保持uniform-clip目标；
  source/manifest SHA256分别为`b2ec7b56...c359b`和
  `937e0d74c612b1a851e533b43127889a3a666afe4ff27505c389835b77079038`。
- immutable source=
  `src-e13aa89e08887813e5470f1659dcb0aba83c7e343338053c7e0f6e0a2b878bdc`，
  archive/source-manifest SHA256分别为
  `4bda4f9267e3271283a2f8160cfbdedecdfb6063612597fe62c588a62418e10b`和
  `e13aa89e08887813e5470f1659dcb0aba83c7e343338053c7e0f6e0a2b878bdc`；
  worker/run-contract SHA256分别为
  `afe8260021b9f4c004161624476de82872c50f898fbee5639660f108e54466b4`和
  `77cfc3229db661a1c7b9474b81b841b4bce87402b26e0a5ced1c3a89dd5ddf18`。
  审计根为
  `/home/ubuntu/FAR/holosoma_runs/formal_student_pure_rl_tracking_original30_cmd000_to050_ws32_e1024_20260731_171611`；
  `formal_acceptance.json` SHA256=
  `39225c6e36a3aa4ef50f107708564137063d790f998387eafbec61c383fbf82f`。
- Rule-90使用最终source/view canonical第一条`scaledown__any_ball_24`的
  fresh policy-free reference replay：single robot、real mesh、
  randomization disabled、H.264 1280x720@50fps、329帧/6.58秒、
  1,330,006 bytes，SHA256=
  `d02551455d0f1791e0fb072844ce2d5f9c99dcf621dc183fd8229c4738290ace`。
  ffprobe、完整decode及14帧人工检查通过；训练history出现后，同一字节只在
  step9写入一条`vis/replay`，live primary summary绑定该history path后删除
  prebind副本。fresh API确认history row=1、MP4=1、远端SHA/size一致；
  replay manifest SHA256=
  `9c5f0e8d8739031869a5ebb0065fe67c412f159ef0ee8ba7f5e3a37f0eb738a2`。
- exact 32-GPU canary完成2个update，iteration time为12.43/9.77秒，
  throughput为63,274/80,530 steps/s。正式run四节点严格同步通过iteration
  53的有界验收，均为8个compute app且无exit；首个progress后的
  Traceback/ChildFailed/CUDA OOM/NCCL fatal/segfault/nonfinite为0。
  fresh W&B step52仍为`running`：`Perf/total_fps=103,007`、collection/
  learning time=`5.591/2.044s`、mean reward=`1.4775`、bad-tracking frac=
  `0.04422`，`ppo_coeff=1`且全部BC/distill loss为0。focused 8项和更广的
  142项定向测试、compileall及worktree/index diff check均通过；这是启动与
  早期数值健康验收，不是最终policy-quality结论。

## original30 reset-only hybrid-velocity 8卡隔离实验（2026-07-31 21:42 UTC）

- 新能力全部由默认false的`hybrid_velocity_*`字段控制，旧experiment/
  command/observation/reward/termination preset不变；正式入口仅为新增的
  `train_hybrid_velocity.sh`和对应独立preset。actor始终收到统一的
  `[vx,vy,yaw_rate]`：tracking row直接读取NPZ的`body_lin_vel_w`/
  `body_ang_vel_w`并转到当前heading，不再有限差分；task row在真实pickup
  latch前为`[0,0,0]`，之后sample-and-hold为`[0.5,0,0]`。task/tracking身份
  只在episode reset改变，fraction从iteration 0的0线性升到iteration 5000
  的0.5。
- actor不接收task ID；critic接收task indicator，并在task row屏蔽五个仅
  reference有效的target channel。tracking row保留原tracking reward和
  termination；task row从episode起点使用lift/carry/velocity/heading/
  cross-track/object-hold reward，不混入隐藏的reference tracking reward。
  offline contact guidance权重为0，但正常物理contact和原randomization继续
  生效。
- 正式run为`zihanw22/carry-any/us9ogral`，name=
  `student_hybrid_velocity_original30_ws8_e1024_20260731_200744`，URL=
  `https://wandb.ai/zihanw22/carry-any/runs/us9ogral`。节点`10.99.0.167`，
  tmux=`hs_hybrid_velocity_us9ogral`；1 node x 8 L40S、每rank 1024 env，
  requested/effective global env均为8192。fresh pure PPO，distill/teacher/
  resume/policy-init均false，actor/critic MLP均为`[512,256,128]`，PT+ONNX
  每1000 update保存。受保护run `xm0hda83`及其四节点/session未被修改或停止。
- exact 30-clip original real-mesh single-slot view保持5 ball、1 barrel、24
  bin；visual/depth使用真实mesh，collision使用convex decomposition，object
  fallback关闭。rank shards按clip依赖闭包覆盖30条恰好一次，clip数为
  `[4,4,4,4,4,4,4,2]`，通过rank-local loss correction使全局权重和恒为8；
  shard manifest SHA256=
  `5535bfe22e2745f79816bd47d3053fd21d27a10cbd1346b19f59c9eec34a0c4b`。
- immutable source snapshot为
  `src-21c824f7677d4abf42a859f9606ddedb06669183772d510af4e90e424aed4ee8`；
  runtime asset manifest SHA256=
  `cd794f5e242bcc84a8a4e6b3dd092a93820c907468d551a2cdcaa4c370dd510f`。
  Rule-90 fresh policy-free canonical replay为`scaledown__any_ball_24`，H.264
  1280x720@50fps、329帧/6.58秒，SHA256=
  `bc7f15ba58e54bf0d595f84431d619c4d4b8f95de7dbbfa7fe27c25607f2bcca`；
  ffprobe、完整decode、人工抽帧和fresh W&B远端SHA/size复核通过，绑定为
  `vis/replay`。
- exact 8-GPU canary的flat NCCL gradient all-reduce在update 1后的minibatch
  5发生rank4/其余rank collective错位死锁；py-spy确认后，仅该隔离launcher
  改为CPU/Gloo sum再除world size，随后10/10 update通过并产出可校验PT和
  ONNX。正式run在有界验收窗口完成至少48个controller update，W&B已到
  step57并仍为running；近期iteration time为5.44-7.08秒，8/8 compute app、
  UECC=0，finite marker和global loss-weight-sum marker均48/48，首个progress
  后hard error为0。task active fraction已到约0.4%，task reward出现非零值且
  task rows中约55%-60%已触发lift，证明新分支实际执行；BC/distill三项恒0。
- 第一次formal启动被preflight拒绝，因为四个semantic env var晚于provenance
  生成；该尝试在simulator/model/history/checkpoint之前退出，W&B只有预绑定
  replay。修正worker顺序并加入semantic-binding self-check后才使用同一fresh
  identity正式启动。启动阶段仍有headless rank-visible renderer的`No device
  could be created`消息，但PhysX scene、8个worker和后续更新均正常；不得把
  该启动消息写成不存在。审计根为
  `/home/ubuntu/FAR/holosoma_runs/formal_student_hybrid_velocity_original30_ws8_e1024_20260731_200744`；
  `formal_acceptance.json` SHA256=
  `583db0e4528002d232934805b7155c7bb4e6924a7e97d52322639ac4ec38448f`。
- 定向测试`test_hybrid_velocity.py`、`test_hybrid_stage2.py`、
  `test_as_rank_shards.py`共83项通过，并完成launcher bash syntax、compile和
  snapshot/diff语义复核。当前验收只证明实验合同、启动和早期数值健康，
  不等同于最终policy quality结论。

## kdw pure-BC 10K的original30三物体单机器人评估（2026-07-31 22:50 UTC）

- 用户指定评估`zihanw22/carry-any/kdw7jhze/model_10000.pt`。fresh W&B
  API在录制前后均确认远端完整checkpoint扫描为1K--10K，run仍为`running`；
  不能用在线summary step 10,075/10,127冒充checkpoint。10K文件为
  127,235,828 bytes，MD5与W&B一致，SHA256=
  `c28d3d6ff3de27cac648981b0d32d906036a7989a1e5a603da8b699770861b70`；
  checkpoint内部`iter=iteration=9999/next_iter=10000`，341个tensor、
  31,758,432个值全部finite。
- 该checkpoint仍是fresh pure-BC depth student：PPO coeff=0、BC coeff=1、
  无replay、teacher action mix=0；训练label来自`mlgjus6q/model_30000`并
  clamp到`[-8,8]`。evaluation显式运行10K student `checkpoint_actor`本身，
  没有运行teacher或把clipped label直接送入环境。30个authenticated object
  mesh与已验证6urn 35K support完全相同，无training resume或policy init。
- exact初始化仍为`unscale__any_ball_29`（0.5 kg）、
  `scaledown__any_barrel_25`（1.5 kg）、`unscale__any_bin_29`（1.0 kg）。
  motion/URDF/object map/real mesh、seed42、timestep0均与1K/7K对照逐字节
  对齐；1 env/1 G1、randomization disabled、initial-pose noise 0、501 steps@
  50Hz、0.2秒prepend、2.0秒append并关闭adaptive/T1/reset。
  `consecutive_steps=0`只取消debounce，仍需object world-z达到初始值+0.30米
  才启用heading-frame relative-pose`[0.15,0,0,0]`。三条timeline SHA256均为
  `775d4bdca62b67b69eb629e0cb677099f8a6226db2f5a53c3b0ca6f5808e8074`。
- 10K在ball/barrel/bin上的最大object相对z增量分别为
  `0.0143/至少0.3004/0.0444 m`；只有barrel在2.69秒触发。最终结果为
  1/3 pickup、1/3 strict-end-carry：barrel的post-trigger carry fraction=1、
  object XY displacement=6.833米、末端robot-object XY距离=0.274米；ball与
  bin command全程保持0。三条max-abs action分别为7.839/7.184/8.240，全部
  finite。相同固定初始化下7K为0/3 pickup，因此10K在barrel上有明确改善，
  但不能外推整个bank或run的总体成功率。
- 审计根为
  `/home/ubuntu/FAR/holosoma_runs/original30_representatives_kdw10000_purebc_consecutive0_20260731_224007`。
  三条raw、三条动态G1/object world-XY trajectory和一个3x1总览共7个MP4，
  均为单一H.264/yuv420p、50fps、501帧/10.02秒；逐文件ffprobe和完整decode
  通过。六时刻3x1 contact sheet人工确认单G1、正确ball/barrel/bin、动作与
  cyan/orange轨迹连续，无default-pose或错物体替换。
- 本地交付严格拆为`/home/ubuntu/FAR/_check_vis/07-31-2248__kdw7jhze_model10000__raw`的3条raw和
  `07-31-2249__kdw7jhze_model10000__trajectory`的3条trajectory加1个3x1总览；7份copy的size/SHA与审计源
  逐项一致。checkpoint-selection/recording/analysis/video/media-validation/
  W&B-recheck/delivery manifest SHA256依次为
  `4960394a1eed31013fcc7b9c890feff557084d63c845f235ebaa2fc90eb65267`、
  `e4b152581cc47ff8cffd124a2557d33fb14fe5d75b9ae9983ce206d4e4d30ea7`、
  `fa8cc137612f4f151a234b76c94850d59addc5c31db099975336a9ba29d41259`、
  `21d849b233be3c0d6f3a30d94ed67a77e03f779807163066d40839bed4fa46bf`、
  `808d292e950c9fe0d02e95b2a6d74697410ea4da458702093b99f0830a104cf0`、
  `a1dab4a1cf1f4a5e856722771245eb4c6669082468a12b84842acc2303fb587b`、
  `b170603b9f1a83c5ca1709e118b95a9400f636c40f94b455b3d21bc83b40a72e`。
  本轮只读W&B、没有上传media或改变run lifecycle；GPU3/4/5均恢复约906MiB/
  0%，所有录制、编码和校验进程已退出。

## kdw pure-BC 10K与xm corrected pure-RL tracking 3K的同门0.5对比（2026-07-31 23:56 UTC）

- 用户要求直接比较`zihanw22/carry-any/kdw7jhze`与
  `zihanw22/carry-any/xm0hda83`。fresh W&B API在录制前把数值上最新的
  完整checkpoint锁定为`kdw7jhze/model_10000.pt`与
  `xm0hda83/model_03000.pt`；录制后再次扫描仍分别只有1K--10K与1K--3K，
  两条run均为`running`。kdw文件为127,235,828 bytes、SHA256=
  `c28d3d6ff3de27cac648981b0d32d906036a7989a1e5a603da8b699770861b70`、
  checkpoint内`iter=iteration=9999/next_iter=10000`；xm文件为
  10,217,569 bytes、SHA256=
  `36fd3a6fa8d92743d2a1d52fd6f733869e6e8d25bd7c99302b9b0c840344cae6`、
  内部`2999/2999/3000`。两者认证的30个object mesh集合逐SHA完全相同，
  support-set SHA256均为
  `b238d027af1014225d2422b89139c1817dbe3dc4c22951559897edc83a43089b`。
- kdw 10K仍是pure-BC student checkpoint actor：PPO恒0、BC系数1、无replay，
  teacher只产生clamp到`[-8,8]`的训练label；本轮没有运行teacher。xm 3K是
  corrected fresh pure PPO tracking actor：`distill=false`、teacher/resume/
  policy-init均false、PPO系数1，reward仍是reference body/object tracking且
  offline contact guidance权重0。两者actor的三维command槽语义都为当前
  heading frame中的relative pose`[dx,dy,dyaw]`，不是速度。
- 为做同条件A/B，两个checkpoint均使用同一个evaluation-only外部门：object
  world-z第一次达到初始值`+0.30 m`且`consecutive_steps=0`时，将actor输入
  从`[0,0,0]`切为并持续保持`[0.5,0,0]`。manual evaluation input优先于xm
  checkpoint内置的pickup latch，因此两个actor实际看到的数值、触发标准和
  command语义一致。exact clip仍为`unscale__any_ball_29`（0.5 kg）、
  `scaledown__any_barrel_25`（1.5 kg）、`unscale__any_bin_29`（1.0 kg）；
  motion/URDF/object map/mesh、seed42、timestep0、randomization disabled、
  zero pose noise、501 steps@50Hz、prepend/append、camera及reset开关逐项相同。
  六条完整motion timeline相同，SHA256仍为
  `775d4bdca62b67b69eb629e0cb677099f8a6226db2f5a53c3b0ca6f5808e8074`。
- kdw在ball/barrel/bin上的最大object相对z分别为
  `0.0143/0.4895/0.0444 m`，只有barrel在2.69秒触发，因此为1/3 pickup、
  0/3 strict-end-carry。barrel在0.5输入下post-trigger carry fraction仅
  `0.1622`、object XY位移3.253米并最终跌倒（末端root z=0.0787米）；这与
  上一批相同10K但0.15输入下barrel持续carry的结果不同，不能把两档混写。
  xm在ball/barrel/bin分别于1.65/1.63/1.67秒触发，最大相对z为
  `0.4856/0.4918/0.5289 m`，是3/3 pickup但仍为0/3 strict-end-carry；三条
  post-trigger carry fraction为`0.3815/0.3853/0.3753`，object XY位移为
  `1.834/2.003/1.833 m`，后段均失持。全部87,174个action值finite；kdw/xm
  三clip最大绝对值分别为10.057/8.966。固定三个诊断clip只能说明xm 3K早期
  pickup明显优于kdw 10K，不能外推整个30-clip bank的robustness。
- 最终生成6条raw、6条动态G1/object world-XY panel、3条逐物体两列、两个
  单策略3x1和一个3x2总览，共18个MP4。全部为单一H.264/yuv420p、50fps、
  501帧/10.02秒；raw为640x360、逐物体两列为1280x560、总览为
  1280x1680。逐文件ffprobe与完整decode通过；六时刻3x2 contact sheet人工
  确认正确ball/barrel/bin、单G1、动作和cyan/orange轨迹连续，无default-pose
  或错物体替换。18个文件同组交付到
  `/home/ubuntu/FAR/_check_vis/07-31-2355__kdw7jhze_model10000_vs_xm0hda83_model03000`，复制后逐SHA一致。
- 本地审计入口为
  `/home/ubuntu/FAR/holosoma_runs/original30_representatives_kdw10000_vs_xm3000_command050_consecutive0_20260731_233727`，
  实体位于`/data/holosoma_eval_audits/`并由原路径symlink保持可访问。首次尝试
  暴露controller根盘已满：默认`/tmp`与repo-local USD转换cache产生截断USD；
  该批全部fail closed并连同日志/cache保存在
  `failed_attempts/rootfs_full_20260731_2344`，没有混入最终manifest。随后把
  TMP/XDG/robot USD/object USD cache全部按task重定向到`/data`后六条从头
  重录。checkpoint-selection/recording/analysis/video/media-validation/
  W&B-recheck/delivery manifest SHA256依次为
  `131ff7a29d24c40c439aa39b624ac98f59d5ca69a4f320d0ca53072e5455a8fc`、
  `776e14ccdc7f9b441942639adee81a705bb46750b007cfc498cd3af9fdc93c8c`、
  `edfd588188202e3f760643a9588aecd8f6c5631448dd79866f42cb7455105e80`、
  `c37be38295f596f64156286e822b9e99a02cdf0ecd5f8ef40d66856d34793fde`、
  `006e2335ecff203482df6607e6d9a79f74337ffa04fedf6e8587f971a032ace0`、
  `b251b63e04f205c7f9ae8c24c16370b9f0298d65aade0c222ecb2743d2e1958f`、
  `a087c533a683801d012e3b4792303bff6c6d283c5f7e502ad3305a5bc4a37e55`。
  本轮W&B只读、未上传media或改变run lifecycle；GPU3/4/5均恢复约906 MiB/
  0%，没有残留evaluation进程。

## kdw pure-BC 10K与xm corrected pure-RL tracking最新7K的同门0.5对比（2026-08-01 05:55 UTC）

- 用户要求在上一批kdw 10K vs xm 3K基础上只把xm替换为最新checkpoint。
  fresh W&B API于05:47 UTC锁定`xm0hda83/model_07000.pt`；文件为
  10,217,697 bytes、SHA256=
  `2bd36bf002218ca928652ff06ff902874f7e23d066e50367dded53cb122d5a8c`，
  checkpoint内部`iter=iteration=6999/next_iter=7000`且所有tensor finite。
  05:54 UTC交付前复核时xm远端仍只有1K--7K、online summary step为7933，
  因此7K既是选择时最新也是交付前最新完整checkpoint。kdw列有意固定为上一批
  `model_10000.pt`及其SHA，不随已继续训练到13K的run漂移；两条run均仍为
  `running`。
- 这不是teacher或reference replay。kdw列仍是pure-BC student checkpoint
  actor，xm 7K仍是`distill=false`、无teacher/resume/policy-init的fresh pure
  PPO reference-tracking actor。两者认证的30个object mesh support逐SHA相同；
  support-set SHA256=
  `b238d027af1014225d2422b89139c1817dbe3dc4c22951559897edc83a43089b`。
- exact初始化和上一批逐项相同：`unscale__any_ball_29`（0.5 kg）、
  `scaledown__any_barrel_25`（1.5 kg）、`unscale__any_bin_29`（1.0 kg），
  单环境/单G1、seed42、timestep0、randomization disabled、zero pose noise、
  501 steps@50Hz、0.2秒prepend、2.0秒append并关闭adaptive/T1/reset。
  evaluation-only门仍是object world-z首次达到初始值`+0.30 m`且
  `consecutive_steps=0`时，将heading-frame relative-pose输入从`[0,0,0]`
  切为`[0.5,0,0]`；不是velocity command。为保持A/B严格可比，kdw 10K的
  三条raw/metrics按checkpoint、clip、协议和timeline SHA复用，xm 7K三条重新
  从头录制。
- 固定三个clip上，kdw 10K仍为1/3 pickup、0/3 strict-end-carry；仅barrel在
  2.69秒触发，post-trigger carry fraction=`0.1622`。xm 7K为3/3 pickup、
  0/3 strict-end-carry：ball/barrel/bin分别在1.63/1.69/1.73秒触发，carry
  fraction为`0.8085/0.3810/0.3828`，object XY位移为
  `4.024/2.082/1.874 m`。相对xm 3K，ball的carry fraction从`0.3815`
  增至`0.8085`且object XY位移从`1.834`增至`4.024 m`；barrel/bin基本持平。
  但三条末段仍未满足strict carry，所以不能宣称7K已实现3/3完整搬运，也不能
  由三个固定诊断clip外推整个30条bank的robustness。
- 交付包含6条raw、6条动态G1/object world-XY panel、3条逐物体两列、两个
  单策略3x1和一个3x2总览，共18个MP4，统一放在
  `/home/ubuntu/FAR/_check_vis/08-01-0554__kdw7jhze_model10000_vs_xm0hda83_model07000`。全部为单一H.264/yuv420p、50fps、
  501帧/10.02秒，逐文件ffprobe与完整decode通过；六时刻contact sheet人工
  确认正确ball/barrel/bin、单G1、动作及cyan/orange轨迹连续，无default-pose
  或错物体替换。18份交付copy的size/SHA与源逐项一致。
- 审计入口为
  `/home/ubuntu/FAR/holosoma_runs/original30_representatives_kdw10000_vs_xm7000_command050_consecutive0_20260801_054614`，
  实体位于`/data/holosoma_eval_audits/`。checkpoint-selection/recording/
  analysis/video/media-validation/W&B-recheck/delivery manifest SHA256依次为
  `562857e79e5dadfdcea1868b637425eea961b8be9e2f6841a713db2076800d94`、
  `548d04d8bca4b0d827fb9f025c183ba700961853ea97be7255031dab370c85cf`、
  `37aa714e0b1c37e164b703e255a51574cc2f8a0df2449caf2ae59648e931f79e`、
  `0a36e6ded08cae12c649463e0020335624303ca452e995324bd4635eea8310b8`、
  `13ebe2363880c787724f5db9453e909cff392d8e4741d1c151e85cdf4cf105a1`、
  `e5124268cc2fe52445c149a5d4e85d93be6040f28d538057fc709df6b95e896f`、
  `a6bb26e716623cb8308f5cb07729191aacad906c17c93dc36945e48244830859`。
  本轮W&B只读、没有上传media或改变run lifecycle；GPU3/4/5均恢复约906 MiB/
  0%，没有残留evaluation进程。

## CORL79 global-command A/B/C正式训练交接（2026-08-01 08:39 UTC）

- 用户要求同时比较三种deployment command语义，三条都是fresh pure PPO，
  不是distillation/hybrid-BC，也没有teacher、training resume或policy init。
  `logger.resume=must`只用于连接训练前已绑定Rule-90 replay的fresh W&B ID，
  不读取任何模型、optimizer、iteration或RNG state。三条actor/critic MLP均为
  `[512,256,128]`，actor/critic初始LR均为`1e-3`，24 steps/env、7 epochs、
  4 minibatches、40,000 updates；PT和ONNX每1,000 updates保存并上传，
  `reset_rollout_at_checkpoint=False`。
- A是canonical per-frame global velocity输入：直接读取motion NPZ已有的root
  `body_lin_vel_w[:2]`与`body_ang_vel_w[z]`，不做finite difference或robot
  heading rotation，因此前进、后退、侧移和转弯逐帧保留。W&B为
  `zihanw22/carry-any/huipupp1`：
  <https://wandb.ai/zihanw22/carry-any/runs/huipupp1>；name=
  `policy_corl79_world_velocity_ws64_e1024_20260801_062031`。拓扑为
  ap-northeast-2a的8 nodes x 8 L40S、每rank 1024 env、global 65,536 env；
  节点依次为`10.99.0.244/.24/.39/.54/.61/.180/.183/.201`，tmux均为
  `hs_huipupp1`。
- B是world-axis hybrid velocity对照。tracking rows收到与A相同的canonical
  per-frame global velocity；task rows在pickup latch前收到`[0,0,0]`，之后
  固定收到world `+X [0.5,0,0]`，不会根据reference或robot反馈改写。
  task fraction只在reset时切换并从iteration 0的0线性升到iteration 5000的
  0.5；对应task reward也使用world X/Y/yaw轴。W&B为
  `zihanw22/carry-any/s3w5pl35`：
  <https://wandb.ai/zihanw22/carry-any/runs/s3w5pl35>；name=
  `hybrid_corl79_world_velocity_forward_ws32_e1020_20260801_062031`。拓扑为
  ap-northeast-2a的4 nodes x 8 L40S、每rank 1020 env、global 32,640 env；
  节点`10.99.0.141/.186/.77/.176`，tmux=`hs_s3w5pl35`。
- C是用户要求的互补实验，不使用B的完全forward输入，而是每帧正常计算
  global pose error：`[x_ref_w-x_robot_w, y_ref_w-y_robot_w,
  wrap(yaw_ref_w-yaw_robot_w)]`。它不做heading-frame rotation、velocity
  conversion、finite difference或task override；reward仍是全阶段正常
  reference tracking。W&B为`zihanw22/carry-any/dauhd33b`：
  <https://wandb.ai/zihanw22/carry-any/runs/dauhd33b>；name=
  `policy_corl79_world_root_error_ws32_e1020_20260801_062031`。它严格使用用户
  指定的ap-northeast-2b四组：`zzzihanw-e=10.99.1.21`、
  `zzzihanw-72=10.99.1.89`、`zzzihanw-f=10.99.1.122`、
  `zzzihanw-z=10.99.1.69`；4 nodes x 8 L40S、每rank 1020 env、global
  32,640 env，tmux=`hs_dauhd33b`。
- 三组都使用CORL80俗称但实际79条的exact bank：box 25、ball 4、barrel 35、
  bin 15。A/B的single-slot view为repo下
  `data/ds_as_data/carryany_filter_scale_noscale_keep169_20260513_plus_box_teacher_rollout_success155_bcleb5oi58000_final0p5_primitiveproj_solid80_clean_box_bin_barrel_ball_cominertia_categorymass_v2/_scientific_corl79_single_slot/by-source/c9e02244ac1e3c870564f70837a963b03a337430bb1b4a58dc50610868df8027`；
  C为跨节点统一绝对路径，把同一闭包materialize在
  `/data/logs_new/holosoma_inputs/corl79_worldcmd_20260801/ds_as_data/`下。
  三组均按clip-object依赖闭包做global exact-once rank shard，没有legacy
  asset shard或DataLoader二次shard；A/B/C rank-shard manifest SHA256分别为
  `3c9aaf89462e8da4caabae05748de91f4904475df005a72595e84f7e244eb4c5`、
  `1648901940bea31b36590cc9a01760c6452281b8a0d86711c21e0b44c164d3c9`、
  `8223d34633c65697385435830a0daaa23d0ed39542d3f697347cb7a806dfafe6`。
- visual和perception depth严格使用original real mesh；physics collision对同一
  real mesh使用`convex_decomposition` collider processing。legacy URDF/
  cuboid fallback关闭，79条URDF/mesh/contact闭包在全部16节点逐文件验证；
  offline contact guidance reward权重为0，但真实物理contact仍正常模拟。
  distributed transport为global Gloo、node-local NCCL gradient reduce和
  cross-node CPU leaders；A/B/C distributed loss-weight sum持续为64/32/32。
- immutable source snapshot为
  `src-4887671862ee036da05d70193441a720af39cf9b66d5642c85eb6eed64184cb9`，
  archive SHA256=
  `995322f46f50abd24c9b4ecd29fefed9ecff53095317ed3b05dcb7104df1c23b`。
  launch root为
  `/home/ubuntu/FAR/holosoma_runs/formal_corl79_world_commands_abc_20260801_062031`；
  A/B/C run-contract SHA256分别为
  `b224c59162a304e741e2306dbe8c936eb7e2668710a6699b0cbf3ee1cf69669f`、
  `e2aee0f27d9d18757629c402243653e883b849613560e7e5eca29a56ac7a4a5b`、
  `5468f4b9b5777561298d77cba73d9a8ca55fe4687df2b76bb342ae9848044f04`；
  final worker SHA256=
  `e06d1b6e88d45188f4bebf1f5278632cf34902fd26eb439d47b5575aacdc3385`。
- A/B/C exact-topology canary均完成2/2 update，第二次update约
  `10.21/9.27/9.74s`，全128个preflight marker存在，finite、transport、
  command preset、real-mesh和fallback门均通过。首次formal尝试在update前
  fail closed：node-specific `logger.base-dir`让非master拒绝rank0广播的实验
  目录，明确报`Experiment directory does not match`；其余Gloo peer-closed
  只是下游结果。修复为各节点相同逻辑absolute logger base、但保留节点本地
  controller/W&B/cache路径后重新跑完整canary和preflight；失败尝试没有
  checkpoint、policy state或training history，正式run仍从iteration 0 fresh
  开始。
- 三份Rule-90均来自最终bank canonical首条`box_10`，是policy-free reference
  replay；H.264 1280x720@50fps、368帧/7.36秒，人工抽帧、完整decode和
  ffprobe通过。A/B/C视频SHA256分别为
  `de2c0dcc07f8c5c5786fc8adfa8db23e824fcb4c7e742b65a6812a492c24a42b`、
  `af2f686e01c8d95f9cbf160d642c835ffa02cc1cd3bbe370525264e59d344e8f`、
  `f37a740284edecec635f68caf1e3d2336ac74a81c8ffddf9058cbcd698b736af`。
  训练history出现后，同一字节已写为各run唯一history-backed `vis/replay`；
  history step为73/91/86，primary summary已绑定，prebind副本已在远端逐SHA
  验证后删除。fresh API确认每run恰好1行media history和1个MP4，run仍为
  `running`。
- 08:39 UTC有界终检时，A/B/C全rank分别同步到completed iteration
  `144/168/163`；16/16 tmux alive，每节点8/8 compute app、8/8 unique GPU、
  8/8 exact snapshot worker，exit marker为0。所有controller日志在5秒内更新，
  精确Traceback/ChildFailed/OOM/NCCL/DistBackend/non-finite fatal与geometry
  fallback均为0。最近20个update的iteration-time中位数约为
  `5.79/5.24/5.32s`。这是启动和早期数值健康验收，不是最终policy quality
  结论；验收后不建立常驻监控，也不得为例行检查attach或重启这些run。

## Global-command A硬件失败与W&B可见性恢复（2026-08-01 20:23 UTC）

- 上节08:39的健康快照属实，但A随后在09:22:03 UTC于completed iteration
  588发生独立硬件故障。master `10.99.0.244`的local rank 1、PCI
  `0000:a0:00`、GPU UUID
  `GPU-09ab58b0-98cf-6ecc-58c6-e3f7b27888d7`由内核明确记录
  `NVRM Xid 79: GPU has fallen off the bus`；同节点所有GPU随即记录
  `Xid 154`并要求`Node Reboot Required`。NCCL watchdog首先看到
  `CUDA error: unknown error`后SIGABRT，其他local ranks由torchrun终止；
  其余7节点在global Gloo peer消失后等待到09:52并退出。因此peer-closed、
  NCCL watchdog和SIGABRT都是GPU掉总线的结果，不是本轮hierarchical
  transport、command实现或dataset shard的首因。
- rank0 W&B core在退出前成功上传history 586--587并以exit code 1完成同步；
  本地debug log没有run delete调用。远端A后来被一个独立于训练worker的删除
  动作移除，GPU失败本身不会删除W&B run；当前本地证据不能确定删除动作的
  发起者。B/C没有被删除，20:17 UTC仍分别在约7.8K/7.6K正常运行。
- 已用W&B `undeleteRuns`按exact entity/project/run ID/display name恢复
  `zihanw22/carry-any/huipupp1`。恢复后明确把state校正为`failed`，没有伪装
  为running，也没有重启训练。fresh API复核保留588条训练history，最后有效
  `Train/command_goal_training_iteration=587`、distributed loss-weight sum=64、
  finite marker=1。因为save cadence为1000，故故障前没有PT/ONNX checkpoint，
  A不能从588 exact resume；若继续该对照，必须更换或重启坏节点并建立新的
  fresh run从iteration 0开始。
- W&B undelete保留了summary和MP4但丢失secondary-writer的media history索引；
  已把审核过的同一Rule-90字节重新写成history step 588，并在校正run state后
  重新绑定summary。最终远端恰好1条`vis/replay` history、1个MP4，path=
  `media/videos/vis/replay_588_de2c0dcc07f8c5c5786f.mp4`，size=1,539,679，
  下载SHA256仍为
  `de2c0dcc07f8c5c5786fc8adfa8db23e824fcb4c7e742b65a6812a492c24a42b`。
  恢复脚本保存在launch root的`restore_failed_A_wandb.py`；它不操作任何训练
  PID、GPU、checkpoint或B/C lifecycle。

## 五个running run的0.05/0.15/0.50 command分批诊断录制（2026-08-01 21:27 UTC）

- 用户先要求对`kdw7jhze`当前checkpoint录制0.05/0.15/0.50，随后要求把另外
  四个当时仍在running的run也分批录制。20:31--20:44 UTC的fresh W&B只读
  扫描锁定：`kdw7jhze/model_20000.pt`（127,235,828 bytes，SHA256=
  `19e4f324f729d00a237f6e887ce718df13f3f04eee663b0012dbed0b69e2c55d`）、
  `xm0hda83/model_17000.pt`（10,217,697 bytes，SHA256=
  `2bac311ddd09f1a5e88f9ae79813d989e026eac28d4e3e2336d735e2594cee30`）、
  `us9ogral/model_18000.pt`（8,284,489 bytes，SHA256=
  `c7cdd4c5fe4e90f3202453962640376204773e71fe476f4b37a2a838a874b92a`）、
  `s3w5pl35/model_08000.pt`（10,373,985 bytes，SHA256=
  `9f1483464da866a2a0328824f1d65b75cd1e782ff5cb763eb908f3f430fbd9f6`）及
  `dauhd33b/model_07000.pt`（10,346,465 bytes，SHA256=
  `658f9c1c97869f55703383d91f622de93f33b037ca83fcf96f6fbfcdefa1f778`）。
  每份checkpoint内部iteration、tensor finite、run provenance和训练合同均逐项
  验证；kdw是pure BC student，其余四个是`distill=false`的fresh pure PPO，
  本轮没有teacher rollout，也不是reference-motion replay。
- 共同evaluation协议为单环境/单G1、seed42、timestep0、randomization disabled、
  initial-pose noise 0、501 steps@50Hz、0.2秒prepend、2秒append、关闭adaptive/
  T1/reset。外部门固定为object world-z首次达到初始值`+0.30 m`，且
  `consecutive_steps=0`；触发前actor command三槽严格为0，触发后的第一actor
  step才写入指定值。三档数值相同但语义不能混写：kdw/xm是robot-heading
  relative pose（m）；us9是robot-heading velocity（m/s）；s3是world velocity
  （m/s）；dau是world root error（m）。actor原始输入逐step sidecar和29维action
  均验证finite；evaluation override默认关闭，只在本轮显式CLI启用。
- kdw/xm/us9使用exact original30三个初始化：`unscale__any_ball_29`（0.5 kg）、
  `scaledown__any_barrel_25`（1.5 kg）、`unscale__any_bin_29`（1.0 kg），view digest=
  `1f3966245545689a2e14909dda31d6673790bdf888168362449a0ac1902f3510`。
  s3/dau的训练bank为CORL79，为避免把OOD物体冒充公平对照，改用同一bank内
  `noscale__any_ball_3`（0.5 kg）、`noscale__any_barrel_12`（1.5 kg）、
  `noscale__any_bin_32`（1.0 kg），view digest=
  `c9e02244ac1e3c870564f70837a963b03a337430bb1b4a58dc50610868df8027`。
  每个run的三档逐clip motion/object URDF/map/seed/timeline完全相同。
- 结果按固定三个诊断clip统计：kdw20K为每档0/3 pickup、0/3 strict-end-carry；
  us9 18K与s3 8K也均为每档0/3、0/3，所以这三组的0.05/0.15/0.50从未激活，
  只能判定pre-lift policy失败，不能比较post-lift command robustness。xm17K三档
  都是3/3 pickup；0.05与0.15均为3/3 strict-end-carry，平均post-trigger carry
  fraction分别为`0.8545/0.8752`，0.50则为0/3且均值降到`0.3849`。dau7K每档
  都是2/3 pickup（ball在5.47秒、bin在2.39秒触发，barrel未触发）；0.05为1/3
  strict-end-carry、平均carry fraction=`0.4693`，0.15/0.50均为0/3，均值分别
  `0.2759/0.3290`。这些都是固定seed三clip诊断，不能外推整个bank成功率。
- 每个run独立生成25个MP4：9 raw、9个动态G1/object world-XY trajectory panel、
  3个逐物体三档对比、3个逐档ball/barrel/bin 3x1和1个3x3总览。五批共125个；
  全部严格为单H.264/yuv420p视频流、50fps、501帧/10.02秒，逐文件ffprobe和完整
  decode通过。五张六时刻contact sheet人工确认单G1、正确物体、连续动作和
  cyan/orange轨迹，无default-object替换；摔倒或静止被保留为真实policy结果。
  byte-exact交付按run分到：kdw=`/home/ubuntu/FAR/_check_vis/08-01-2126__kdw7jhze_model20000__command005_vs_015_vs_050`、
  xm=`08-01-2127__xm0hda83_model17000__command005_vs_015_vs_050`、us9=`08-01-2128__us9ogral_model18000__heading_velocity005_vs_015_vs_050`、s3=`08-01-2129__s3w5pl35_model08000__world_velocity005_vs_015_vs_050`、dau=`08-01-2130__dauhd33b_model07000__world_root_error005_vs_015_vs_050`，
  每目录恰好25个MP4。`_check_vis`现为29个`MM-DD-HHMM`目录、240个MP4，顶层
  仍无散落MP4。
- 21:25 UTC fresh W&B复核时五条run仍为`running`。kdw20K、us9 18K、s3 8K
  仍是远端最新完整checkpoint；锁定后xm新出现18K、dau新出现8K。本批保持请求
  开始时immutable lock，不追逐移动中的run，并在manifest显式记录新checkpoint；
  W&B全程只读，未上传media或改变任何run lifecycle。
- 为支持真实command语义，本轮在`record_checkpoint_inference.py`、command和
  observation terms中加入默认关闭的manual semantics路由，并新增unit tests；
  相关24项测试全部通过。首次XM汇总审计曾错误把file-backed source OBJ字节SHA
  与runtime importer规范化后的geometry identity直接比较而fail closed；actor
  实际已在没有OOD override的strict checkpoint geometry gate中通过。审计已改为
  以strict actor commit为准并保留source OBJ只作provenance，9条无需重录且全部
  重新汇总通过。
- 五个审计root依次为
  `/home/ubuntu/FAR/holosoma_runs/original30_representatives_kdw20000_command_sweep005_015_050_consecutive0_20260801_202737`、
  `/home/ubuntu/FAR/holosoma_runs/original30_representatives_xm17000_command_sweep005_015_050_consecutive0_20260801_204300`、
  `/home/ubuntu/FAR/holosoma_runs/original30_representatives_us9ogral18000_heading_velocity_sweep005_015_050_consecutive0_20260801_204300`、
  `/home/ubuntu/FAR/holosoma_runs/corl79_representatives_s3w5pl35_08000_world_velocity_sweep005_015_050_consecutive0_20260801_204300`、
  `/home/ubuntu/FAR/holosoma_runs/corl79_representatives_dauhd33b_07000_world_root_error_sweep005_015_050_consecutive0_20260801_204300`；
  实体均在`/data/holosoma_eval_audits/`。每root保存checkpoint-selection、recording、
  analysis、video、media-validation、visual-review、W&B-recheck和delivery manifest。
  其delivery manifest SHA256依次为
  `0c22e084133d41cfb97107f4f0bba27bbd7fc5e8521a7004745e7b25789d87da`、
  `9c9e0f8fa51debd78b4636efe1e1f686251760efea7a69f8078df3f8a50b13e0`、
  `82ab4af1de8926fa17995e64af5bb1903a9884b7ca07fb91d6655ba432137b3f`、
  `f4cb66a1cdb26fde8b1b1cf9bcf58c25ded726d04024628ebaa27ea1b3633fe1`、
  `547614750696fdb84282da810d6365a23df46e4ac3203b9047548e029031c60f`。
  录制仅使用GPU3/4/5，结束后GPU3--7均约906 MiB/0%；没有残留evaluation进程，
  GPU0--2上的训练进程未被操作。

## CORL79 turn/forward解耦command正式训练交接（2026-08-02 07:18 UTC）

- 用户以`xm0hda83`的heading-relative pose command为基线，要求CORL79中曲折
  reference path仍可预先表达，但尽量让每个时刻只做一件事：转向时只给yaw，
  前进时只给forward，XY与yaw不得耦合。本轮新增
  `precomputed_turn_then_forward` command mode；actor仍接收原有三维
  `[dx,dy,dyaw]`槽，runtime pickup latch前严格为零，latch后按reference
  timestep读取NPZ中的`policy_command_xy_yaw`和`policy_command_phase`。
  `dy`始终精确为0；forward phase严格为`[dx>0,0,0]`，yaw phase严格为
  `[0,0,dyaw!=0]`，两者不会同帧出现。command不依赖robot-state feedback，
  也不做finite difference；reward、critic reference、termination及motion
  timeline均未改变。
- 预计算采用XY 5-frame smoothing、RDP tolerance 0.06 m、最短polyline leg
  0.10 m、最小turn 25 deg、8--30 frame yaw window、相邻turn间至少5个forward
  frame；forward command最大0.15 m并在leg末端缩小。loader对字段是否成对存在、
  dtype/shape/finite、phase集合、exact-zero dy、forward/yaw不重叠和数值范围全部
  fail closed，不存在fallback。实现入口为
  `scripts/build_decoupled_root_command_bank.py`、
  `scripts/export_heading_path_commands.py`及command/observation terms；相关focused
  test共83项通过，`compileall`和`git diff --check`通过。
- 最终derived immutable bank位于
  `data/ds_as_data/carryany_filter_scale_noscale_keep169_20260513_plus_box_teacher_rollout_success155_bcleb5oi58000_final0p5_primitiveproj_solid80_clean_box_bin_barrel_ball_cominertia_categorymass_v2/_scientific_corl79_decoupled_turn_then_forward_v1/by-source/f7f2a9e78519318a549c4ac7c184e7e9de2e3c01e6e2a316d42fd1285b4b3a67`。
  它包含actual 79 clips（box25、ball4、barrel35、bin15）和26,096 frames；source
  arrays、object map及real URDF/mesh依赖闭包与源逐字节一致。全bank phase总数为
  zero=7,112、forward=16,488、yaw=2,496。derived digest=
  `f7f2a9e78519318a549c4ac7c184e7e9de2e3c01e6e2a316d42fd1285b4b3a67`，
  manifest SHA256=
  `7b1619d0958eb34bb06ac7753b571ea4960a64c5641ffb3068b390d54f7a5f78`，
  object-map SHA256=
  `cddc45058f70751d1d4d033b8138ab3a4a33d78bd973ab5c038124f05ca1af9b`。
- immutable source snapshot为
  `src-6b846ee72a1246372fd6d9abfb1fa0dbb521ba904b6fa690a85cc551a5f31a34`，
  source archive SHA256=
  `8487461c14ba1a8e794f6e9413805f610d4e49f8b346e438fd7211200aa0b068`。
  external input closure有239 files，SHA256=
  `49cb29abf330096ed874625dce84cd763e978d369ecf3e7a992d9a9e5208abdd`；
  79份NPZ/URDF/visual/collision mesh完整且无fallback。world32 rank shard按
  clip-object依赖闭包global exact-once分配，17 ranks x2 clips、15 ranks x3
  clips，无legacy asset shard或DataLoader二次shard；manifest SHA256=
  `bc89320a4c940a371c22e87dce6ab6cb0339f3fe8fdc9fe625dab3dd9d5faff5`。
- exact-topology canary使用4 nodes x8 L40S、每rank 1026 env，完成2/2 update并
  自然退出；第二次update为9.78 s、约80,556 steps/s，PPO coeff=1，BC/distill/
  dagger/replay-BC均为0，distributed loss-weight sum=32，全部tensor/metric finite。
  canary acceptance SHA256=
  `12963ddb46a85976d873748aa1da980fa40ecf9c9cbdf98f17edfdbd6ec2c594`；
  canary checkpoint仅作完整性审计，正式训练没有用它初始化。
- 第一次formal identity `zg8ue4y8`在iteration 0、首个rollout和checkpoint之前
  被ONNX export语义保护拒绝：standalone inference尚未实现这套exact
  precomputed-command加pickup-latch合同，继续导出会产生语义错误的ONNX。
  该run没有PT/ONNX或policy update，state保持`failed`，并显式标记
  `pre_iteration_failure=1`、`training_target_reached=0`、
  `superseded_by=kpl2p2gn`；不得resume或用作实验结果。
- 正式fresh pure PPO run为`zihanw22/carry-any/kpl2p2gn`：
  <https://wandb.ai/zihanw22/carry-any/runs/kpl2p2gn>，name=
  `pure_rl_corl79_precomputed_turn_forward_ws32_e1026_ptonly_20260802_064432`。
  拓扑为ap-northeast-2a的`10.99.0.39/.183/.180/.201`，4 nodes x8 L40S，
  每rank 1026 env、global 32,832 env，tmux=`hs_kpl2p2gn`。它从iteration 0 fresh
  开始，teacher/distill/BC/dagger/replay-BC/training resume/policy init均关闭；
  actor scalar94+depth32，actor/critic MLP均为`[512,256,128]`，29D action，
  24 steps/env、7 epochs、4 minibatches、40,000 updates、actor/critic初始LR
  均为`1e-3` adaptive。offline contact guidance reward权重为0。PT每1,000 update
  保存并上传，禁止skip upload；ONNX暂不导出，直到standalone inference实现并
  验证exact command parity，不能用一个语义不等价的ONNX冒充正式导出。
- Rule-90由该run最终冻结输入的canonical第一条`box_10`重新录制：single G1、
  correct mapped real mesh、randomization disabled，H.264 1280x720@50fps、368帧/
  7.36 s、1,535,814 bytes，SHA256=
  `d37d3ba6fae47a792083f90e58bda05cfe696120abb0a4fa741f26a8f60429a5`。
  contact sheet人工确认approach/pickup/carry/drop/return连续，无default pose、
  错物体或fallback。训练history建立后，同一字节在step102写为唯一
  history-backed `vis/replay`；primary summary已绑定，prebind副本在远端下载
  逐SHA确认后删除。fresh API确认恰好1条media history、1个MP4，run仍为
  `running`。
- 07:18 UTC有界验收时四节点均有8/8 rank logs、8/8 active GPUs，completed
  iteration为`141/141/141/142`；controller和32个rank的Traceback/ChildFailed/OOM/
  DistBackend/non-finite及geometry fallback均为0，每个rank都明确记录cuboid
  primitives disabled，Isaac `[Error]`总数也为0。W&B step142吞吐为
  151,412 steps/s，collection=3.40 s、learning=1.80 s；PPO coeff=1，所有BC/
  distill/dagger/replay-BC指标为0，distributed loss-weight sum=32，loss/KL及
  noise finite。command phase在线占比约zero 0.485、forward 0.457、yaw 0.0576。
  完整快照保存在launch root的`formal_acceptance.json`，SHA256=
  `1cf7f3a27b9c565bbe77b44378b8fb07a5ab8daf0cbc8e7705c1fde771c544ae`。
  这些只证明启动、数据、分布式和早期数值健康，不代表最终policy quality；
  验收后不建立常驻monitor，也不得为例行检查attach、停止或重启该run。

## 六个running run最新checkpoint的三档policy录制（2026-08-02 21:18 UTC）

- 用户要求把W&B远端当前仍为`running`的run各取最新checkpoint录制video。本轮
  19:54--20:00 UTC fresh只读扫描枚举并immutable lock六条：
  `kdw7jhze/model_31000.pt`（SHA256=
  `6981fbd4dc97652598022cd8f66390d6755ddb74c91a02a954778088531d08ca`）、
  `xm0hda83/model_33000.pt`（
  `e97b04f990295de64287b3e31f52acc7d1140437c81f17a62cabe1d7f87d430c`）、
  `us9ogral/model_36000.pt`（
  `fe8193b99baee7af6ba7e218fb9aba153ac09d44dc34dc529f82213d54bbe0f0`）、
  `s3w5pl35/model_23000.pt`（
  `2aaf5d913a5d640fe10f4fa1aaee4e109bea873d84655ea734d5ce108b2b1e60`）、
  `dauhd33b/model_22000.pt`（
  `26ddc62711eede840f67aa7e420b4ea85a86bd3fae28a65d09214ef2e1f304a3`）及
  `kpl2p2gn/model_08000.pt`（
  `d6fe6db042b07dbac8b4e2e696f63386f98d4991fd55ce895b7101be5236550a`）。
  checkpoint内部iteration、finite tensor、run provenance和训练合同均验证；
  kdw为pure BC，其余为`distill=false`的fresh pure PPO。这是checkpoint actor
  policy evaluation，不是teacher rollout，也不是`replay.py` reference replay。
- 共用协议延续上一批：single env/G1、seed42、timestep0、randomization disabled、
  initial-pose noise 0、501 steps@50Hz、prepend0.2s、append2s、关闭adaptive/T1/
  reset；object world-z首次达到初始值`+0.30 m`后立刻启用command，
  `consecutive_steps=0`，三档为0.05/0.15/0.50。kdw/xm是真实
  robot-heading relative pose（m），us9是robot-heading velocity（m/s），s3是
  world velocity（m/s），dau是world root error（m）。所有rollout都审计了501行
  policy-I/O、actor command三槽、29维finite action和跨三档完全一致的pre-trigger
  timeline；checkpoint加载没有启用OOD geometry override，strict live geometry
  membership gate实际通过。
- kdw/xm/us9继续使用exact original30 ball/barrel/bin三初始化，view digest=
  `1f3966245545689a2e14909dda31d6673790bdf888168362449a0ac1902f3510`；s3/dau
  使用各自CORL79 bank内`noscale__any_ball_3/barrel_12/bin_32`，view digest=
  `c9e02244ac1e3c870564f70837a963b03a337430bb1b4a58dc50610868df8027`。
  kpl严格使用其actual79 derived precomputed训练view，digest=
  `f7f2a9e78519318a549c4ac7c184e7e9de2e3c01e6e2a316d42fd1285b4b3a67`；但
  native precomputed turn-then-forward schedule在本次诊断中被evaluation-only
  constant heading-relative-pose override替换，manifest明确标为override ablation，
  禁止把这些视频描述为kpl native precomputed rollout。
- 三个固定clip上的pickup/strict-end-carry按0.05/0.15/0.50分别为：kdw31K=
  `0/3,0/3`三档全相同；xm33K=`3/3,3/3`、`3/3,3/3`、`3/3,0/3`，平均
  post-trigger carry fraction=`0.8871/0.8778/0.3784`；us9 36K和s3 23K均为
  三档`0/3,0/3`；dau22K=`2/3,1/3`、`2/3,1/3`、`2/3,0/3`，对应均值
  `0.2999/0.3220/0.1221`；kpl8K override三档均为`2/3,0/3`，均值
  `0.2852/0.2355/0.3485`。kdw/us9/s3三物体都未越过lift门；dau的ball/bin触发、
  barrel未触发；kpl的barrel/bin触发、ball未触发。未触发组的三档值从未进入
  actor，不能拿来比较post-lift command robustness；结果也不能外推整个bank。
- 每run生成25个MP4：9 raw、9 dynamic G1/object world-XY trajectory panels、
  3 per-object 3-command comparisons、3 per-command 3x1和1 master 3x3；六批共150。
  全部为单H.264/yuv420p stream、50fps、501帧/10.02秒，逐文件ffprobe、完整
  decode通过。六张master contact sheet按0/2/4/6/8/10秒人工检查，确认G1、正确
  ball/barrel/bin、动作与cyan/orange轨迹连续，无default-object/frustum/depth
  debug替换；跌倒、静止和失持均保留为真实policy结果。
- `_check_vis`按每次实验实际recording-manifest完成分钟分成六个独立目录：
  kpl=`/home/ubuntu/FAR/_check_vis/08-02-2017__kpl2p2gn_model08000__relative_pose_override005_vs_015_vs_050`、kdw=`08-02-2032__kdw7jhze_model31000__relative_pose005_vs_015_vs_050`、
  xm=`08-02-2057__xm0hda83_model33000__relative_pose005_vs_015_vs_050`、us9=`08-02-2103__us9ogral_model36000__heading_velocity005_vs_015_vs_050`、s3=`08-02-2108__s3w5pl35_model23000__world_velocity005_vs_015_vs_050`、dau=`08-02-2113__dauhd33b_model22000__world_root_error005_vs_015_vs_050`，
  每目录恰好25个byte-identical symlink MP4，没有把六组耦合到同一目录。
  总交付manifest为
  `/data/holosoma_eval_audits/running_latest_command_sweeps_20260802_195444/delivery_manifest.json`，
  SHA256=`4b3a8959492a1641801b9981cfb3133d4c9255fc3eb088e10d7c97cc53c676eb`；
  `_check_vis/MP4_SUMMARY_BY_DATE.md`已更新为35个含MP4的时间目录、390个MP4。
- 21:17 UTC fresh W&B只读复核确认六条run仍全部`running`、已锁checkpoint仍存在且
  size一致。kdw仍以31K为最新；持续训练期间xm/us9/s3/dau/kpl分别新出现
  34K/37K/24K/23K/9K。本批保持请求开始时的immutable snapshot，不无限追逐
  moving run；每root的`wandb_recheck.json`显式记录二者。W&B全程无media上传、
  config/summary修改或lifecycle写入。
- 主审计root位于`/data/holosoma_eval_audits/`下六个以
  `*_20260802_195444`结尾的目录，并在`/home/ubuntu/FAR/holosoma_runs/`保留同名
  symlink。每root含checkpoint-selection、recording、evaluation-analysis、video、
  media-validation、visual-review和W&B-recheck manifest；campaign脚本保存在
  `running_latest_command_sweeps_20260802_195444/scripts/`。XM对本地改写URDF的
  fail-closed诊断和KPL的早期并行mesh-cache冲突尝试均隔离在各自
  `retry_quarantine/`，没有进入final/manifest或交付文件。

## 六个run按checkpoint原生command模式重录（2026-08-03 06:03 UTC）

- 用户要求纠正上一批统一`+0.30 m/consecutive_steps=0`诊断，把每个run按其训练
  contract里的正确command模式重新render。本轮先按请求开始时的W&B状态immutable
  lock：`kdw7jhze/model_35000.pt`（SHA256=
  `95191784d34eb70cec6edf7e3b27aaf9fdcd2082994dd0f2533ca9989a8d79fc`）、
  `xm0hda83/model_39000.pt`（
  `3186bf02c556f1774c70117847fa63bcde326273fcf26d251a71bf686f0d8fbf`）、
  `us9ogral/model_40000.pt`（
  `3b9d3eab65a2b19dfb546233356bd46286c98daa48016e08f15f2421b705df8c`）、
  `s3w5pl35/model_29000.pt`（
  `3ca5205186b86340dae7d3ba1eec99c35430169acb52dc914638b208ea499333`）、
  `dauhd33b/model_29000.pt`（
  `e9c2d54af3fad0e954ca7209b9c65907fc4264e24a2c6e468e1561709574638a`）和
  `kpl2p2gn/model_15000.pt`（
  `60ab14ece98b733a7adf3f4bc4530ba9d6e08968c2bd699771ead46cb73b9ef9`）。
  这是checkpoint actor policy evaluation，不是teacher rollout，也不是
  `replay.py` reference replay；W&B全程只读，无media/config/summary/lifecycle写入。
- 每个run的实际原生模式为：kdw使用native reference carry window门控的robot-heading
  reference tracking error `[dx,dy,dyaw]`；xm使用仓库原生pickup latch后再给
  robot-heading relative pose `[0.5 m,0,0]`；us9 hybrid分别强制选择tracking/task
  branch，其中tracking逐帧使用reference heading-frame velocity，task使用原生latch
  前zero、后heading `+0.5 m/s`；s3 hybrid同样分别录tracking/task，坐标系为world；
  dau逐帧使用world robot-root-to-reference error；kpl严格读取immutable NPZ中的
  `policy_command_xy_yaw/policy_command_phase`，执行原生latch后的turn-then-forward。
  本轮没有任何`--manual-forward-*`或统一relative-pose override；hybrid仅固定分支，
  未改写分支内command。
- 保持三个固定、各自训练bank内的ball/barrel/bin初始化；single env/G1、seed42、
  timestep0、randomization disabled、initial-pose noise 0、adaptive/T1/reset关闭，
  501 frames@50fps。共完成24/24个rollout，并记录每帧policy-I/O；公式/坐标变换审计
  的最大误差均在`6.27e-6`以内，kpl命令与NPZ逐帧完全一致。三固定clip上的
  pickup/strict-end-carry为：kdw35K=`3/3,0/3`，xm39K=`3/3,0/3`；us9 40K
  tracking=`3/3,0/3`、task=`0/3,0/3`；s3 29K tracking=`3/3,1/3`、task=
  `0/3,0/3`；dau29K=`3/3,0/3`；kpl15K=`3/3,0/3`。us9/s3的task列三条都没有
  触发native latch，因此actor实际持续收到zero task command；这是真实模式结果，
  不是录制失败。以上只是三条固定clip诊断，不代表全bank robustness。
- 24个raw和对应dynamic G1/object world-XY trajectory视频均成功；加上comparison共
  68个审计MP4，全数为单H.264/yuv420p stream、50fps、501帧/10.02秒，逐文件
  ffprobe和完整decode通过。六张master contact sheet按多个时刻人工检查，确认单G1、
  正确ball/barrel/bin、动作和轨迹连续，无default-object、depth/frustum overlay；
  跌倒、静止和失持均按真实policy输出保留。
- `_check_vis`按run拆成六个独立目录：kdw35K=`/home/ubuntu/FAR/_check_vis/08-03-0600__kdw7jhze_model35000__native_tracking_error`
  （7 MP4）、xm39K=`08-03-0601__xm0hda83_model39000__native_post_pickup_relative_pose050`（7）、us9 40K=`08-03-0602__us9ogral_model40000__native_tracking_vs_task_heading_velocity050`（18）、s3 29K=
  `08-03-0603__s3w5pl35_model29000__native_tracking_vs_task_world_velocity050`（18）、dau29K=`08-03-0604__dauhd33b_model29000__native_world_root_error`（7）、kpl15K=`08-03-0605__kpl2p2gn_model15000__native_precomputed_turn_then_forward`（7），
  合计64个byte-exact交付MP4。每目录含`00_master`、raw、trajectory、README及审计
  manifest；hybrid额外含两列mode和逐物体对照。`_check_vis/MP4_SUMMARY_BY_DATE.md`
  已更新为43个时间目录、474个MP4。
- 主审计root为
  `/data/holosoma_eval_audits/native_checkpoint_modes_six_runs_20260803_053222/`。
  frozen-selection/evaluation-matrix/scheduler/analysis/delivery/visual-review/W&B-recheck
  manifest SHA256依次为
  `59168b3f9649cbcdde999dc4d356f777471e83235a736402c853fd0cde12bc29`、
  `d09c4aa06b846cf9c06dfb48c4ea44a08e5a5a184a57ee747baa6e7b7b1569c5`、
  `79630822184260c964a4d892f5f6beff7f4ff5976864b1f96a5191163e7cf6e2`、
  `471886f4a7583a7ed8eedfdd1fa285e7f4b4b30a72983b007f3d584ea719c384`、
  `600a2a110f1e6d2b436604951fffda720caada70e09f6046e9551a9c33c22b2b`、
  `a5e27568edfa0f2a0dbc55857fea13d0acba461be67da4ed8f3d66fbf19dbd9e`、
  `371b51463aff5cc439e0518a15d0b06c311480313ff79164ab1237ca8451a076`。
- 06:03 UTC fresh W&B复核时，除xm外的冻结checkpoint仍是各自最新；xm在本轮
  39K冻结后结束并新增terminal `model_40000.pt`。本批xm39K仍是请求开始时的
  immutable snapshot，不能描述为当前terminal/latest；若比较terminal XM，必须
  另开一批明确录40K，不能静默替换本批结果。

## `_check_vis` 日期目录补充checkpoint/对照语义（2026-08-03 16:46 UTC）

- 用户因纯日期目录无法直接识别kpl交付，要求日期后同时写明checkpoint或
  `A_vs_B`。确认kpl已经完成：`kpl2p2gn/model_15000.pt`的checkpoint-native
  precomputed turn-then-forward位于改名后的
  `/home/ubuntu/FAR/_check_vis/08-03-0605__kpl2p2gn_model15000__native_precomputed_turn_then_forward/`，
  含7个MP4，master仍为原byte-identical文件。
- 将`_check_vis`下全部43个含MP4的日期目录从`MM-DD-HHMM`统一迁移为
  `MM-DD-HHMM__run_id_modelXXXXX__mode`；多checkpoint/多策略写`A_vs_B`，多档
  command写`commandA_vs_B_vs_C`。日期时间前缀未变，字典序和原交付时间顺序不变；
  474个MP4均未改名、重编码或改symlink target。三个无MP4临时目录
  `08-01-2330/2345/2355`和非日期旧归档未改动。
- `_check_vis/MP4_SUMMARY_BY_DATE.md`的43行目录映射和全部相对链接已同步；完整
  old-to-new记录位于`_check_vis/DIRECTORY_RENAME_2026-08-03.md`。迁移前各次审计
  manifest保持immutable，里面的旧短路径作为historical delivery target保留；可按
  相同`MM-DD-HHMM`前缀在rename表中唯一解析。当前六run的README、final handoff和
  staging脚本已使用新路径。

## KPL最新checkpoint原生pickup-latch/phase-pure录制（2026-08-03 17:01 UTC）

- 用户要求立即录`kpl2p2gn`最新checkpoint，并强调pickup后forward command必须
  pure。16:53 UTC fresh W&B listing时最新完整文件为`model_22000.pt`（live summary
  step=22949，23K尚未上传）；immutable下载SHA256=
  `af1422421825d462fa3c03417d90d0f9d03ee91d736a3cc03cb32808fb5ce497`，size=
  10,352,929 bytes。checkpoint内部iter/iteration=21999、next_iter=22000、run
  identity、pure-PPO/no-teacher/no-resume/no-policy-init、856个checkpoint tensors
  finite、79个geometry support及actor command group均验证。
- 严格使用run绑定的derived CORL79 view digest=
  `f7f2a9e78519318a549c4ac7c184e7e9de2e3c01e6e2a316d42fd1285b4b3a67`和同一
  in-bank `noscale__any_ball_3/barrel_12/bin_32`。模式为checkpoint-native
  `precomputed_turn_then_forward`：repository pickup latch前actor前三维精确为零；
  latch后逐帧读取NPZ `policy_command_xy_yaw/policy_command_phase`。forward phase
  仅`[dx>0,0,0]`，yaw phase仅`[0,0,dyaw]`，没有恒定0.5m或任何
  `--manual-forward-*` override。
- 三条物理pickup诊断step为85/91/104，首个native command actor step为88/95/109，
  均确认command在pickup后才激活。ball/barrel/bin的forward-only actor rows分别为
  226/203/203，yaw-only为13/29/15，`dy`最大绝对值=0，同帧forward/yaw overlap=
  0，与immutable NPZ逐帧最大误差=0；forward最大为0.15m。结果为3/3 physical
  pickup、0/3 strict-end-carry，平均post-trigger carry fraction=0.2862；只代表三个
  固定clip，跌倒和后段失持均作为22K真实actor结果保留。
- 单环境/G1、seed42、timestep0、randomization disabled、noise0、adaptive/T1/reset
  关闭，501帧@50fps、H.264，拉远相机和dynamic G1/object world-XY trajectory。
  3/3 rollout完成；8个审计MP4全部为单H.264 stream、50fps、501帧并通过完整
  decode。六时刻contact sheet人工确认正确ball/barrel/bin、单G1、动作/轨迹连续，
  无default object或depth/frustum overlay。
- 7个byte-exact交付MP4位于
  `/home/ubuntu/FAR/_check_vis/08-03-1653__kpl2p2gn_model22000__native_precomputed_turn_then_forward/`，
  master为`00_master__kpl2p2gn_model22000__checkpoint_native_modes__ball_barrel_bin__3x1.mp4`。
  `_check_vis/MP4_SUMMARY_BY_DATE.md`已更新为44个目录/481个MP4，146个相对链接全
  存在。主审计root为
  `/data/holosoma_eval_audits/kpl2p2gn_model22000_native_precomputed_20260803_165239/`。
- 17:00 UTC fresh W&B只读复核时run仍`running`、summary step=23022，冻结22K的
  size/MD5不变；`model_23000.pt`在录制后刚上传。本批保持用户请求开始时锁定的
  latest-complete 22K，不静默追逐moving run。W&B全程无写入；录制仅用空闲
  GPU3/4/5，结束后GPU3--7均约906MiB/0%，GPU0--2训练未操作。

## KPL 23K pickup 后只给 NPZ forward-dx 的纠正评估（2026-08-03 17:26 UTC）

- 用户明确纠正上一批语义：“只给forward command”。因此22K原生
  turn-then-forward目录保留为有明确标签的历史诊断，但不能作为本要求的结果；它
  含yaw-only actor command。本批17:11 UTC fresh W&B listing冻结当时最新完整
  `kpl2p2gn/model_23000.pt`，size=10,352,929 bytes、SHA256=
  `0d8051f6c31509759087cf547293cd8492da52b4aedc077d0ae039134d566448`。
- 新增默认关闭、严格evaluation-only的NPZ dx-only actor-observation override：仍用
  repository-native pickup latch；latch前`[0,0,0]`，latch后仅复制immutable NPZ
  `policy_command_xy_yaw[t,0]`形成`[dx,0,0]`，`dy/dyaw`逐帧严格为0。它不是constant
  0.5 command，也不是native turn/forward；NPZ yaw-only phase变成zero command。
  `scripts/record_checkpoint_inference.py`限制该开关只能用于exact checkpoint actor、
  single env、native precomputed mode且不得和manual-forward共用；定向unit tests为
  8 passed，py_compile和diff-check通过。
- 同三个in-bank `noscale__any_ball_3/barrel_12/bin_32`、single env/G1、seed42、
  timestep0、randomization disabled、noise0、adaptive/T1/reset关闭，501帧@50fps。
  3x501 actor输入审计的`max|dy|=max|dyaw|=0`，latch后dx对NPZ第一列最大误差=0；
  首次非零forward step=105/97/114，非零forward rows=209/203/203。latch后经过的
  NPZ yaw-only rows=13/29/17，输出yaw均精确为0。
- 三条均物理pickup但strict-end-carry=0/3，平均post-trigger carry fraction=0.2302；
  跌倒和后段失持按真实23K policy结果保留。8个审计视频均为单H.264 stream、50fps、
  501帧并通过完整decode；六时刻contact sheet确认正确物体、单G1、动作和dynamic
  G1/object world-XY轨迹连续。
- 7个byte-exact交付MP4位于
  `/home/ubuntu/FAR/_check_vis/08-03-1711__kpl2p2gn_model23000__post_pickup_npz_dx_only/`，
  master为`00_master__kpl2p2gn_model23000__post_pickup_npz_dx_only__ball_barrel_bin__3x1.mp4`。
  `_check_vis/MP4_SUMMARY_BY_DATE.md`已更新为45目录/488 MP4。完整审计root为
  `/data/holosoma_eval_audits/kpl2p2gn_model23000_post_pickup_npz_dx_only_20260803_171101/`。
- 第一次retry因Tyro布尔flag多传了`True`而在CLI parse、simulator/policy step前拒绝；
  0 metrics/0 video，日志已隔离并写retry manifest。纠正flag后才录正式三条。
  17:24 UTC fresh W&B只读复核时run仍`running`、summary step=23300，23K仍为最新
  完整checkpoint且远端size/MD5不变；无W&B写入。GPU0--2训练未操作，评估结束后
  GPU3--7均约906MiB/0%，无残留evaluation进程。

## KPL 23K 先 lift、再持续 pure-forward 0.15/0.50（2026-08-03 18:02 UTC）

- 用户纠正语义为“等他 lift 了，然后再给”。本批锁定
  `kpl2p2gn/model_23000.pt`，SHA256=
  `0d8051f6c31509759087cf547293cd8492da52b4aedc077d0ae039134d566448`，
  在同一 run-bank 内固定 `noscale__any_ball_3`、`noscale__any_barrel_12`、
  `noscale__any_bin_32`。这是 checkpoint actor policy evaluation 的明示
  evaluation-only override，不是 `replay.py` reference replay，也不是 NPZ command。
- actor 在物体 world-z 相对配置基线达到 `+0.30 m` 前严格收到 `[0,0,0]`；
  `consecutive_steps=0`。首次越过 gate 后永久锁存 robot-heading relative-pose
  `[0.15,0,0]` 或 `[0.50,0,0]` 到第 501 帧。没有 heading lock，`dy/dyaw` 每帧
  严格为0。ball/barrel/bin 首个非零 actor step 分别为119/107/154；两档 gate 前
  policy action 逐帧 bitwise exact，zero-then-constant 最大误差为0。物体后来落回
  gate 以下时 command 仍持续，验证了 latch-to-end 语义。
- 两档均为3/3 physical pickup、0/3 strict-end-carry；平均 post-trigger carry
  fraction 为0.2475/0.2713。0.50档三个物体净XY位移都更远（ball
  1.072→1.795 m、barrel 1.218→2.778 m、bin 0.678→2.279 m），但两档后段均有
  跌倒或失持，因此不能解读为稳定carry成功。
- 6个raw、6个dynamic G1/object world-XY trajectory、3个逐物体1×2、2个逐档
  3×1和1个3×2 master共18个交付MP4，位于
  `/home/ubuntu/FAR/_check_vis/08-03-1743__kpl2p2gn_model23000__post_lift_forward015_vs_050/`。
  全部symlink可解析且与source SHA256 byte-exact；单H.264视频流、50fps、501帧、
  正尺寸/时长、ffprobe和完整decode均通过。六时刻contact sheet人工确认单G1、
  正确ball/barrel/bin、轨迹连续且无depth/frustum debug overlay；真实失败未裁剪。
- 完整审计root为
  `/data/holosoma_eval_audits/kpl2p2gn_model23000_post_lift_forward015_vs_050_20260803_174352/`。
  17:58 UTC fresh W&B只读复核时run仍`running`、summary step=23669，23K仍是最新
  完整checkpoint且远端size/MD5不变；全程无W&B写入。录制使用空闲GPU3/4/5，
  完成后GPU3--7均约906MiB/0%，未操作GPU0--2训练。

## KPL 23K 额外12条、lift后持续pure-forward 0.15/0.30（2026-08-03 18:57 UTC）

- 用户要求用更多data只测forward x=0.15/0.30。为与紧邻的三clip 23K诊断直接可比，
  本批有意复用exact `kpl2p2gn/model_23000.pt`（SHA256=
  `0d8051f6c31509759087cf547293cd8492da52b4aedc077d0ae039134d566448`），
  而不是静默追逐录制前已经出现的24K。evaluation bank仍是run绑定的actual79
  derived view，digest=`f7f2a9e78519318a549c4ac7c184e7e9de2e3c01e6e2a316d42fd1285b4b3a67`。
- outcome-blind选择额外12条：排除前批`noscale__any_ball_3/barrel_12/bin_32`后，
  每category按manifest-canonical顺序取首/中/末。box=`box_10/60/92`；ball=
  `noscale__any_ball_6/82/84`；barrel=`noscale__any_barrel_1`、
  `scale__any_barrel_13/9`；bin=`noscale__any_bin_34/85`、`scale__any_bin_85`。
  使用各clip同view内正确URDF；mass为box/bin 1.0kg、ball0.5kg、barrel1.5kg。
- 24条checkpoint-actor policy rollout均为single env/G1、seed42、timestep0、
  randomization disabled、initial-pose noise0、adaptive/T1/reset关闭、501帧@50fps。
  actor在actual object world-z相对配置基线达到+0.30m前严格收到`[0,0,0]`，
  `consecutive_steps=0`；越过后永久锁存robot-heading relative-pose
  `[0.15,0,0]`或`[0.30,0,0]`到rollout末尾。没有heading lock、lateral/yaw、NPZ
  command或motion-following override；drop button在24×501 actor rows全部精确为0。
- 配对审计确认同clip两档的initial robot/object state、timeline、URDF、NPZ和gate
  step完全相同，gate前policy action bitwise exact；zero-then-constant、dy、dyaw、drop
  的最大误差均为0。两档都是10/12 physical pickup、9/12越过+0.30m gate；未激活
  的`noscale__any_ball_6`、`noscale__any_bin_85`、`scale__any_bin_85`两档全程zero
  command/action bitwise exact，不能用于0.15/0.30效果归因。
- strict-end-carry按全部12条为0.15m=`2/12`、0.30m=`8/12`；category分别为
  box`1→3`、ball`0→2`、barrel`1→2`、bin`0→1`，平均post-trigger carry fraction
  `0.3895→0.7179`。这只是单checkpoint的12条确定性扩展诊断，不能直接外推全bank
  robustness，尤其三条command未激活必须和九条active样本区分。
- 完整审计root为
  `/data/holosoma_eval_audits/kpl2p2gn_model23000_stratified12_post_lift_forward015_vs_030_20260803_183805/`。
  24 raw、24 dynamic G1/object world-XY trajectory、12 per-clip 1x2、4 per-category
  3x2、2 per-mode 4x3 master共66个视频，全部单H.264/yuv420p、50fps、501帧/
  10.02s并通过ffprobe和完整decode。两张覆盖24 rollout的六时刻contact sheet人工
  确认四类正确物体、单G1、动作/轨迹连续且无depth/frustum overlay。
- 18个主要对照byte-exact symlink交付在
  `/home/ubuntu/FAR/_check_vis/08-03-1838__kpl2p2gn_model23000__stratified12_post_lift_forward015_vs_030/`；
  `00_master...forward015...4x3.mp4`和`01_master...forward030...4x3.mp4`是用户要求的
  两支总览，另含4条category和12条clip paired对照。`_check_vis`索引更新为47目录/
  524 MP4。
- evaluation matrix、recording、media validation、delivery和W&B post-recheck manifest
  SHA256分别为`f14aa51d1ba926d7af0270ff7c4dbc6d9295ff80c35dfcd6bc4ee19a0a7b13a7`、
  `76a9b4cdafdd044248a830bd608d08c94c282a24b9714893d5e3c64c46e5d499`、
  `b2bd2f2a09304659ec2d151617a26d141da019977c7b0a29a67ed94e676472aa`、
  `1c61c6884ab006f59935e26237bbaf371520686f75e6e87533a9ea790da1f7f0`、
  `a663567fe60124bda926d57e0ddaba00e6d93f5dd286997f68829b3e1d0b33b4`。
  18:56 UTC fresh W&B只读复核时run仍`running`、summary step=24333，24K仍为最新，
  23K远端size/MD5不变；没有任何W&B写入。GPU0--2训练未操作，结束后GPU3--7均
  约906MiB/0%，无残留evaluation进程。

## CORL79 + debug30 portable real-mesh训练bank（2026-08-03）

- 用户要求把实际CORL79和canonical debug30合并为一份可跨节点训练数据。本次只做
  数据构建、NFS发布和空闲节点同步；没有启动、停止、attach或修改任何训练进程。
  两个immutable输入分别为CORL79 single-slot view digest
  `c9e02244ac1e3c870564f70837a963b03a337430bb1b4a58dc50610868df8027`
  （source digest=`89e1f7d099a741b03a6153654e1821e52deaf023eac8072927e965415b766fac`）
  和debug30 source digest
  `1f3966245545689a2e14909dda31d6673790bdf888168362449a0ac1902f3510`。
  两边clip ID零重叠；motion字段、50 FPS、32 body、29 joint及张量尾维契约一致。
- 合并结果精确为109条：box25、ball9、barrel36、bin39。109/109保留源motion NPZ
  数值字节，109/109有独立URDF和contact目录。visual与collision均使用源real mesh；
  0 primitive、0 fallback、0缺失mesh。URDF只把mesh filename改写为bank内相对路径，
  scale/origin/material/mass/COM/inertia等XML语义逐节点等价验证。109个mesh引用按完整
  文件SHA去重为81个真实OBJ；这只是byte-identical去重，不合并近似geometry。
- canonical本地bank为
  `/data/holosoma_inputs/corl79_plus_debug30_realmesh_categorymass_v1/by-source/aa4dcb12bc14df37446417d98d7179236960d2c715975d0753438d164ceafa5c`。
  payload digest=`aa4dcb12bc14df37446417d98d7179236960d2c715975d0753438d164ceafa5c`，
  manifest SHA256=`4ce2d9bae329be7f4a89ada211d2a3cf70ce92d4b59afdc094c70d8e3063c878`；
  1,993个manifest-bound payload files、0 symlink、0 writable path，未压缩总量
  3,082,720,229 bytes（约2.9 GiB）。
- NFS使用单archive避免`far-research-internal` FUSE逐小文件metadata瓶颈：
  `/nfs/zzzihanw/ds_as_data/_distill/corl79_plus_debug30_realmesh_categorymass_v1/archives/aa4dcb12bc14df37446417d98d7179236960d2c715975d0753438d164ceafa5c.tar.gz`。
  archive size=1,080,025,978 bytes，SHA256=
  `651c3459e26752c4318d201730fbd1b5de98e841b79f56f34922a1872d0e3387`，
  2,109/2,109 archive members经无落盘流式解压哈希验证；publication JSON同目录，
  SHA256=`84334ff0bc080bf995a9ad08e0a2b578771af97d14ea6b5c50bc76e77694e7ec`。
  NFS FUSE忽略`chmod 0444`，因此消费者必须同时钉死archive SHA、payload digest和
  manifest SHA，不得只按可变路径信任文件。早期直接写NFS小文件目录的hidden
  incoming已完整删除；NFS bank root最终只保留正式archive与publication JSON。
- contact root为
  `contact_export_corl79_success133_plus_debug30_realmesh_model05000`。按正式runtime的
  default-pose prepend补偿`0.2 s`重验：valid windows=109/109，contact target=
  109/109，wrist target=106/109；不得关闭prepend compensation后把源interval误判
  为越界，也不得修改label来掩盖时间基。
- `cp_corl79_debug30.sh`默认安装到上述统一`/data/holosoma_inputs/.../by-source/<digest>`；
  每台先把NFS archive复制到本地cache，验证archive，再解包验证payload，原子rename
  后第三次验证；已有错误target时fail closed，不覆盖、不fallback。构建/验证实现为
  `scripts/build_merged_training_bank.py`，定向测试为
  `tests/unit/test_build_merged_training_bank.py`。
- 已在12台当时8/8 GPU空闲节点安装并fresh post-sync验证：`zzzihanw-17/25/26/27/28/30/34/46/47/79/93/102`。
  每台均为109 motion NPZ、109 URDF、81 mesh、109 contact dirs、0 symlink、0 writable、
  0 incoming目录，manifest SHA一致；安装后GPU compute app仍为0。17/26/27/30/47
  根盘为0 KB可用，因此12台的operational installer/verifier均放在
  `/data/holosoma_sync`，远端repo里本次传入的untracked工具已清除，避免未来git pull
  与正式提交冲突；数据本体在所有节点均使用统一`/data`路径。正式训练仍需基于该exact view另行生成
  world-size绑定的rank shards并完成Rule-90/训练contract，不能直接把本次同步描述为
  已启动实验。

## 未来formal run强制原生PT+ONNX（2026-08-03）

- 用户明确要求“以后必须export ONNX”。已将其写入仓库级`AGENTS.md`的独立硬契约，
  并加入本文件长期约束Rule-98。今后任何new/restart/migration formal identity均不得
  再以`training.export_onnx=false`或`PT-only`启动；缺deployment parity时应阻止正式
  launch，先完成inference实现和PyTorch-vs-ORT验证，不能用关闭导出来换取开跑。
- required checkpoint cadence上的PT/ONNX必须同iteration一一配对，经过真实导出、
  checker、ORT load、数值parity、原子发布、SHA/provenance绑定和W&B上传复核；任一
  ONNX环节失败时该boundary整体fail closed，terminal PT/ONNX/completion marker也
  必须同iteration。actor-only或依赖未声明外部command/latch的图不得标成完整policy。
- 当前live `kpl2p2gn`的immutable合同已明确为历史PT-only例外，本次只增加未来规则，
  没有attach、暂停、重启、热改或写入该run。它可继续当前进程；若未来需要resume或
  migration，必须先补`precomputed_turn_then_forward + runtime pickup latch`的部署
  parity，并使用符合新规则的新formal identity。历史sidecar backfill可做修复，但不
  能作为未来formal原生PT+ONNX合同的替代品。

## kpl2p2gn latest native interactive inference（25K -> 26K，2026-08-03）

- 20:52:49 UTC fresh W&B只读选择时，`zihanw22/carry-any/kpl2p2gn`仍为`running`，
  summary step=25650，最高完整checkpoint为`model_25000.pt`；远端size=10,352,929，
  本地SHA256=`7182539a6e05173f661ce05cf63e9dab6eecfcf368396cddf46ad3da92e3c3e8`，
  checkpoint内部completed iteration=24999/next=25000且856个tensor全finite。
- 已用exact immutable source `src-6b846ee72a1246372fd6d9abfb1fa0dbb521ba904b6fa690a85cc551a5f31a34`
  和训练绑定motion view digest
  `f7f2a9e78519318a549c4ac7c184e7e9de2e3c01e6e2a316d42fd1285b4b3a67`
  拉起单环境、完整79-clip bank的PT checkpoint-actor inference，初始clip为`box_10`。
  Viser的`Clip Playback -> Clip -> Apply Clip`可在全部79条中切换，并同步切换clip对应
  的真实物体actor。它使用checkpoint-native `precomputed_turn_then_forward`与runtime
  pickup latch，不是reference replay、teacher、manual command override或ONNX inference；
  randomization disabled、timestep0、pose noise0。
- 在25K viewer准备期间，W&B于21:23:32 UTC新增了`model_26000.pt`；21:45 fresh只读
  listing后已将服务替换为该当时最高完整checkpoint。26K远端size=10,352,929，本地
  SHA256=`f2323377632d15c0d83800e6730c85e06235ee827ef909f470f162367ea510e5`，
  checkpoint内部completed iteration=25999/next=26000，856个tensor、2,451,687个值
  全finite；其experiment、79-object geometry support与motion transition合同均和25K相同。
  21:56 UTC再次fresh W&B只读复核时run仍为`running`、summary step=26364，26K仍是
  最高完整checkpoint。
- 当前服务运行于physical GPU7、tmux `hs_kpl2p2gn_model26000_native_inference`、Viser
  `0.0.0.0:34103`，PID=758629，用户入口为`http://localhost:34103`。21:51 UTC runtime
  明确加载26K actor（completed iteration=25999）；viewer-only candidate-bank验证投影逐一
  确认79个object mesh集合完全相等、camera source与robot mesh bindings完全一致，且不修改
  runtime perception或policy语义。前24个连续policy steps（0--23）已写入probe且action非零。
- 21:52--21:55 UTC通过带ViewerCamera握手的真实Viser GUI WebSocket消息实测切换：dropdown
  精确包含79个options，`box_10 -> noscale__any_ball_3(index 25) -> box_10`；runtime日志
  逐次确认visual/collision actor与ground status由`Object_0_box_10`切为
  `Object_25_noscale__any_ball_3`后再恢复，最终状态再次确认为`box_10`。早期验证客户端
  断连时留下3个nonfatal Viser callback `KeyError`，随后用保持连接的客户端完成恢复；
  HTTP=200、tmux/PID持续存活，policy/runtime无fatal error。
- 22:37 UTC用户明确实际eval rollout必须是strict manual-reset-only：不允许timeout、
  bad-tracking、motion-end或command rollover自动reset，只有显式Viser操作可以reset。
  服务已按该合同重启：`HOLOSOMA_DISABLE_AUTO_RESET=1`使immutable source的
  `BaseTask._check_termination()`在生成任何reset/timeout mask前直接返回，也使
  `MotionCommand`在clip末帧clamp而不rollover；另显式设置
  `HOLOSOMA_DISABLE_MOTION_END_RESET=1`和`HOLOSOMA_DISABLE_BAD_TRACKING_RESET=1`。
  独立wrapper中的termination-manager gate是第二层fail-safe；正常路径因BaseTask原生
  guard先返回而不会进入。22:45 UTC通过真实Viser GUI WebSocket `Reset`按钮验证唯一的
  手动路径可用，日志记录`Viser reset visible replay envs: [0]`且`box_10`机器人/物体恢复
  初态；HTTP=200、79-clip selector仍完整、无callback/runtime error。
- 审计root为
  `/home/ubuntu/FAR/holosoma_runs/kpl2p2gn_model25000_native_inference_20260803_205147/`，
  `manifests/inference_health.json`记录26K health、切换证据与哈希；25K的成功日志、probe和
  manifest已另存为带`model25000`后缀的审计文件。W&B无写入，训练GPU/process
  及原有34100--34102 Viser均未操作。失败启动均审计保留：attempt1为日志级别不足；
  attempt3为single-slot单环境无法覆盖79 clips；attempt4/5确认legacy multi-actor viewer
  与strict direct-deployment topology验证的差异，最终只在独立wrapper中增加上述精确
  candidate-bank验证投影，没有编辑immutable source。

## CORL79+debug30 x-only/yaw-only 64卡formal pure-RL（2026-08-03）

- 用户要求在8台机器上、每卡2048 env启动“只有x / 只有yaw，drop照旧”的正式训练。
  最终合同是fresh pure PPO，不加载teacher、student、BC/DAgger、resume或policy-init。
  actor的三维root command在runtime pickup latch前恒为`[0,0,0]`；之后每帧只能是
  zero、`[dx>0,0,0]`或`[0,0,dyaw!=0]`，dy恒为0且dx/dyaw不重叠。该改动只作用于
  actor command input；drop button、reference tracking reward、privileged critic、
  termination、motion timeline及物理contact保持原合同。offline contact guidance
  reward权重为0。
- exact bank为CORL79加debug30的109条immutable derived view：
  `/data/holosoma_inputs/corl79_plus_debug30_decoupled_turn_forward_v1/by-source/307e9662d498bd507b9d17ca9abf74a3654f7bf66ac6ab989c6f19c3889bddef`。
  类别为box25、ball9、barrel36、bin39；phase总数为zero8888、forward24110、yaw2668。
  manifest SHA256=`2de9ee5ca188b70e877c32dd9f0d2975eea99d11aa077bb077cf06ea9ab897bb`，
  object-map SHA256=`70b466aad04837a79f6dd0f4491cb345a73c687209981acd3eb7f4a0365d8f5c`。
  real visual/depth mesh保持原mesh，collision以同一mesh的`convex_decomposition`导入；
  301-file external closure SHA256=`3219b4de13c6eb7f89dbd1f46619f9fdcf7aca75fb02473798423bc0cf2c158f`，
  禁止geometry fallback。
- world64 rank shards位于
  `/data/holosoma_inputs/corl79_plus_debug30_decoupled_turn_forward_v1/_rank_shards/by-source/1cd6b8f0595cb6374a7f3f46eeddf7ccdfbc7e064ada6d3b6eae37a6d1df2b17/ws64`，
  manifest SHA256=`ba33c280272c42b35dab1993a5bc543b5ddb0cbb04468e4cae56bf0a74508033`。
  19 ranks各1 clip、45 ranks各2 clips，109条global exact-once，每个local clip数都整除
  2048；legacy asset sharding关闭，没有DataLoader二次shard。
- 最终source snapshot为
  `src-ed55f5d4be259d8218f268c0f65e0d7f57b8e9626b00fb84c23f18f28965f2aa`，archive
  SHA256=`fdbfa76bb9177c272023431d9fd8882c4929ccce2635d4f5ef4d93ded0d3ccfb`。
  已补齐`precomputed_turn_then_forward`的standalone inference/pickup-latch合同和原生
  PT+ONNX checkpoint pair发布；最终focused/relevant tests共852项通过。exact 8-node、
  64-rank、2048 env/rank canary完成2/2 updates并自然退出，64个rank exit code均为0，
  第二次为101,934 steps/s。其`model_00002.pt/.onnx/.pair.json` SHA256依次为
  `3ee7b6d0d513da96c89cb78b8fac4a88086dabc7157deb90c315d9623564e5a0`、
  `dbe11d052d298b3a0de7b8cdd584ecfa13eaae34527dee83ac47d9730a9bf3f0`、
  `e8082145355a1dd1ecf1567a77f86cf4c42922deab0050fcba5c97df6111e982`；
  checker、ORT CPU load和PyTorch-vs-ORT parity通过，max abs error=`6.7055e-08`。
  canary checkpoint只作验收，正式训练没有用它初始化。
- 正式W&B run为`zihanw22/carry-any/4kf71d6g`：
  <https://wandb.ai/zihanw22/carry-any/runs/4kf71d6g>，name=
  `pure_rl_corl79_debug30_precomputed_turn_forward_ws64_e2048_onnx_20260803_231350`。
  节点依次为`zzzihanw-17/25/26/28/30/34/47/93`，每台8张L40S，world64、每rank
  2048 env、global131072 env，tmux均为`hs_tf109_4kf71d6g`。actor/critic MLP均为
  `[512,256,128]`，actor scalar94加depth，24 steps、7 epochs、4 minibatches，
  actor/critic LR均为`1e-3` adaptive KL 0.01，目标40,000 updates。
  `training.export_onnx=true`，每1000 update必须原子保存并上传同iteration PT+ONNX pair，
  upload skip未设置；首个required formal checkpoint为iteration1000，当前启动验收时尚未
  到达，因此不得把canary pair误报为formal checkpoint。
- Rule-90使用最终bank canonical第一条`box_10`从timestep0做全新policy-free replay；
  H.264 1280x720@50fps、368帧/7.36s，SHA256=
  `be2d7120c0f8257e0b3f2d763f49268ea9fa539d3f94c79fe53161958b2299ee`。ffprobe、完整
  decode和人工contact-sheet检查通过，无default pose、错物体或fallback。训练history
  建立后，同一字节已提升为step2唯一history-backed `vis/replay`；fresh API确认恰好1条
  media history和1个MP4，run保持`running`。
- 23:39:49 UTC有界启动验收时8个tmux均存活、每节点8/8 GPU app、launcher日志中的
  Traceback/ChildFailed/OOM/DistBackend/non-finite/Xid均为0。fresh W&B step13读回
  `Perf/total_fps=171176`、collection=`14.6226s`、learning=`3.7545s`；distill/dagger/
  BC/replay-BC全为0，distributed loss-weight sum为64。W&B system-monitor曾有一个
  独立的GPU telemetry portfile timeout，不影响训练history、优化或64个worker。
- 完整审计root为
  `/home/ubuntu/FAR/holosoma_runs/formal_corl79_debug30_precomputed_turn_forward_ws64_e2048_20260803_220222/`。
  run contract、replay manifest、canary acceptance、formal worker和formal acceptance
  SHA256依次为`1648b81a7ad289faeb059202395a7aa200e5504204ed086e263a7e153dcdd2d7`、
  `5dcc7c5828aa5274103abc457b997c93caed844b4ac47f657a61d6107856aa73`、
  `4f5e42b418b830a4c4733f26bd47b46b87b5b8e1a496f2d80704f855e04c24db`、
  `50f0f26abf7930465b2535f73663944d803803b691cd86749e91250afdb96d61`、
  `ff4ac8c50ccfc23bf362498af5eb8bf5fa016c2dc07c94f1f114cefd62db9d5c`。

## XM terminal 40K vs KPL 27K真实policy-input面板评估（2026-08-04 00:20 UTC）

- 用户要求重测`xm0hda83`与`kpl2p2gn`最新checkpoint，并让底部面板反映policy
  的实时input。23:57 UTC fresh W&B只读冻结时，XM已`finished`且terminal为
  `model_40000.pt`（SHA256=
  `62a33b7e7417e8f6296d8acd0bd314e1281ef07897ae3627262bb36d4e4bd86d`）；KPL仍
  `running`、summary step=27743，最高完整为`model_27000.pt`（SHA256=
  `8282fbe383025d1822f821b2c0ebd3cec13211b81a8ca7e217cf0ca91608e693`）。两份
  checkpoint内部completed/next分别为39999/40000与26999/27000，856个tensor均
  finite；actor input合同相同为command3 + drop1 + proprio/actions90，共94维，另有
  `58x87=5046`维model-ready depth。两者`camera_pitch_deg=0`，没有相机角度mismatch。
- 六条均为checkpoint-native actor policy evaluation，不是`replay.py` reference replay、
  teacher或manual-forward override。XM使用original30训练bank内固定
  `unscale__any_ball_29/scaledown__any_barrel_25/unscale__any_bin_29`，原生pickup
  latch后actor收到heading-relative `[0.5,0,0]`；KPL使用其actual79 derived bank内
  `noscale__any_ball_3/barrel_12/bin_32`，原生latch后逐帧读取immutable NPZ的
  precomputed `[dx,0,dyaw]`。两个bank没有重叠motion ID，因此master同列只表示同
  object category，不冒充同初始化严格A/B。
- 每条保存501行actor-forward前真实policy I/O。面板直接使用
  `actor_obs_raw_values[0:3]`、真实drop槽、完整94维raw/normalized统计和完整
  `perception_obs_values[5046]` reshape的`58x87` depth，不从command manager配置
  反推。depth nominal映射约`[-0.5,+0.5]`且含checkpoint-native noise，本批全局实际
  为约`[-0.55,+0.56]`；成片统一固定`[-0.6,+0.6]`色标，不做逐帧min/max拉伸。
  第一次构图因错误假设色标为`[0,1]`在交付前被审计拒绝并全部重编码；rollout tensor
  本身未丢失或重跑。
- XM ball/barrel/bin首次非零command step=`70/69/71`，之后严格持续dx=0.5、
  dy=dyaw=0；KPL首次非零step=`96/96/109`，latch后对NPZ逐帧最大误差=0，dy恒0且
  forward/turn不重叠。原生drop button并非0：XM首次drop=1为`291/292/291`，KPL为
  `292/274/321`，两者均持续到末帧；面板忠实显示而没有沿用manual-forward评估中的
  drop=0假设。六条均>=0.30m physical pickup但0/6 strict-end-carry；XM三条
  post-command carry fraction=`0.406/0.382/0.388`，KPL=`0.316/0.356/0.375`，不能
  从不同初始化的三条样本外推full-bank robustness。
- 共6 raw、6单rollout input panel、3逐category两列、2单policy 3x1和1份2x3 master，
  合计18个MP4；全部单H.264/yuv420p、50fps、501帧/10.02秒，ffprobe和完整decode
  通过。六时刻master及XM command/drop、KPL turn/drop切换边界已人工抽帧确认单G1、
  正确物体、连续depth/轨迹和真实失败。byte-exact交付目录为
  `/home/ubuntu/FAR/_check_vis/08-04-0019__xm0hda83_model40000_vs_kpl2p2gn_model27000__native_policy_input_panel/`；
  `00_master...2x3.mp4`为首选入口，`_check_vis`索引更新为48目录/542 MP4。
- 完整审计root为
  `/data/holosoma_eval_audits/xm40000_vs_kpl27000_native_policy_input_panel_20260803_235726/`。
  evaluation/media与W&B recheck manifest SHA256分别为
  `67454805114f0530c21f7c35b8bcf13deb2f1aa6c1558cf8869c3e71453f0c70`、
  `a697218d3d94471e3db3fe85ad61ff4d762ce7a2546bc478ba2b4e244be33e45`。
  00:19 UTC fresh API复核时40K/27K仍分别为最新完整PT，KPL summary step=27988；
  W&B全程只读。只使用空闲GPU5/6，结束后均约906MiB/0%；GPU0--2训练、GPU7现有
  KPL 26K localhost inference和其他34100--34102服务均未操作。

## CORL79+debug30 canonical D435 URDF depth外参替换（2026-08-04）

- 用户确认上一条formal run `4kf71d6g`使用了不需要的depth外参。fresh W&B config与
  quaternion反算均确认其实际为translation=`[0.01,0.01,0.44]` m、RPY约
  `[1,27,1]` deg；该run在completed iteration 63停止，未到save-1000边界，因此没有
  formal checkpoint。W&B保持`finished`并标记`formal_invalid=1`、
  `stop_reason=user_requested_replace_camera_extrinsics`，现在明确
  `superseded_by=sx9wctkd`，不得resume或作为正确相机实验使用。
- `camera_depth_d435i`现在固定canonical G1 URDF `d435_joint`：translation=
  `[0.0576235,0.01753,0.41987]` m，mount RPY=`[0,0.8307767239493009,0]` rad=
  `[0,47.6,0]` deg，对应xyzw quaternion=
  `[0,0.40354529635239006,0,0.9149596678498247]`；sensor frame quaternion仍为
  `[-0.5,0.5,-0.5,0.5]`。同时修正strict Warp depth manager：sensor build与reset
  jitter的基准RPY从configured mount quaternion归一化后推导，不再使用隐藏的
  `[1,27,1]`常量；noise、frame transform和depth preprocessing未改变。perception
  manager全套48 tests通过，constructor smoke精确读回47.6 deg。
- 最终immutable source为
  `src-bab9ec5d0e2c9f61255442db7e21168303000bf7ae458ac45f37f393677ae33d`，archive
  SHA256=`7b56b4acbf5de52854794818973227d3826550dd743011457449512a8452ab2f`。
  exact 8-node/64-rank/2048-env-per-rank canary完成2/2 update并自然退出，第二轮
  101,515 steps/s；同iteration `model_00002.pt/.onnx`通过checker、ORT CPU load与
  PyTorch parity，max abs error=`1.043081e-7`。canary pair只作验收，不用于正式初始化。
- 新formal W&B run为`zihanw22/carry-any/sx9wctkd`：
  <https://wandb.ai/zihanw22/carry-any/runs/sx9wctkd>，name=
  `pure_rl_corl79_debug30_precomputed_turn_forward_d435urdf_ws64_e2048_onnx_20260804_002553`。
  节点仍为`zzzihanw-17/25/26/28/30/34/47/93`，每台8张L40S，world64、每rank
  2048 env、global131072 env，tmux=`hs_form_d435_sx9wctkd`。除三项显式camera CLI
  binding外，TRAIN_ARGS与被替换run完全相同：fresh pure PPO、109条CORL79+debug30
  x-only/yaw-only bank、无teacher/distill/BC/DAgger/resume/policy-init、offline contact
  guidance权重0、real visual/depth mesh、collision=`convex_decomposition`、无fallback，
  actor/critic `[512,256,128]`，24 steps、7 epochs、4 minibatches、LR `1e-3`、40K目标。
- `training.export_onnx=true`、save interval 1000、upload未跳过；每个required boundary
  必须原子发布并上传同iteration PT+ONNX pair。启动验收时尚未到iteration1000，故当前
  不能把canary pair误报成formal checkpoint。Rule-90使用最终source/config的canonical
  `box_10`重新录制，H.264 1280x720@50fps、368帧/7.36s，SHA256=
  `91dd81874cded55ef386f2fc83414ea988cc78418ca267e3cc05c473e1960783`；ffprobe、完整
  decode与人工抽帧通过。它已成为W&B step3唯一history-backed `vis/replay`，fresh API
  验证恰好1条media history和1个MP4。
- 00:50 UTC有界formal验收时8/8 tmux存活、每节点8/8 GPU app、64个worker的
  Traceback/ChildFailed/OOM/DistBackend/NCCL/Gloo/non-finite/Xid均为0。fresh W&B已复核
  19个finite numeric update；step18为166,853 steps/s、collection=`15.5665s`、
  learning=`3.2867s`，distill/dagger/BC/replay-BC全0、PPO coeff=1、distributed
  loss-weight sum=64。W&B config精确读回新translation/quaternion、
  `far_tracking_warp` mesh depth、pure-RL provenance及ONNX/save合同。
- 完整审计root为
  `/home/ubuntu/FAR/holosoma_runs/formal_corl79_debug30_precomputed_turn_forward_d435urdf_ws64_e2048_20260804_000220/`。
  run contract、replay manifest、canary acceptance、formal worker和formal acceptance
  SHA256依次为`0cd589aa53b0419c9128f5dd9ca7d131f1dc32139ba945fb16eb66f618edea58`、
  `017bc57af78c5b6e4d1314aa8f72eecf931514d605acebbf53814febc3cb3886`、
  `967624fa815a92d4f701a68670e4769084916ce0f0fff441ce39450de33e8302`、
  `af0a827bc20cc6cc5af69ae990c0699fdc5b013762a903e07053cd580f69434d`、
  `c8b61c260c2a361d428418239feaa22b63e02dc875099d80b488f48144841b77`。

## XM 40K vs KPL 28K lift后恒定0.15/0.50与真实policy-input纠正评估（2026-08-04 01:05 UTC）

- 用户指出上一批错误使用了checkpoint-native command；该`08-04-0019`目录已在
  README与`WRONG_PROTOCOL_DO_NOT_USE.md`明确标为错误协议/已作废，只留审计，不能
  再用于constant-forward结论。正式纠正重新通过fresh W&B只读冻结
  `xm0hda83/model_40000.pt`（SHA256=
  `62a33b7e7417e8f6296d8acd0bd314e1281ef07897ae3627262bb36d4e4bd86d`）与纠正请求时
  KPL最新完整`kpl2p2gn/model_28000.pt`（SHA256=
  `e9ce4b8a761fce2b94db76a74d439fc867643c778558881c91c85e5119bc8f03`）。内部
  completed/next为39999/40000与27999/28000，均856个tensor且全部finite；actor合同
  仍为command3+drop1+proprio/actions90=94维，另有58x87 model-ready depth与29D action。
- 正确矩阵为2 checkpoint x 3个各自in-bank固定代表 x 2档，共12条。XM继续使用
  `unscale__any_ball_29/scaledown__any_barrel_25/unscale__any_bin_29`；KPL使用
  `noscale__any_ball_3/noscale__any_barrel_12/noscale__any_bin_32`。单环境、seed42、
  timestep0、501步@50Hz、randomization disabled、initial-pose noise0、所有自动/motion/
  bad-tracking reset关闭。两个bank没有重叠motion ID，跨policy只比较相同category，
  不冒充same-init严格A/B。
- 两份policy统一使用evaluation-only external lift gate：配置时manual command与drop
  强制清零；物体world-z相对配置基线首次达到`+0.30 m`（`consecutive_steps=0`）后，
  分别永久锁存robot-heading relative-pose `[0.15,0,0]`或`[0.50,0,0]`到末帧。
  没有heading lock、lateral/yaw、NPZ schedule或reference后段drop；所有12条逐帧
  `dy=dyaw=drop=0` bitwise exact。10条触发任务均通过zero-prefix/constant-suffix
  bitwise审计；同checkpoint/clip的0.15/0.50在command前action逐帧bitwise identical。
- XM ball/barrel/bin两档首次非零actor step均为`79/79/81`。0.15档3/3 strict-end-carry，
  post-command carry fraction=`0.910/0.858/0.831`；0.50档0/3，fraction=
  `0.396/0.370/0.376`。KPL ball/barrel首次非零step=`123/107`，两档均未strict-end-carry；
  0.15 fraction=`0.397/0.396`，0.50=`0.087/0.556`。KPL bin两条配对轨迹最大相对抬高
  都为`0.2873 m`，未过0.30m gate，actor command因而全程正确保持零；这是pre-lift
  failure，不能降低门槛或把它计入0.15/0.50效果。
- 每条底部面板直接使用actor forward前保存的真实`actor_obs_raw_values[0:3]`、drop、
  完整94维raw/normalized统计、完整`perception_obs_values[5046]` reshape 58x87 depth、
  29D action与dynamic G1/object world-XY trajectory。人工检查XM-ball step78->79与
  KPL-barrel step106->107切换、KPL-bin六时刻不触发，以及4x3六时刻master；物体、
  depth和真实跌倒/失持均连续，没有剪裁或伪造触发。
- 交付共33个MP4：12 raw、12 individual input panel、4个单policy/command 3x1、2个
  单policy 0.15-vs-0.50 2x3、2个单command policy 2x3及1个4x3 master。全部单
  H.264/yuv420p、50fps、501帧/10.02秒，逐文件ffprobe与完整decode通过。byte-exact
  目录为
  `/home/ubuntu/FAR/_check_vis/08-04-0103__xm0hda83_model40000_vs_kpl2p2gn_model28000__post_lift_forward015_vs_050__policy_input/`；
  master SHA256=`a1c85179799c697d22993929e7c73134fb792b1340fc8c30c0edac371e1e1e0b`。
- 完整审计root为
  `/data/holosoma_eval_audits/xm40000_vs_kpl28000_post_lift_forward015_vs_050_policy_input_20260804_004259/`。
  frozen/evaluation/scheduler/media/W&B/visual/delivery manifest SHA256依次为
  `34af97fdeec7867ea244b1b371ff9646b5c968b5f9249db8b4a071882f56179a`、
  `c19cbc23c1a605f54d1a45c25a4a06aacaba5fcf5ed30d3a110b1ab493ea7f13`、
  `bc75a37c7eb9ca70b348969cfacbdeaf1a554411a523e8365f6de8ba4412e0ba`、
  `c90c5b9691feb1a6a232d141d8a0f1d5b81dc92d9d468ed001c35f9c8bb9475f`、
  `1fa280f684c054126960de6a33ccec962eab021e62552a8ee0538c7ec558661f`、
  `6e74aca38628d16dd5cd5712201f9dfde52fabb17e14628037315be1f47bdd56`、
  `a20b8b3f44b2a29a51b63dc741fc29da81ade9861ce5dac941d068862131ffcf`。
  01:04 UTC fresh W&B复核时40K/28K仍分别是最新完整PT，KPL summary step=28492且
  run仍running；全程W&B只读。只使用空闲GPU4/5/6，评估进程已全部清理，未操作
  GPU0--2训练或现有localhost inference。

## KPL 37K vs SX 5K严格same-init、lift后pure-forward与真实policy-input评估（2026-08-04 15:12 UTC）

- 用户要求用KPL最新checkpoint重新对比`zihanw22/carry-any/sx9wctkd`。14:46 UTC
  fresh W&B只读冻结时，KPL仍running、summary step=37684，最新完整为
  `kpl2p2gn/model_37000.pt`（SHA256=
  `d5fd80309247826e1a90515275693e95c9cfb7519332e672400442c3f76d6d7a`，内部
  completed/next=36999/37000）；SX仍running、summary step=5231，最新完整pair为
  `sx9wctkd/model_05000.pt/.onnx`（SHA256分别为
  `e057ada05a213de5e44d7a44b243f550c8b27f09f13530be46ec5f706d67bda0`、
  `4c883b8f887d405143fafb77369a7952917ca5457f0e6a272ae72ff85f5f7c4c`，内部
  completed/next=4999/5000）。SX ONNX通过checker和ORT CPU load，输入为94D actor+
  5046D perception、输出29D action；KPL仍按正式启动时记录的历史PT-only例外处理。
- 两份policy严格复用同一CORL79固定初始化：`noscale__any_ball_3`（0.5 kg）、
  `noscale__any_barrel_12`（1.5 kg）、`noscale__any_bin_32`（1.0 kg）。motion bytes、
  URDF、object map、seed42、timestep0完全相同；三个motion SHA256为
  `b47f046448bc683771042807644cb6ee70732b322a2a80cca566e3bf5bbb5def`、
  `61b145e29a7760eab8b36702043c412dc08c8cd75e5d58ec04fabd00e939310b`、
  `2310a82299e0c344e5777aada27f1b737c5a86215b76632c63e94989b4e41ff6`。
  这是strict same-init A/B，但相机不是统一消融：KPL checkpoint合同为旧D435 offset
  `[0.01,0.01,0.44]`、mount约`[1,27,1]°`；SX为canonical D435 offset
  `[0.0576235,0.01753,0.41987]`、mount `[0,47.6,0]°`。两者均按各自训练输入合同
  inference，必须保留该comparison boundary。
- 评估矩阵为2 checkpoint x 3 init x 2 command，共12条。单环境、randomization
  disabled、501步@50Hz；actor在物体world-z相对配置基线不足`+0.30 m`时严格收到
  `[0,0,0]`，首次过门后分别永久锁存robot-heading relative-pose `[0.15,0,0]`或
  `[0.50,0,0]`到末帧。`consecutive_steps=0`，不用NPZ/native command或heading
  lock，全部逐帧`dy=dyaw=drop=0` bitwise exact。同policy/clip两档在command前
  action逐帧bitwise identical；94D actor、58x87 model-ready depth和29D action均在
  actor forward前真实保存，不从config反推。
- KPL的ball/barrel/bin两档均过gate，首次非零actor step为`95/107/137`；SX barrel/bin
  为`107/131`，SX ball两档最大相对抬高都只有`0.000263 m`，故command全程保持0。
  KPL 6/6、SX 4/6触发；12条均未strict-end-carry。KPL 0.15的post-command carry
  fraction为`0.342/0.398/0.442`，0.50为`0.140/0.094/0.357`；SX 0.15为
  `none/0.378/0.384`，0.50为`none/0.246/0.395`。这只是固定三样本诊断，且camera
  contract不同，不得外推full-bank robustness或把差异纯归因于weights。
- 录制初次尝试时GPU3--5被三条无关Curiousity Isaac训练各占约25GB；四个worker在
  rollout前出现PhysX显存申请/articulation startup错误。确认四条均0 metrics、0
  policy-I/O、0 video后移入
  `retries/attempt1_startup_gpu_contention/`，随后保持其他训练不动，只在GPU6串行重跑
  12/12。失败attempt manifest SHA256=
  `f244d8db76b444fa9c71b6547ad58928141f2c45c9ae28356d37ffde243e6d90`，没有任何失败
  产物进入科学结果或交付。
- 共12 raw、12 individual input panel、8 grouped comparison和1个4x3 master，合计
  33 MP4；全部单H.264/yuv420p、50fps、501帧/10.02秒，ffprobe与完整decode通过。
  六时刻master和全部12组切换边界经人工抽帧，单G1/正确物体、真实depth、轨迹、
  SX-ball零command及后段失持连续可见。byte-exact交付目录为
  `/home/ubuntu/FAR/_check_vis/08-04-1441__kpl2p2gn_model37000_vs_sx9wctkd_model05000__same_init_post_lift_forward015_vs_050__policy_input/`；
  master SHA256=`41bb3438551dd48412ea6232e2ca6394d18ce403d176944e7fec2af50821cf12`。
- 完整审计root为
  `/data/holosoma_eval_audits/kpl37000_vs_sx05000_same_init_post_lift_forward015_vs_050_policy_input_20260804_144119/`。
  frozen/matrix/scheduler/evaluation/W&B/visual/delivery manifest SHA256依次为
  `0f8faf81b150313f5f7ba0b6bbb3a38a967c0db3c8d88fece4aeeef85c8679a0`、
  `017794396d12f6d5ed48ed6f6e417c3ef782d8f9d181be113f99d6637057ba8a`、
  `b45f494914373d4168cced03ba1ff00e70d7238e842e279f8a9993ec6ed5ae7b`、
  `6ce9a69722dd26470db6dc0618bb91f5ec813359f066cdc09feb31e1e9c82b01`、
  `738f10212d9eeb60658418478b82dafbea9647fde3a3e8bac8e5947eff6c8f4a`、
  `02410a0f17c526dc7c9a38202ce795c624c64edbf5da0a128bd7cf985a6794c7`、
  `b5d29e0833c370c25810f9e7ed5ed17ef78e14db3f934bf60e6d788b4511a9eb`。
  15:11 UTC fresh API复核时SX仍为5K最新完整pair；KPL已新增38K（15:06上传），但本批
  按request-time freeze保持37K。W&B全程只读，无媒体上传或run lifecycle修改；评估
  结束后没有残留本批进程，且未操作现有localhost inference。

## KPL 38K localhost Viser interactive inference（2026-08-04 15:45 UTC）

- 15:19 UTC fresh W&B只读冻结`zihanw22/carry-any/kpl2p2gn/model_38000.pt`，远端
  size=`10352929`、MD5 base64=`m8x2j3ov+Fl6dgz38Bhd8w==`，本地SHA256=
  `c3d3616cb895621d52ea3a4f0a041cee9e811a66238cd44117c6ad29825ce2b5`；checkpoint内部
  completed/next=`37999/38000`、856 tensors/2,451,687 numel且全部finite。15:44 UTC
  fresh API复核时run仍running、summary step=38411，38K仍是最新完整PT。W&B全程只读。
- 正式服务为tmux=`hs_kpl2p2gn_model38000_native_inference`、PID=`1782092`、physical
  GPU6、`0.0.0.0:34103`，入口`http://localhost:34103`。它已替换并停止旧的
  `hs_kpl2p2gn_model26000_native_inference`/PID 758629；训练GPU0--2和训练进程未改动。
  runtime明确从immutable source
  `src-6b846ee72a1246372fd6d9abfb1fa0dbb521ba904b6fa690a85cc551a5f31a34`加载38K actor，
  不是teacher、reference replay或ONNX；KPL仍按启动时记录的历史PT-only例外处理。
- motion/command保持checkpoint-native CORL79：view digest=
  `f7f2a9e78519318a549c4ac7c184e7e9de2e3c01e6e2a316d42fd1285b4b3a67`，完整79个clip和
  对应URDF，初始`box_10`；`Advanced / Clip Playback / Clip / Apply Clip`可切换。
  command为`precomputed_turn_then_forward`，runtime验证zero/forward/yaw frames=
  `7112/16488/2496`、`dy_always_zero=true`、无dx/dyaw overlap，并保留runtime pickup
  latch。canary实际把`box_10`切到`box_20`，visual/collision和ground status同步通过。
- reset严格为实际eval rollout合同：`HOLOSOMA_EVAL_MANUAL_RESET_ONLY=1`及auto/motion-end/
  bad-tracking reset三项disable均从正式PID环境读回；clip结束clamp，不自动reset。正式
  WebSocket用真实ViewerCamera handshake触发`Simulation Control / Reset`，日志marker从
  0变1并恢复`box_10`正确ground status，额外观察10秒仍为1。79项dropdown、Apply Clip和
  Reset均从正式端口的persistent GUI messages重新解析通过。
- 真实policy-I/O probe保存24条step 0--23：actor raw/norm 94D、model-ready perception
  5046D、action/torque 29D，全部numeric finite；696/696 action值非零、max abs action=
  `5.1822638511657715`，torque saturated joint count最大0。HTTP=200、正式log中
  Traceback/KeyError/fatal error均为0。
- 完整root为
  `/home/ubuntu/FAR/holosoma_runs/kpl2p2gn_model38000_native_inference_20260804_151922/`；
  launcher/selection/contract/health manifest SHA256依次为
  `8c921ebf18e8b9d1fe3ffe79811cab0de7ac418b3d5b2c33880ca4be558a2197`、
  `226e5a918474c6418860d4139e72842939bf0f482b81704a703d27b52d49f037`、
  `9ebc3a5506e7cc6b6053c4f97f93146cb0ae1d0d75760605769e9d70ebeb5108`、
  `b49762abe9cef987e5a92601d2409b370c7cd51d29ee01d1345a47595a5974f1`。

## SX 5K localhost Viser interactive inference替换KPL（2026-08-04 16:33 UTC）

- 用户要求把既有KPL localhost服务换成SX最新checkpoint。15:56:59 UTC fresh W&B只读
  冻结时，`zihanw22/carry-any/sx9wctkd`仍running、summary step=5726，最新完整原生
  PT+ONNX+pair为`model_05000`。PT/ONNX SHA256分别为
  `e057ada05a213de5e44d7a44b243f550c8b27f09f13530be46ec5f706d67bda0`、
  `4c883b8f887d405143fafb77369a7952917ca5457f0e6a272ae72ff85f5f7c4c`；内部
  completed/next=`4999/5000`。ONNX checker、ORT CPU load及pair记录的PyTorch-vs-ORT
  parity通过，输入94D actor+5046D depth、输出29D action。16:31:28 UTC fresh API
  复核时run仍running、summary step=5949，5K仍为最新完整pair；W&B全程只读。
- runtime使用正式run绑定的immutable source
  `src-bab9ec5d0e2c9f61255442db7e21168303000bf7ae458ac45f37f393677ae33d`，1027项source
  manifest及asset symlink closure通过；未使用当前dirty checkout。motion使用SX原生
  CORL79+debug30 exact109 bank，object map SHA256=
  `70b466aad04837a79f6dd0f4491cb345a73c687209981acd3eb7f4a0365d8f5c`。仿真、clip map和
  Warp depth均确认109/109个真实URDF/mesh，viewer geometry mesh-set SHA256=
  `846668e62c69baa67c3e60e95a94432ac778e5599e20684f762b4e4d50dfc15f`，camera/robot
  bindings exact。
- policy为SX 5K pure-PPO checkpoint actor，不是teacher、BC、distill、reference replay或
  manual-forward override。command保持checkpoint-native
  `precomputed_turn_then_forward`，runtime统计zero/forward/yaw=`8888/24110/2668`，dy恒0、
  dx/dyaw不重叠并保留`object_root_rel_z_v1` pickup latch。相机严格使用canonical D435
  translation=`[0.0576235,0.01753,0.41987]` m、mount=`[0,47.6,0]°`、frame quaternion=
  `[-0.5,0.5,-0.5,0.5]`、`far_tracking_warp` mesh depth。
- canary从persistent GUI messages解析出恰好109项dropdown，实际由`box_10`切到末条
  `unscale__any_bin_31`；visual/collision均切到
  `/World/envs/env_0/Object_108_unscale__any_bin_31`，extents约
  `[0.266,0.338,0.266]` m，sim/ref collision bottom均`+0.003121 m`，随后恢复`box_10`。
  Reset合同仍为实际eval rollout要求：auto/timeout/motion-end/bad-tracking reset全部关闭；
  内部34104和外部34103各用真实ViewerCamera handshake触发一次显式Reset，日志marker
  共2且settle后不增加，Traceback/KeyError/OOM均0。
- 真实policy-I/O probe保存24条step0--23：actor raw/norm 94D、model-ready depth 5046D、
  action/torque 29D全部finite，696/696 action非零，max abs action=
  `2.3714661598205566`，最大torque saturated joint count=0。SX simulator为PID 1836079、
  physical GPU7、内部`0.0.0.0:34104`；为保留已完整加载并验证的42GB/109-object进程，
  外部使用byte-transparent TCP forwarder PID 1872679把`0.0.0.0:34103`转到34104，HTTP与
  WebSocket读写均实测通过。正式入口仍是`http://localhost:34103`，tmux分别为
  `hs_sx9wctkd_model05000_native_inference`与
  `hs_sx9wctkd_model05000_localhost34103`。
- 旧KPL 38K tmux已停止、PID 1782092已退出，GPU6释放；GPU0--2训练进程未操作。完整审计
  root为
  `/home/ubuntu/FAR/holosoma_runs/sx9wctkd_model05000_native_inference_20260804_155659/`；
  launcher/validator/proxy/selection/contract/health SHA256依次为
  `ac1bf63820eeda7e80f7406a54effeea11a39e6026067b7256e69e2101b39b0f`、
  `14dacc4469c3851327a5e39f1dcd87c6d483e6907a3a96ed98a5ac1755c1f656`、
  `45f770a35f2266666a7f33bb4cad0fc808dc06086b00a24fb1eb0d00c519eab0`、
  `4f3ee863bc3919da9082882c00fb984444a0cae9e33924e468deed155fdc204f`、
  `65db9034bd46c7461db34776a860ae5a5837eca5bd9f0af4e1c0f50c0ae0bc0c`、
  `4584ab1be4e46c8ac42105c94a46d20acbbb8647935b7f7414fd1fd8a4bf88e9`。

## SX 5K barrel motion真实actor-command虚拟键盘Viser（2026-08-04 17:30 UTC）

- 用户要求在一条barrel motion上直观显示训练同语义的实际policy command。服务仍为
  `zihanw22/carry-any/sx9wctkd/model_05000.pt` checkpoint actor（completed=4999，PT
  SHA256=`e057ada05a213de5e44d7a44b243f550c8b27f09f13530be46ec5f706d67bda0`），并保留
  same-iteration原生ONNX SHA256=
  `4c883b8f887d405143fafb77369a7952917ca5457f0e6a272ae72ff85f5f7c4c`。使用原正式immutable
  source `src-bab9ec5d0e2c9f61255442db7e21168303000bf7ae458ac45f37f393677ae33d`
  和exact109 motion/object bank，不是teacher、distillation-label rollout、reference replay或manual
  command override；W&B全程只读。
- 新的隔离inference identity为
  `/home/ubuntu/FAR/holosoma_runs/sx9wctkd_model05000_barrel_command_keyboard_20260804_170437/`。
  默认clip是`noscale__any_barrel_35`（index32，319帧），底部展开的
  `Actual Actor Input · 虚拟按键（只读）`面板从
  `PPO._pre_eval_env_step`完成checkpoint actor action的同一份observation直接取
  `actor_obs_root_contact_aware[dx,dy,dyaw]`与`actor_obs_drop_button`，同时显示
  raw physical slot和checkpoint-normalized slot。面板不写入command/action；W/S对应±x、
  A/D对应±y、Q/E对应±yaw、G对应drop，绿色仅表示该actor帧实际非零。
- 端到端跑完该clip后的transition log覆盖motion frame0--318，观测到的键为
  `E/Q/W/G`；pickup latch前command/drop全零，首次非零为frame109的E
  (`dyaw=-1.2681308`)，随后frame110为W (`dx=0.150000006 m`)，后续Q/E与W
  交替，frame298为W+G，frame318为G。全16条状态切换记录中`dy`逐条bitwise
  为0、dx/dyaw从不重叠、manual override从未打开，因此A/D/S均未亮起。另存
  24条真实policy-I/O probe：94D actor、5046D depth、29D action全finite，696/696 action
  value非零，最大abs action=`7.003119945526123`，最大torque saturated joint count=0。
- Viser persistent messages已用有效`ViewerCameraMessage` handshake重建：109项clip
  dropdown、默认barrel、root最下方order51的expanded键盘面板与实时markdown均通过。
  交付时已显式Reset并单步刷新，停在frame0，`Play=false`、pickup waiting、
  command/drop全零。reset仍为实eval合同：auto/timeout/motion-end/bad-tracking全关，
  clip-end clamp；两个reset marker都来自本次明确的Viser操作，随后10秒计数2->2，
  没有自动reset。
- 当前正式进程为tmux=`hs_sx9wctkd_model05000_barrel_command_keyboard`、PID
  `1925489`、physical GPU7、internal34104；已验证的byte-transparent proxy PID
  `1872679`继续将external34103转到34104。内外HTTP均200，入口为
  `http://localhost:34103`。新启动marker `2026-08-04T17:12:02Z`之后Traceback/
  no-space/OOM/diagnostic load failure/segfault/CUDA error均0。
- 首次重启在actor加载前因旧runtime cache继续写根盘而`No space left on device`，
  PID1920306已停止，没有产生科学结果。没有删除该cache，而是可恢复地隔离到
  `/data/holosoma_eval_audits/cache_quarantine/sx9wctkd_model05000_canary_usd_cache_nospace_20260804_171035/`；
  成功实例的active cache在
  `/data/holosoma_eval_audits/sx9wctkd_model05000_barrel_command_keyboard_20260804_170437/runtime_cache/`。
  checkpoint/source/motion/训练进程均未改动。
- wrapper/launcher/validator/selection/contract/health SHA256依次为
  `d164ce78b2790ba0dd7d66bb2264ba91f9a3c0354e955e38582fea2e594e6c65`、
  `073a610c18884f62552c7b22da44d84311e470fed37fa5f2dac3e0a9cc610fa6`、
  `fdfe27faa8b79d97b41391ec7bccdfdd165d661974f8ec23f9a2a1c5b3f4fc6e`、
  `4f3ee863bc3919da9082882c00fb984444a0cae9e33924e468deed155fdc204f`、
  `438ddde759a7487e8ce9feba7b6a055b75928f7a5a3989bd26491df9f4e35d89`、
  `d6c3657089bdda15bb67d0445fbabc8064b4d95489bb55f2fa677e1f3647f07b`。

## CORL79+debug30 drop-exclusive 32-GPU pure-RL formal run（2026-08-04 17:44 UTC）

- 新建正式W&B run `zihanw22/carry-any/0mcqao8k`：
  <https://wandb.ai/zihanw22/carry-any/runs/0mcqao8k>，名称为
  `pure_rl_corl79_debug30_dropzero_precomputed_turn_forward_d435urdf_ws32_e2048_onnx_20260804_171221`。
  这是与`sx9wctkd`对照的fresh pure PPO；没有resume、policy init、teacher、distillation、
  BC或offline contact-guidance。actor command继续使用预计算的turn-then-forward，但drop
  有效时强制actor最终输入恰为`[dx,dy,dyaw,drop]=[0,0,0,1]`。
- effective bank是CORL实际79条与debug30合并后的exact 109条：box25、ball9、barrel36、
  bin39。immutable data closure SHA256为
  `3219b4de13c6eb7f89dbd1f46619f9fdcf7aca75fb02473798423bc0cf2c158f`，object map
  SHA256为`70b466aad04837a79f6dd0f4491cb345a73c687209981acd3eb7f4a0365d8f5c`。
  运行时使用original real visual/depth mesh和real-mesh convex-decomposition collision；
  geometry缺失时fail closed，禁止cuboid fallback。ws32 shard对109条做exact-once partition，
  没有asset或DataLoader二次shard，manifest SHA256为
  `19500fe84e4fef7c70cadc574b2581a41f1e85b5bf5e6cede7fa58a57ab8c858`。
- depth camera不是49度，而是`sx9wctkd`使用的canonical 47.6度：parent=`torso_link`，
  translation=`[0.0576235,0.01753,0.41987]` m，RPY=
  `[0,0.8307767239493009,0]` rad，mount quaternion=
  `[0,0.40354529635239006,0,0.9149596678498247]`，frame quaternion=
  `[-0.5,0.5,-0.5,0.5]`，preset=`camera_depth_d435i`，strict source=
  `far_tracking_warp`。
- topology为ap-northeast-2a的4台机器、每台8张L40S、共32 ranks：
  `zzzihanw-15/27/35/42`；每rank 2048 environments，global 65536。四台的tmux均为
  `hs_form_dropzero_0mcqao8k`。冻结源码为
  `src-309efb4303690cea9960b10a2e459041608243842e284f2732cd27f0cccbca57`，正式审计root为
  `/home/ubuntu/FAR/holosoma_runs/formal_corl79_debug30_dropzero_d435urdf_ws32_e2048_20260804_165326/`。
- exact-topology两步canary原生生成同iteration PT+ONNX；PT/ONNX SHA256分别为
  `51dea17baec6abce43f9f2aa852898fa91670dbe0a18621a01c10c8d1d601717`和
  `f8a3ea6321a3087649398184dd843cfd34c4ce0813c23ed88a78cbaaa901aa8e`，ORT最大绝对误差
  `8.568167686462402e-8`。正式contract固定`training.export_onnx=true`、每1000 iteration
  原子保存并上传PT+ONNX、任一失败即fail closed；验收时正式run尚未到iteration1000，不能把
  canary pair称为正式checkpoint。
- Rule-90 canonical `box_10` replay已完整通过并写入该训练run的history step6；H.264视频为
  1280x720、50 FPS、368帧、7.36秒，SHA256=
  `796667afa1c1a5e27b0395a9ab8d8d2c8be3d00a2a6d207f593629736874f1b2`。fresh API复核
  只有一个history-backed `vis/replay` MP4，没有prebind副本。
- bounded acceptance于W&B step18通过：total FPS=`95148`、collection=`13.489 s`、
  learning=`3.041 s`、distributed weight sum=`32`，所有distill/dagger/BC系数均0；32个
  rank和GPU application均存活，无exit marker和fatal Traceback/OOM/NCCL/Gloo/segfault。
  acceptance SHA256为
  `eb51015de948ae2b86fd4d75a621f256fe414549e9180fbb09d57ceef45400ea`；验收完成后未安装
  recurring monitor，也未修改live immutable contract。

## CORL79+debug30 turn-forward bank跨节点同步（2026-08-04 19:47 UTC）

- 同步对象是当前formal run实际使用的derived 109-bank，而不是旧
  `corl79_plus_debug30_realmesh_categorymass_v1`：payload digest=
  `307e9662d498bd507b9d17ca9abf74a3654f7bf66ac6ab989c6f19c3889bddef`，manifest/object-map
  SHA256分别为`2de9ee5ca188b70e877c32dd9f0d2975eea99d11aa077bb077cf06ea9ab897bb`和
  `70b466aad04837a79f6dd0f4491cb345a73c687209981acd3eb7f4a0365d8f5c`。每台最终为
  109 NPZ、109 URDF、81个真实OBJ和109个contact clip目录；禁止geometry fallback。
- 新增`cp_corl79_debug30_turn_forward.sh`，不改变旧`cp_corl79_debug30.sh`语义。新脚本对
  已有bank按manifest的1994条generated records逐文件验证size/SHA，加manifest共1995文件；
  缺失bank先验证903,331,462-byte archive SHA256=
  `9413b7ea54d40cb57c4f5ffd6d4ed5061187faad1c0b3f17380ef3f000e71be2`，再同文件系统临时
  解包、完整验证和原子rename。错误已有target拒绝覆盖，所有同步任务用nice19/ionice-idle。
  installer SHA256=`8466a814050a7dab792eb33e1db4e4059243ef38650f2c26e681de3fa3e169d6`。
- 同时发布并安装本次world-size32、2048 env/rank shard view。NFS shard archive只有
  29,022 bytes，SHA256=`4d72c044a97dbd45fcbfd903afc69309bef235f9b0720df3b148945f0e9d800c`；
  shard manifest SHA256=`19500fe84e4fef7c70cadc574b2581a41f1e85b5bf5e6cede7fa58a57ab8c858`。
  每台复核32个rank目录、109个NPZ相对链接、global exact-once、0 broken link和0 incoming
  目录；该view只用于ws32，不应被其他world size误用，也不允许再做asset/DataLoader二次shard。
- AWS当前发现30台`zzzihanw-*`/`z1hanw` running节点。28台已独立post-check为完全一致：
  `z1hanw`、`zzzihanw-15/17/25/26/27/28/30/31/34/35/39/40/42/45/46/47/49/65/71/72/79/93/102/c/e/f/z`。
  `zzzihanw-40`没有NFS mount，使用控制节点直传且验证同一archive；`zzzihanw-71`的
  `/data`初始为root-owned，使用sudo只创建专用data/sync目录后成功安装。
- 两台没有冒险处理：用户原28节点清单中的`zzzihanw-66`处于EC2
  system=`ok`/instance=`impaired`且TCP22持续超时；新增但不在原清单的`zzzihanw-7`
  使用旧SSH key，当前controller key被拒绝且本机IAM无EC2 Instance Connect权限。没有为同步
  reboot/stop任何instance或训练进程。同步后正式run `0mcqao8k`四个tmux仍存活、每台8 GPU
  process、四台一致到completed iteration849、fatal marker为0。
- 完整操作日志和machine-readable结果位于
  `/home/ubuntu/FAR/holosoma_runs/sync_corl79_debug30_turn_forward_20260804_193235/`，其中
  `sync_manifest.json`记录28个成功节点与两个基础设施例外。

## KDW terminal 40K checkpoint-native三物体录制（2026-08-04 19:56 UTC）

- 用户要求补录`zihanw22/carry-any/kdw7jhze`最终checkpoint。fresh W&B只读选择确认run
  已`finished`、summary step=`39999`，terminal为`model_40000.pt`，SHA256=
  `17d5ba030c3a7486258c874ea05013fd7142ec33ce042445e42431171b0e7942`；checkpoint内部
  `iter=iteration=39999/next_iter=40000`，341个tensor、31,758,432个值全部finite。
  同iteration `model_40000.onnx`存在，远端1K--40K的PT/ONNX iteration集合严格一一对应。
- 这是40K pure-BC student的checkpoint actor rollout，不是teacher rollout或reference replay。
  按训练合同使用native contact-aware reference carry window门控的robot-heading tracking-error
  `[dx,dy,dyaw]`，没有任何manual-forward override。沿用35K批次完全相同的original30
  ball/barrel/bin motion、URDF、质量、seed42、timestep0、无随机化、零initial-pose noise、
  501步@50Hz和关闭reset的协议。
- ball/barrel/bin均完成pickup，触发时间分别约`1.51/1.59/1.53 s`，物体净XY位移约
  `1.279/1.863/0.971 m`，最大相对z约`0.536/0.474/0.517 m`；三条strict-end-carry仍为
  `0/3`，平均post-trigger carry fraction=`0.3532`。actor tracking-error逐帧公式最大误差
  `4.54e-7`，所有policy action finite。该结论只覆盖固定三个诊断clip。
- 交付目录为
  `/home/ubuntu/FAR/_check_vis/08-04-1947__kdw7jhze_model40000__native_tracking_error/`：1个
  3x1 master、3 raw、3个动态G1/object world-XY trajectory，共7个MP4。全部为单H.264
  stream、50fps、501帧/10.02秒，逐文件ffprobe与完整decode通过；0/2/4/6/8/10秒contact
  sheet人工确认正确单G1和ball/barrel/bin、连续pickup/carry/后段失持，无default pose、
  错物体或depth/frustum debug overlay。master SHA256=
  `5784f5140f922529024a9336b7cc8377fef2903a313b1a92a5460b68b573e0af`。
- 完整审计root为
  `/data/holosoma_eval_audits/kdw7jhze_model40000_native_tracking_error_20260804_194737/`；final
  handoff SHA256=`fabfc76a05027736a3c5d23dc89d5fcc859cf7a54b5734814c32e00e4c9655ce`。
  完成后fresh W&B复核仍为相同terminal PT+ONNX pair；全程无W&B写入，GPU3/4/5回到
  906MiB/0%，没有残留evaluation进程，GPU7上的SX Viser实例未被触碰。

## KDW terminal 40K抬起后恒定0.15非闭环command消融（2026-08-04 21:47 UTC）

- 用户在确认KDW native command为reference-root闭环tracking-error后，要求补一个非闭环
  版本。本批沿用相同`model_40000.pt`字节（SHA256=
  `17d5ba030c3a7486258c874ea05013fd7142ec33ce042445e42431171b0e7942`）和相同original30
  ball/barrel/bin初始化；不是teacher rollout或reference replay。fresh W&B复核run仍
  `finished`、summary39999，terminal PT/ONNX仍为40K一一配对，全程无W&B写入。
- evaluation override明确替换checkpoint-native tracking-error：实际object world-z相对初始
  达到`+0.30 m`后立即锁存actor前三槽为robot-heading relative pose `[0.15,0,0]`并持续到
  结尾；`consecutive_steps=0`。逐帧policy-I/O验证触发前三槽全零、触发后数值恒定，dy、
  dyaw和drop均exact zero。后续审计发现该历史manual-control路径还把`pickup_button`从actor
  step1起清零；因此这不是仅替换root command的干净消融。严格说lift gate仍是一次状态反馈，gate后的command不再读取
  reference或robot tracking error；该视频必须标成override ablation，不能标成native rollout。
- ball/barrel最大object相对z仅`0.00720/0.01735 m`，没有过gate，所以两条实际始终收到零
  command；bin在metric step269、约`5.45 s`首次达到`+0.30 m`，actor从step270起收到
  `[0.15,0,0]`，post-trigger object XY位移约`0.674 m`但carry fraction为0。固定三clip为
  1/3 trigger、0/3 strict-end-carry。原先据此写下的“native 3/3 pickup明显依赖reference
  tracking-error”结论已撤销：root-command替换与pickup/drop按钮清零发生耦合，不能归因。
  所有action finite；bin后段
  最大绝对action约29,882，按真实失败结果保留。
- `_check_vis`交付为
  `/home/ubuntu/FAR/_check_vis/08-04-2139__kdw7jhze_model40000__post_lift_constant_forward015/`，
  含1 master、3 raw、3 trajectory共7个MP4。11个审计MP4和7个交付MP4全部为单H.264、
  50fps、501帧/10.02秒，ffprobe与完整decode通过；六时刻人工抽帧确认正确物体、连续轨迹、
  ball/barrel未触发及bin晚触发后失持。master SHA256=
  `d55c4334b90d1bd89eba667ecce4788ad24d8611080b262d2f0d6a697aa8bcc8`。
- 审计root为
  `/data/holosoma_eval_audits/kdw7jhze_model40000_post_lift_constant_forward015_20260804_213909/`，
  final handoff SHA256=`01fb530e0cc47d6806c1612cfbcaf9f0a7076919240bd028e1941c2ca5aee29d`。
  GPU3/4/5已回到906MiB/0%，没有残留evaluation进程，GPU7的SX Viser没有被停止或改写。

## KDW terminal 40K仅替换root command、保留原生按钮的重录（2026-08-05 02:05 UTC）

- 为修正上一节的混杂因素，新增默认关闭的evaluation-only
  `manual_forward_after_lift_preserve_native_buttons`：manual root command仍替换actor前三槽，
  但`pickup_button`/`drop_button`继续从checkpoint-native reference-timed contact window产生；
  既有manual-control和训练默认语义不变。`record_checkpoint_inference.py`显式暴露对应CLI，
  command/observation单测覆盖配置状态及两按钮回退，定向组合为`64 passed`。
- 重录继续使用exact `kdw7jhze/model_40000.pt`（SHA256=
  `17d5ba030c3a7486258c874ea05013fd7142ec33ce042445e42431171b0e7942`）、同一original30
  ball/barrel/bin初始化、质量/URDF、seed42、timestep0、零initial-pose noise、关闭物理随机化
  与reset、501 actor steps@50Hz。实际object world-z达到初始`+0.30 m`后，下一actor step开始
  持续输入纯robot-heading relative-pose `[0.15,0,0]`，`consecutive_steps=0`；不是速度。
- 三条均成功达到实际lift gate：ball/bin为metric step79（actor step80，约1.65s），barrel为
  metric step83（actor step84，约1.73s）。policy-I/O将actor slots3:5与前一批checkpoint-native
  trace逐帧比较，三条均501/501完全一致：pickup active末帧分别为ball67/barrel68/bin66，
  drop active首帧分别为241/240/235；因此本批确实只更换外部root-command信号。
- 三条post-trigger object XY位移分别为ball `6.026 m`、barrel `5.017 m`、bin `4.900 m`；
  post-trigger carry fraction分别为`0.6303/0.6077/0.6825`。native drop cue及之后的真实失持/摔倒
  完整保留，所以strict-end-carry仍为`0/3`，不能把该末端门解释成未完成pickup。
- `_check_vis`交付目录为
  `/home/ubuntu/FAR/_check_vis/08-05-0158__kdw7jhze_model40000__native_pickup_drop__post_lift_forward015/`，
  含1 master、3 raw与3条动态G1/object world-XY trajectory，共7个MP4；完整审计root为
  `/data/holosoma_eval_audits/kdw7jhze_model40000_command_only_native_buttons_forward015_20260805_015803/`。
  11个审计MP4和7个交付MP4均为单H.264、50fps、501帧/10.02秒并通过完整解码；六时刻
  contact sheet人工确认三物体均先pickup再carry、后期失败未裁剪。master SHA256=
  `937bcd2b707cc20177933972f48a986995bbcf80dbdd840a4d6993627c528379`，final handoff SHA256=
  `691d881b446ac0e2482760687c3d02d4578037700f752308e1aa8055b777be94`。
- fresh W&B只读复核run仍`finished`、summary step39999、1K--40K PT/ONNX iteration集合严格
  一一对应且terminal两文件size/MD5未变；没有W&B写入。退出后GPU3/4/5均为906MiB/0%，
  没有残留evaluation进程，GPU7上的SX Viser未被触碰。

## KDW terminal 40K原生pickup、drop=0、抬起后持续forward重录（2026-08-05 22:10 UTC）

- 用户明确拒绝上一节保留native drop的版本；目标语义是只借用checkpoint-native pickup cue
  建立抓取，实际物体抬高`+0.30 m`后始终向actor输入纯`[0.15,0,0]`直到第501步，同时
  `drop_button=0`贯穿整个rollout。为避免再次耦合，evaluation-only接口进一步拆成独立
  `preserve_native_pickup_button`与`preserve_native_drop_button`，默认均false；本批只启用前者。
  训练与历史manual模式默认不变，command/observation/recorder定向组合为`68 passed`。
- exact checkpoint、ball/barrel/bin初始化、URDF/质量、seed42、timestep0、关闭randomization/
  reset及501步@50Hz均与前两批相同。三条`pickup_button`各自与checkpoint-native policy-I/O
  501/501 actor steps严格一致；三条`drop_button`均在step0--500 exact zero。lift gate分别为
  ball/bin metric step79（actor step80起forward，约1.65s）与barrel step83（actor step84起，
  约1.73s）；之后三维command逐步验证直到结尾始终为float32 `[0.15,0,0]`，dy/dyaw无其他分量。
- 结果为pickup `3/3`、strict-end-carry `3/3`，三条post-trigger carry fraction均为`1.0`；
  post-trigger object XY位移为ball `3.349 m`、barrel `6.997 m`、bin `6.694 m`。纯forward输入
  不等于世界航向锁定，policy自身仍可改变heading；实际robot yaw range分别为
  `191.86/78.61/47.11 deg`，视频与轨迹均未隐藏这一行为。
- 用户交付目录为
  `/home/ubuntu/FAR/_check_vis/08-05-2204__kdw7jhze_model40000__native_pickup_drop0__persistent_forward015/`，
  含1个3x1 master、3 raw和3条动态G1/object world-XY trajectory，共7个MP4。审计root为
  `/data/holosoma_eval_audits/kdw7jhze_model40000_native_pickup_drop0_persistent_forward015_20260805_220403/`；
  11个审计MP4与7个交付MP4全部为单H.264、50fps、501帧/10.02秒并通过完整解码，六时刻
  contact sheet人工确认三物体均从地面pickup并抱持到最后一帧。master SHA256=
  `ad20e975e6745d631628d35832f409c3d45a0fdd8ce276a6347765058564663c`，final handoff SHA256=
  `e6c21c62b23ce604ebb6482a48115e3fca7ee806da1336014ee6f49b400abdff`。
- fresh W&B只读复核仍为`finished`/summary39999，1K--40K PT/ONNX集合严格一一对应，terminal
  size/MD5未变；无W&B写入。退出后GPU3/4/5为906MiB/0%，无残留evaluation进程，GPU7
  的SX Viser未被触碰。

## CORL79+debug30 canonical-depth 4-GPU privileged teacher tracking（2026-08-05 07:17 UTC）

- 新建正式W&B run `zihanw22/carry-any/gu5d3qo8`：
  <https://wandb.ai/zihanw22/carry-any/runs/gu5d3qo8>，名称为
  `teacher_tracking_corl79_debug30_d435urdf_ws4_e1512_onnx_20260805_062906`。它是fresh
  privileged tracking teacher pure PPO：experiment为
  `g1-29dof-wbt-w-object-generalist-teacher-linvel`，actor/critic输入分别为178D/310D，
  MLP均为`[512,256,128]`，action为29D；没有resume、policy init、外部teacher checkpoint、
  distillation、BC、DAgger或offline contact-guidance reward。
- “同样的command”严格保留paired student环境中的`precomputed_turn_then_forward`、runtime
  pickup latch和drop-exclusive metadata，但该privileged teacher actor仍只读取其原生178D
  state/reference observation，不读取sparse root command或drop button；不得把这次teacher称为
  command-conditioned student。tracking reward覆盖reference motion全部阶段，offline contact
  guidance权重为0，真实物理contact仍参与仿真。
- exact bank为CORL实际79条加debug30后的109条：box25、ball9、barrel36、bin39。bank
  manifest/object-map/data-closure SHA256分别为
  `2de9ee5ca188b70e877c32dd9f0d2975eea99d11aa077bb077cf06ea9ab897bb`、
  `70b466aad04837a79f6dd0f4491cb345a73c687209981acd3eb7f4a0365d8f5c`和
  `3219b4de13c6eb7f89dbd1f46619f9fdcf7aca75fb02473798423bc0cf2c158f`。ws4 shard按
  `[28,27,27,27]` exact-once划分，无asset/DataLoader二次shard；visual/depth使用original
  real mesh，collision为real-mesh `convex_decomposition`，geometry缺失时fail closed。
- depth使用与paired student一致的canonical torso D435：translation=
  `[0.0576235,0.01753,0.41987]` m、pitch=`47.6°`、mount quaternion=
  `[0,0.40354529635239006,0,0.9149596678498247]`、frame quaternion=
  `[-0.5,0.5,-0.5,0.5]`、source=`far_tracking_warp`。depth在训练中真实生成并保留sensor
  noise、holes和reset pose randomization，但不注入teacher actor/critic；四个rank的独立
  `train_rank_*.log`均记录了有效camera runtime state。
- topology为`zzzihanw-39`（`10.99.0.39`）单机GPU0--3、world size4，每rank1512 env，
  global6048；tmux为`hs_teacher109_gu5d3qo8`。GPU4--7和其他节点上的既有进程未操作。
  PPO为24 steps、7 epochs、4 minibatches、adaptive LR（actor/critic初始`1e-3`，边界
  `[1e-5,1e-2]`）、40K iterations，每1000 iteration保存。
- 为不改变既有teacher preset，源码仅新增隔离的
  `g1_29dof_wbt_w_object_teacher_state_robust_with_camera`：继承teacher原物理/state DR，仅补齐
  paired student使用的camera setup/reset randomizers；定向测试为`63 passed`。正式运行使用
  immutable source `src-978fe563636934c6a38d16aa59357f2541b336e25211ea17e353aaa352e4a1cc`，
  source archive SHA256=
  `55eb0776a9ce89fd072c8b0bbd500bb3635e330655be48cd55e05e7e3791ed00`。
- exact-topology两步canary原生生成同iteration PT+ONNX；PT/ONNX SHA256分别为
  `bfe28181a4aeb46d9e128e40f24bea28603f8f9d5b12276ce9c0402153bb551b`和
  `1a8e7f249a63d4a081d5b463b8fba4d85d4ddb265aec920a16a0ad421e5a0e3d`，ONNX checker、
  ORT CPU load和PyTorch-vs-ORT parity通过，最大绝对误差`8.940696716308594e-8`。
  正式contract固定`training.export_onnx=true`和same-iteration PT+ONNX原子发布；验收时尚未到
  iteration1000，不能把canary pair称为正式checkpoint。
- Rule-90 canonical `box_10` reference replay为H.264 1280x720、50 FPS、368帧、7.36秒，
  人工确认完整approach/pickup/carry/drop且物体正确；视频SHA256=
  `fdb3b046585ed2b59209af571a011d096f4a80c01207ad2b99ee2339c11a9e0b`。fresh API复核它已
  作为唯一history-backed `vis/replay`写在step17，prebind副本已删除。
- bounded acceptance于W&B step94通过：已观察95条numeric history，total FPS=`34859`、
  collection=`3.761 s`、learning=`0.403 s`、distributed weight sum=`4`，所有
  distill/DAgger/BC loss为0；四个worker/GPU均存活，无exit marker及fatal
  Traceback/OOM/NCCL/Gloo/segfault。完整审计root为
  `/home/ubuntu/FAR/holosoma_runs/formal_teacher_tracking_corl79_debug30_d435urdf_ws4_e1512_20260805_055336/`，
  immutable run contract和formal acceptance SHA256分别为
  `6f772d45af9a4c1b67e11b550db4320273ea8557bff690ad4d3328d1a845b813`和
  `94ef662724993a62854ab08d454b890fe1f3ce044c39ffed2bc0b42a3aed9a75`；验收后未安装recurring
  monitor，也未热改live immutable contract。

## 2026-08-08 seed17x 32xL40S distill student formal run handoff

当前正式训练已启动并通过 transactional startup-health gate：

- W&B：`zihanw22/carry-any/jbsmj8lx`
- URL：<https://wandb.ai/zihanw22/carry-any/runs/jbsmj8lx>
- run name：`distill_depth_seed17x128_ws32_e1024_onnx_20260808_2120`
- tmux session（四节点同名）：`distill_seed17x_ws32_onnx_20260808_2120`
- 节点：`10.99.0.39`、`10.99.0.183`、`10.99.0.54`、`10.99.0.180`，每节点 8x L40S，world size 32
- node-0 run dir：`/home/ubuntu/FAR/holosoma_runs/formal_distill_seed17x_ws32_20260808/training_logs_formal/carry-any/20260808_212302-distill_depth_seed17x128_ws32_e1024_onnx-locomotion`
- source snapshot：`src-50933707876363c2e2c2877050fe5674b29b5100ed7d72c8a621f35ebe4a765b`
- Rule-90 manifest SHA256：`8af0eaae9cdc371362e837db578bfedd445ae93c8bfc9b360f0b48905ab349f5`
- Rule-90 video SHA256：`219ade9bff51298d8f445d78731e223222f357b4f0ac37c9545c939579ecde07`

Formal immutable contract：128 条 seed17x motion，32 rank 各 4 条；1024 env/GPU，32768 total env；40,000 iterations；save interval 1000；`training.export_onnx=true`；每个保存边界必须 atomic same-iteration PT+ONNX；teacher `model_67000.pt` SHA256 `1c441a7eea24fb28d67cc4b5edeb123b91d589ec095d383034f32210a87b6c5b`；numeric training seed 42。

启动验收：四节点均为 `batch_preflight=1/1`、`torchrun_boundary=1/1`、`distributed_provenance=8/8`、`final_workers=8/8`，并稳定 10 秒；每节点 tmux 存活且 `nvidia-smi` 恰好 8 个 compute apps。首次验收时训练已越过 iteration 2，W&B state 为 `running`，已有 38 个有限训练指标（例如 BC loss 0.43536795397812966、mean reward 0.7665311977594673）。

本次正式启动前的 exact 4x8 / 1024-env/GPU / two-update canary：

- run dir：`/home/ubuntu/FAR/holosoma_runs/formal_distill_seed17x_ws32_20260808/canary/training_logs/carry-any/20260808_210409-distill_depth_seed17x128_ws32_e1024_onnx_canary-locomotion`
- PT SHA256：`e6435816418293edbc4d50fc1d2364abe36131bdcf6c5c9af847e6d12375a500`
- ONNX SHA256：`76e1eb734f95f7f46dd1822645b3ef00d9fb54ad46189716d86bf751a5e3857d`
- `onnx.checker`、ORT load、PyTorch-vs-ORT parity 均通过；max abs `1.1920928955078125e-06`，max rel `1.4231542991183233e-05`
- 四节点 canary 均 rc=0，OOM/PhysX-capacity/traceback count 均为 0

必须保留的 runtime memory contract：

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- `HOLOSOMA_RANK_VISIBLE_DEVICES=1`，正式入口必须显示 `train_agent_rank_visible.py`；每个 worker 只创建一条 GPU compute context
- PhysX：found/lost pairs `671088640`，found/lost aggregate `301989888`，total aggregate `201326592`，collision stack `268435456`

故障记录：旧 run `md3vhsbn` 和 `2zog0gf2` 均已标记 failed。前者暴露 allocator fragmentation；后者证明 formal 的 `HOLOSOMA_RANK_VISIBLE_DEVICES=0` 会让每节点其余 7 个 rank 在 GPU 0 各占约 466.89 MiB context，导致四台 GPU 0 在首个 rollout OOM。禁止用这两个 run resume，也禁止把 rank-visible 改回 0。更早的 `ifizc01l` 是 PhysX-capacity preflight 失败身份，同样不得 resume。

## CORL79+debug30 47.6-degree depth distillation formal run（2026-08-10 09:15 UTC）

- 新建正式 W&B run `zihanw22/carry-any/oosdgi7q`：
  <https://wandb.ai/zihanw22/carry-any/runs/oosdgi7q>，名称为
  `distill_tf109_gu5t40k_d43547p6_ws32_e1024_onnx_20260810`。这是 fresh depth student
  distillation，不是 teacher tracking；没有 resume 或 policy init。四节点为
  `zzzihanw-15/25/45/46`（`10.99.0.141/117/183/54`），每节点 8 张 L40S、每 rank 1024 env，
  world size 32、global env 32768；四节点 tmux 均为
  `distill_tf109_gu5t40k_ws32_20260810`。
- exact bank 是 CORL 实际 79 条加 debug30 的 turn-then-forward 109 条：box25、ball9、
  barrel36、bin39。single-slot view digest=`c3110b5d871bce803ba5cfe2fbb2849615d8aa1b6122551512069c2a1107d402`，
  ws32 rank source digest=`5661346f9fa045ded05dfea8d81ac84359b1ed84ea3342fa81517f8b8bbcfe09`，
  rank manifest SHA256=`5974ca388f06d5e8e95aaab5e7e3462b77e4d50502df8158fc627218fa0ae48c`。
  global clips exact-once，无 object/DataLoader 二次 shard；每 env 一个 object slot。visual/depth
  使用 original real mesh，collision 使用同一 real mesh 的 `convex_decomposition`，geometry
  缺失时 fail closed。
- label teacher 是 privileged 178D `gu5d3qo8/model_40000.pt`，PT/ONNX/pair SHA256 分别为
  `ec79589e4a04248b972ff011ce4ff396c2e341e377b0970f239f449e8e216a95`、
  `318c8dc0663bff6d963612635dd86c07bfa5a8ad4d1da0b54b8f825522332d6b`、
  `55ba36f81f9eaf4a37b108e5affca89db576fd487799bdc0f1e8ebacef137723`。历史 motion generator
  仍是不同的 `u8udzw0u/model_05000.pt`（SHA256=`80cb13e13590239d015ba0a29bdbae901b7785a9789d144745c7bd330059cd68`），
  不能把两者混称为同一个 teacher。
- student actor 输入为 sparse root command、drop button、无 linvel proprio/action history 和
  real-mesh depth，MLP 为 `[2048,1024,512,256,128]`。command 是预计算且可部署的
  `precomputed_turn_then_forward`：dy 恒零、dx 与 dyaw 不重叠、runtime pickup latch，drop
  阶段前三维清零；reference tracking reward 不变，offline contact guidance 权重为 0。
  PPO coefficient 从 0 起、每 700 iteration 增加 0.1、4900 时到 0.7，BC 为其余权重；
  DAgger replay 关闭，actor/critic adaptive LR 初始均为 `1e-3`，总计 40K iterations。
- depth 是 canonical torso D435：translation=`[0.0576235,0.01753,0.41987]` m、mount pitch=
  `47.6 deg`、mount quaternion=`[0,0.40354529635239006,0,0.9149596678498247]`、frame
  quaternion=`[-0.5,0.5,-0.5,0.5]`、source=`far_tracking_warp`、residual pitch=0；sensor
  noise、holes 和 reset pose randomization 保持启用。不能把 residual 0 误读成旧 27-degree camera。
- immutable source 为
  `src-6823c29b434fd32f5f1c75550af92762b93a396bccf99de94ea08a588f375fd4`，source archive
  SHA256=`d47a429772255355d394ac2a90bd0a3975f02c455d0ebf87abfc5c5806b50189`。
  exact 4x8、1024 env/rank 两步 canary 原生生成 PT+ONNX：SHA256 分别为
  `1e4e86ef56f3c1748cdb3271cfb99a7a761d30873758e3e283b9386ab7a4ce97` 和
  `92fedd7773d9994d3384bdc58cfa6dfb149134ddb2035c51a670a5e24304a57f`；ONNX checker、ORT
  load 和 PyTorch-vs-ORT parity 通过，最大绝对误差 `8.344650268554688e-7`。正式 run 固定
  `training.export_onnx=true`、每 1000 iteration 原子保存/上传同 iteration PT+ONNX；验收时
  尚未到 1000，不能把 canary pair 称为正式 checkpoint。
- 第一轮正式启动在任何训练 update 前因四节点缺少 content-addressed NCCL runtime 而 fail
  closed；随后只把 exact SHA256=`e4a7aee9c3eecf53fac780441d2f03b578ab8db8874b71f8e391bcec7adb2899`
  原子安装并逐节点复核，再用同一 immutable command 重启。训练实际采用 flat Gloo gradient
  all-reduce，不 preload NCCL；启动 token 为
  `9aa66429b70aa96335d74a7d250f50682fbbeeb9bcfb92ac2e5d86e125ba73a8`。
- Rule-90 canonical `box_10` reference replay 已人工确认完整 approach/pickup/carry/drop、真实
  mesh 且无 fallback；H.264 1280x720、50 FPS、368 帧、7.36 秒，视频 SHA256=
  `d13d233633e38bc26da39454501072cb6c40eafe7a251ede4eb668b15b6b4eba`。fresh API 复核它是
  W&B step38 唯一 history-backed `vis/replay`，prebind 副本已删除。首次写 history row 时
  一个非训练辅助字段误抄了 rank-manifest digest；不可变视频未受影响，正确 digest 已由签名
  preflight、run config、data closure 和 live summary 四处复核并作为 authoritative 值保留。
- bounded acceptance 于 W&B step88 通过：观察 89 条 numeric updates，最近 10 轮平均
  `90,854.5 FPS`，最新 collection/learning=`3.790/4.998 s`、BC/distill loss=`0.170385`、
  PPO coefficient=0、distributed weight sum=32。四节点各 8 worker/8 GPU application，fatal
  marker 和 uncorrected ECC 均为 0；绑定 replay 后训练继续推进，未热改 live contract。
  审计目录为
  `/home/ubuntu/FAR/holosoma/outputs/formal_distill_corl79_debug30_gu5t40k_ws32_e1024_20260810/`，
  run contract/acceptance SHA256 分别为
  `89110ff39a82c5d5d6eebc4dd27e0d2a554ebf4d86d604e3cb27cd02077e4c60` 和
  `77c328ccebaef15cd05ecf60cad111ea8bed2b914a5794adfeec018f60b3ae7d`。

## Rev-1.0 D435 两脚中心标定锁定（2026-08-10）

- 当前训练实际加载的 `g1/g1_29dof.urdf` 使用 Rev-1.0 torso mesh，且 `d435_joint` 明确为
  `torso_link -> d435_link` translation=`[0.0576235,0.01753,0.42987]` m、RPY=
  `[0,0.8307767239493009,0]` rad。workspace 的 `camera_depth_d435i` nominal 已与该 asset 对齐；
  旧版 deprecated URDF 的 `z=0.41987` 不再作为未来 run 的默认值。
- 标定闭环固定使用所有腿/腰关节为零的 URDF `q=0` neutral pose，以及
  `left_ankle_roll_link/right_ankle_roll_link` 两个原点的中点。以水平、pelvis-heading
  对齐坐标轴表达，camera center 相对该中点为
  `[0.05366232609678057,0.01753,1.230733752422211]` m；对应 ankle-center 间距为
  `0.23701291` m。回归测试同时从真实 URDF 做 FK，并核对 perception nominal 与
  `torso_link -> d435_link` 完全一致，避免再次出现 1 cm 漂移。
- 两脚中心只作为指定站立姿态下的标定和审计 frame；运行时 camera 仍刚性挂在
  `torso_link`。禁止在行走时把 camera 动态挂到两脚中点，否则会产生不符合真实机械结构的
  独立相机运动。camera reset pose DR、sensor noise、holes 和 latency 语义未改变。
- `vis_depth_replay.sh` 与 `sync_isaacsim_viser.sh` 的默认 D435 extrinsics 也已同步为上述
  Rev-1.0 URDF 值；后者仍允许显式环境变量覆盖以做有审计的 camera A/B，但默认 replay/Viser
  不再使用旧的 `[0.01,0.01,0.44]`、约 `27 deg` 通用相机位姿。
- 已在运行的 formal run `oosdgi7q` 的 immutable source 仍是旧 nominal
  `[0.0576235,0.01753,0.41987]` m；本次只修改 workspace 供未来新 identity 使用，没有热改、
  重启或停止该进程。它的历史 contract 必须保留原值，不能追溯性改写成 Rev-1.0 nominal。

## 0mc 40K MuJoCo 原未触发15条的2.5秒forward deadline重录（2026-08-10 22:27 UTC）

- 用户在+0.27 m补录后进一步明确“最晚2.5s后给forward”。本批从原+0.30 m full-109
  batch的`not_triggered`状态精确选出同一15条，继续使用exact `0mcqao8k/model_40000.onnx`
  （SHA256=`b2eb5206e255efb7a8974def2aa533f3ad493378affd351829521d11c38a4483`）和同一
  immutable bank/object map/physics。它是checkpoint actor policy rollout，不是reference replay，
  全程无W&B写入。
- evaluation-only command语义为：实际object world-z达到初始值`+0.27 m`时立即锁存；否则
  为保证actor不晚于2.500 s看到forward，observer在2.460 s启用time fallback。触发前root
  command为零，触发后直到step500逐帧严格为`[0.15,0,0]`，drop在501个actor step全零，
  checkpoint-native pickup cue保留，heading lock关闭。本批15条均由time fallback触发，逐条
  policy-I/O的首次forward均为2.460 s，15/15 deadline/command audit通过，不再有
  `not_triggered`。
- 终局结果为6条carry（`box_45/55/60/85/88`、`noscale__any_bin_80`）和9条loss
  （`box_33/62/70/74/89`、`noscale__any_bin_36`、`scale__any_barrel_33/35/53`）。loss是
  真实物理结果，不是command deadline失败。
- 历史checkpoint仍通过隔离runtime overlay使用旧sensor offset
  `[0.0576235,0.01753,0.41987]`和authenticated perception contract SHA256=
  `17bc4990533e804baceca55ee73b17447454832538030378da8974de17ec0456`；没有修改当前workspace
  的Rev-1.0 0.42987 m配置。初始`box_74/box_88`尝试撞到dcgm-exporter固定端口9400，失败尝试
  已保留，两条随后在预检空闲端口成功重跑；final batch failure=0。
- 交付目录为
  `/home/ubuntu/FAR/_check_vis_mujoco/08-10-2222__0mcqao8k_model40000__original_nottriggered15__mujoco_gate027_or_deadline2p5_forward015__xy_trajectories/`；
  15个individual与一个15-panel master均为H.264、50fps、501帧/10.02秒，master为
  1600x840、SHA256=`d50f0efd501874505962dfff3225dcdb9b6bd61b865ee3835ca8052035b2cbb6`。
  ffprobe、完整decode、媒体SHA、四时刻全panel人工审核和96项定向测试全部通过。审计root为
  `/data/holosoma_eval_audits/0mcqao8k_model40000_mujoco_original_nottriggered15_gate027_or_deadline2p5_20260810_220552/`；
  完成后8张GPU均为0 MiB/0%，无残留录制进程。

## exact109 pure-RL camera A/B 正式训练（2026-08-11）

- 用户要求在同一109条 turn-then-forward bank上做匹配的 camera A/B：base 放在
  `zzzihanw-e`（`10.99.1.21`），CORL 放在 `zzzihanw-102`（`10.99.1.134`），目标均为
  60,000 updates。4096 env/rank 真实 probe 在两节点都超过安全的 PhysX/GPU memory envelope，
  因此按用户原先授权 fallback 到 2048 env/rank；没有把4096或其他更高容量静默带进正式run。
  用户随后允许少量 PhysX capacity warning，但正式2048 canary和bounded acceptance实际均为
  capacity warning=0。
- base 正式 run 为 `zihanw22/carry-any/z9e7vxcv`：
  <https://wandb.ai/zihanw22/carry-any/runs/z9e7vxcv>，session=`formal_base_z9e7vxcv`，
  node=`10.99.1.21`。CORL 正式 run 为 `zihanw22/carry-any/2xmp4whp`：
  <https://wandb.ai/zihanw22/carry-any/runs/2xmp4whp>，session=`formal_corl_2xmp4whp`，
  node=`10.99.1.134`。每条均为1 node x 8 GPU、world size=8、2048 env/rank、全局16384 env；
  fresh actor/critic/optimizers/seed42，无 training resume 或 policy init。`WANDB_RESUME=must`
  只连接启动前 Rule-90 预绑定的 fresh identity。
- 两臂唯一预期训练差异是相机 nominal extrinsics。base 使用 Rev-1.0 torso D435：mount down=
  `47.6 deg`、residual pitch=`0`、offset=`[0.0576235,0.01753,0.42987]`、mount quaternion=
  `[0,0.40354529635239006,0,0.9149596678498249]`。CORL 使用历史 SW 配置：约27-degree
  mount、residual pitch=`10 deg`、effective down约`37 deg`、offset=`[0.01,0.01,0.44]`、
  mount quaternion=`[0.00644801,0.23350163,0.00644801,0.97231365]`。两边相机 translation DR
  都是XYZ各`+-0.035 m`，rotation DR都是roll/pitch/yaw各`+-3.5 deg`，58x87 depth CNN输出32D，
  sensor noise/edge/holes/3--4 frame latency完全一致。
- shared pure-PPO为24 steps/env、7 epochs、4 minibatches、actor/critic MLP均
  `[512,256,128]`、adaptive LR初始`1e-3`和bounds `[1e-5,1e-2]`、desired KL=`0.01`。
  entropy在0--2000为`0.005`，随后线性降到iteration10000的0并保持到60K。actor scalar输入94D
  加32D depth；distill/teacher/BC/DAgger均关闭，W&B实测`ppo_coeff=1`、`bc_loss=0`。
- shared DR为：每关节zero bias `U[-0.01,0.01] rad`、Kp/Kd scale各`U[0.9,1.1]`；torque
  RFI关闭、action delay固定0；torso COM x/y/z=`+-[0.055,0.08,0.1] m`；push interval=
  `[0.5,2.0] s`、max velocity=`[0.7,0.7,0.25,0.7,0.7,1.0]`；object mass与full inertia
  coupled log-uniform scale=`[0.25,4]`，object static friction=`[0.1,0.7]`、dynamic/static ratio=
  `[0.7,0.99]`、restitution=`[0,1]`。termination为`BadTrackingZOnly`加episodic motion-end：
  root z阈值1.0、ankle/wrist z阈值0.5、object full-XYZ位置阈值0.5、robot/object orientation
  阈值均1.1。
- exact bank为CORL实际79条加debug30的109条，category=`box25/ball9/barrel36/bin39`；
  single-slot source/view digest=`0d1ae14d...e6c5/307e9662...bddef`，ws8 shard source digest=
  `cdb52e87be4582a03535f30e5c820221bca66a1a3b9444e133851e96a937080c`，rank manifest SHA256=
  `092b6f89a3cf185d05ea9c22a1bec14614203be9c601d2299cc6e31a66b222a8`。rank clip count=
  `[32,16,16,16,16,8,4,1]`，109 clips global exact-once且每个local clip数整除2048。command固定
  `precomputed_turn_then_forward`，drop active时前三维清零；t0概率固定0.2，T1 boost关闭。
- PhysX显式使用从 `0mcqao8k` fresh config读回并在两节点真实复验的值：found/lost pairs=
  `335544320`、aggregate=`469762048`、total aggregate=`83886080`、collision stack=
  `268435456`。先前误用的671m/302m/201m与临时939m aggregate都没有进入正式run。
- immutable source=`src-2dfc78ee41466fa95bfd587a20ddcb69c3b2d9552c85d1cc26833ce68f521a3c`，
  archive SHA256=`7c83c289093b2356eeffd0fd220f0dc0d26fa518e05c3fab0ec4cad854ed66c3`，
  Python runtime manifest SHA256=`dd7ca81fa848917c362b3a239893a7a26f4c89d42b4f85cb515d91622f1690bc`，
  formal worker SHA256=`c6778c3a04b62ddb26d8824871c102f400458bfb5cde56fc1eae4da8e6985e9c`。
  base/CORL run-contract SHA256分别为`5495792994181ddaf0ca1fd7dd2209bcda20a8dd877941d407f7915f6416b94a`、
  `c02beb892b9c3eb999be8cb8c6a4a238d75dd19857ebc99e73dc99e29e3579c1`。
- exact 2048 canary两边均完成2个PPO updates并原生生成同iteration PT+ONNX pair；ONNX checker、
  ORT load与PyTorch-vs-ORT parity均通过。base/CORL pair SHA256分别为
  `a87eb62b27bdf137d0ae821d21b73e7aaab11be0e5873d702144785c1742a03f`、
  `8fe39c26e6c5787bf5c044888e4192ac379f5340cc3758ee7d1588be7e402f32`，最大绝对误差分别为
  `8.195638656616211e-8/8.940696716308594e-8`。正式run固定`training.export_onnx=true`、每1000
  update原子保存/上传同iteration PT+ONNX并在失败时fail closed；bounded acceptance尚未到1000，
  禁止把canary pair称作正式checkpoint。
- 两条Rule-90都从各自最终输入重新录制canonical第一条`box_10`，H.264 1280x720@50fps、
  368帧/7.36秒。base/CORL视频SHA256分别为
  `8c84af2480a6ccaf82ffbb25367bb40412681ee8eb17c47221b71815494142da`、
  `3ccdd6f0cd597b662b908894872da7835ec3e03764f013caf7fe7738a07a1721`；contact sheet人工确认
  approach/pickup/carry/drop/return连续、物体正确且无default-pose替换。fresh API证明base step38、
  CORL step36各自恰好一行history-backed `vis/replay`和一个MP4；summary-only预绑定副本已删除。
- 2026-08-11 02:01 UTC bounded acceptance：base/CORL remote progress=`123/119`，W&B step=
  `115/111`且state均为`running`；每节点8 compute apps/8 unique GPU UUID、exact tmux alive、无exit
  sidecar。两边日志的CUDA OOM/PhysicsScene failure/capacity warning/Traceback/nonfinite均为0。
  rank-visible headless启动期保留base/CORL `79/77`条gpu.foundation/renderer错误，但首个
  `HOLOSOMA_PROGRESS`后新增为0，不得隐瞒为零启动错误或误报为训练crash。W&B最新base/CORL
  reward=`2.14584/2.12304`、KL=`0.01477/0.01319`、FPS=`51157/47071`，所有抽样numeric finite；
  replay promotion后两边继续推进。审计root为
  `/home/ubuntu/FAR/holosoma/outputs/formal_camera_ab109_60k_20260811_000020/`，bounded acceptance
  SHA256=`011a7d465c03eb8ecad1cfb9caa9f558fb0314ccac04c36704123238c5b6d746`。

## exact109 Critic-381 pure-RL camera A/B、4x8 L40S/arm 正式训练（2026-08-12）

- 用户要求以2026-08-11 camera A/B pure-RL contract为基线重新做匹配实验；唯一科学改动是
  Critic从377D在末尾追加Actor当前policy step实际收到的3D root command和1D drop button，成为
  381D。Actor保持94D scalar加32D depth embedding即126D，输入仍为
  `actor_obs_root_contact_aware`、`actor_obs_drop_button`、
  `actor_obs_proprio_with_actions_no_linvel`；Critic exact input order为`critic_obs`、
  `critic_proprio_history`、`critic_actions`、`actor_obs_root_contact_aware`、
  `actor_obs_drop_button`。没有让Critic重算command，没有加入depth/phase/pickup latch，没有删除
  history-length=1的64D proprio，也没有改camera、DR、termination、PPO或60K target。
- Base使用ap-northeast-2a的`zzzihanw-17/39/42/47`
  （`10.99.0.97/39/176/24`）；CORL使用ap-northeast-2b的
  `z1hanw, zzzihanw-72/79/z`（`10.99.1.60/89/154/69`）。每臂4 node x 8 L40S、
  world size32、2048 env/rank、global65536。Base tmux为
  `holosoma_formal_base_uo0nx5q6`，CORL tmux为`holosoma_formal_corl_549ynm08`。
- Base fresh W&B run为`zihanw22/carry-any/uo0nx5q6`：
  <https://wandb.ai/zihanw22/carry-any/runs/uo0nx5q6>；CORL为
  `zihanw22/carry-any/549ynm08`：
  <https://wandb.ai/zihanw22/carry-any/runs/549ynm08>。两者均是fresh pure PPO、seed42，
  无training resume、policy init、teacher、BC或DAgger；`resume=must`仅用于连接启动前已做
  Rule-90预绑定的fresh identity。PPO仍为24 steps、7 epochs、4 minibatches、actor/critic
  `[512,256,128]`、adaptive actor/critic LR初始`1e-3`、entropy 0--2000为0.005并在10000
  线性降到0，总目标60,000 iteration、save interval1000。
- 相机A/B仍保持旧contract：Base是torso-mounted Rev-1.0 D435，offset=
  `[0.0576235,0.01753,0.42987]`、mount down=`47.6 deg`、residual pitch=0；CORL是历史SW
  setup，offset=`[0.01,0.01,0.44]`、mount约27 deg、residual pitch=10 deg、effective down约
  37 deg。两边camera XYZ translation DR均为`+-0.035 m`，roll/pitch/yaw均为`+-3.5 deg`，
  depth noise/holes/latency及其余既有DR、z-only termination完全未改。
- exact 109 bank、command与sampling沿用匹配基线：single-slot view digest=
  `307e9662d498bd507b9d17ca9abf74a3654f7bf66ac6ab989c6f19c3889bddef`，ws32 shard source
  digest=`13db668f710806bf4bc6b0541f1c99a3e2b36ad3e5179cccd31d3af8f1ab4928`，rank manifest
  SHA256=`19500fe84e4fef7c70cadc574b2581a41f1e85b5bf5e6cede7fa58a57ab8c858`；
  `precomputed_turn_then_forward`、drop时root command清零、t0=0.2、T1 boost关闭。
- immutable source为
  `src-d6abe8cd6a518af3754a8f543054e8c61d954e3be44cccbc17f3a6592a0c3ea4`，archive
  SHA256=`f719bd0bc53674ecf6415358863f1661a2674411f20fd5c6cf41e257a79c48ac`，worker SHA256=
  `3fed7d0cd8df9f37976804d99911ef240d66e22f4fcdbe1c2397e679676fc908`，runner SHA256=
  `f25cfec90210e57df364bde71a00d4bf2d79d8ac87d5a987c2114c07bf42f42e`。Base/CORL immutable
  contract SHA256分别为`db137bfb7f353c92e63c3e6022a9a4028fd998b605f462f0db4380892630c651`和
  `849ecb5508eec5704e8f51dd4d1699125644ade412f5273dce8903f8bd57b8e5`。
- 两边都通过exact 4x8、2048 env/rank、2-update真实canary并原生生成same-iteration PT+ONNX。
  Base PT/ONNX/pair-manifest SHA256分别为
  `6c45411db34f17676e0cc313e958fec4a4ab783ac9ac248d6b03e25a16e45b6c`、
  `ab51725e08bdc9ced2fbc35a16751e020bfb3a5af8ff20c3fbc516e2d87b6277`、
  `37763caef2144c47cf96de9c9af915822e4c549c7f6c5d006d6a215c40b41459`，最大绝对parity误差
  `9.313225746154785e-8`；CORL对应为
  `33f27338127fce3af3586c44b66f9e44cf62be83a9b7ba5bb6e06bd7c0250776`、
  `ee54d7d8c58c9cd26f5cc3a6142c26f4613e93442b6c49c2e2d4a0985ecb4543`、
  `177913c7f1add59b36f296637db46bb92cc53c5f1a3c1527b4db606dcf11d2e6`，最大绝对误差
  `9.685754776000977e-8`。ONNX checker、ORT load和PyTorch-vs-ORT parity均通过；formal
  contract固定`training.export_onnx=true`及每1000 iteration PT+ONNX原子配对，验收时尚未到
  1000，不得把canary pair称为正式checkpoint。
- Rule-90 Base/CORL canonical `box_10`均为H.264 1280x720、50fps、368帧、7.36s，并已人工
  抽帧确认approach/pickup/carry/drop/return连续且物体正确。视频SHA256分别为
  `822d32a66eabf1574a95e0e83cf32f1b46d052b0627993774407b733342421a4`和
  `8bc5dcddd220e78a5ebde18dfde9abb73b4a88b30a87d984211cfaea47457c72`；fresh API最终复核
  Base step15、CORL step11各恰好一行history-backed `vis/replay`和一个MP4，prebind副本已删除。
- 2026-08-12 06:19 UTC bounded acceptance：Base四节点completed-iteration floor=51，CORL=43；
  W&B snapshot step=39/33且state均为running。8个节点各自tmux存活、8 compute apps、8 unique
  GPU UUID，exit sidecar为空，fatal/OOM/NCCL/non-finite/PhysX-capacity warning和uncorrected ECC
  均为0；W&B抽样numeric全部finite，PPO coeff=1、BC loss=0。Base/CORL latest W&B reward=
  `0.48691/0.20152`、KL=`0.01277/0.01370`、FPS=`180788/161760`。
- 运维异常没有静默隐藏：Base master `zzzihanw-17`的root filesystem在launch前已满，首次formal
  preflight因W&B写`~/.config/wandb`失败；把W&B cache/data/artifact/config/run全部定向到该run
  scoped `/data`后4/4重验通过。随后Base Rule-90 bind helper还因Python `tempfile`探测`/tmp`失败
  一次，补同一scoped `/data/.../tmp`后成功；两次都发生在辅助/preflight路径，不是训练failure，
  没有改科学配置或停止live PPO。脚本最初一次`apply_patch`失败则是从旧run复制的launcher为
  0555只读；恢复owner write后同一patch成功，不是patch冲突。
- 审计root为
  `/home/ubuntu/FAR/holosoma/outputs/formal_camera_ab109_critic381_ws32_60k_20260812_051847/`；
  bounded acceptance SHA256=
  `9f00d49a83b069c81e37916685e8338345f7e90b34db06efd514bf1613fc833a`。验收后没有热改live
  scientific contract，也没有安装recurring monitor。

## exact109 canonical object frame独立派生与Critic-317 preset（2026-08-12）

- 用户确认下一轮可以删除Critic里history-length=1的`critic_proprio_history`，但禁止热改现有数据和
  live 381D run。workspace新增独立experiment preset
  `g1_29dof_wbt_w_object_pure_rl_policy_command_after_lift_critic317`：Actor input仍严格为
  `actor_obs_root_contact_aware + actor_obs_drop_button + actor_obs_proprio_with_actions_no_linvel`
  （94D scalar，camera训练时另接32D depth embedding）；Critic input改为
  `critic_obs(284) + critic_actions(29) + actor同policy-step command(3) + drop(1) = 317D`。
  新observation preset中物理删除`critic_proprio_history` group；旧preset仍保留该group，已在跑的
  `uo0nx5q6/549ynm08`继续使用各自immutable 381D contract，没有停止、resume或热改。
- 新增builder `scripts/build_canonical_object_frame_bank.py`，从旧exact109只读派生完整的新object-frame
  bank。canonical contract=`canonical_object_frame_com_geometry_axes_v1`：origin为physical COM；+Z为
  几何/惯量主轴中最接近reference初始world-up的方向；+X为其正交平面内较长的主方向；
  `+Y=Z cross X`。轴符号以world-up和投影legacy +X作确定性tie-break；相对惯量gap不超过0.02的
  近轴对称/近球体显式记录axis ambiguity，不能把不可观测的yaw语义伪称唯一。
- 派生不仅改observation数据：81个OBJ的vertex/normal、109个URDF的link frame/inertial origin和完整
  inertia tensor、109个main motion的object pose/quaternion/linear+angular velocity/size/URDF path、
  109个teacher rollout reference的current/target object state，以及488个`*_contact_points.npy`
  全部按同一SE(3)变换同步修改。main motion quaternion仍为WXYZ，teacher sidecar仍为XYZW。
  `actor_obs_raw/actor_obs_norm`共218个旧frame opaque数组从新sidecar中删除，避免把无法可靠重建的
  legacy teacher observation冒充canonical；actions、robot/reference states、command/drop、contact
  intervals/counts和其他非object-frame字段保持原语义。
- 新immutable bank为
  `/data/holosoma_inputs/corl79_plus_debug30_decoupled_turn_forward_canonical_object_frame_v1/by-source/6ea4e78886a463b0dd59c1d69e6b15d13fb90fed88fbd397323a9eb441233616/`，
  大小3.6G、109 NPZ、109 URDF、81 OBJ、109 rollout reference、1995个manifest-bound payload files，
  无symlink且所有文件/目录只读。完整manifest为
  `_canonical_object_frame_manifest.json`，payload digest与目录名均为
  `6ea4e78886a463b0dd59c1d69e6b15d13fb90fed88fbd397323a9eb441233616`；1995/1995文件已fresh SHA256
  复核。mesh symmetry统计为45 asymmetric、35 near-axisymmetric、1 near-spherical。
- 旧source bank保持原路径和2.9G内容；构建前/后/发布后tree metadata fingerprint均为
  `cfb230f7a7766a4053092dcad939f8609b520240aab4fec71ea96c68093b6513`，证明没有原地修改。
  对每个clip的每一帧做surface geometry/velocity等价检查，并抽帧做world inertia等价；全bank最大
  绝对误差分别为`2.3994438524965744e-7 m`、`3.003313211458192e-7 m/s`和
  `1.488667350849071e-9 kg m^2`。URDF inertial origin 109/109为零，mesh引用109/109可解析。
- repo原生`validate_contact_sidecars.py`在正确的0.2s runtime prepend compensation contract下通过
  109/109，contact target coverage=109/109；`validate_runtime_contact_intervals.py`不适用于该源bank，
  因为旧/新motion都没有它专门要求的`hand_contact_valid`字段。相关新单元与回归测试共31项通过。
  新bank目前未分ws32 shards、未启动训练；未来若正式使用，必须以它为新immutable data identity
  重新做sharding、canary、Rule-90及原生same-iteration PT+ONNX contract，不能把live 381D run迁移进来。
- 为人工检查frame语义，另在
  `outputs/canonical_object_frame_axis_comparison_glbs_20260812_v2/`导出7个GLB：1个aligned box
  control、2个bin、2个barrel、2个near-spherical object。每个GLB在同一旧坐标mesh上叠加旧/新两套
  frame：旧轴为浅色短细RGB、新轴为饱和长粗RGB、旧原点为白球、新COM原点为金球、二者偏移为
  金线；X/Y/Z固定红/绿/蓝。7/7均经trimesh重新加载并验证所需node，代表barrel另经assimp解析通过。

## 549ynm08 三小时后迁移到 0mc camera + BadTracking（2026-08-13）

- 用户要求从消息时刻精确三小时后取消 `zihanw22/carry-any/549ynm08`，复用其
  `z1hanw, zzzihanw-72/79/z` 四节点共32张L40S启动fresh replacement；绝对触发时间为
  `2026-08-13T10:00:02Z`。调度session为
  `holosoma_schedule_549ynm08_to_0mc_20260813`，审计root为
  `outputs/formal_0mc_badtracking_tf109_critic381_ws32_e2048_60k_20260813_0700/`，冻结bundle
  manifest SHA256=`600942898dbf39e81d4a3309bfb4f7f0b7894f59d6e6295e9dd75a5eba814578`。
- 停机选择器只接受exact tmux=`holosoma_formal_corl_549ynm08`、run ID、source snapshot、worker、
  contract、master addr/port和每节点8个train rank全部匹配；2026-08-13 07:19 UTC四节点只读
  preflight通过且没有发送signal。禁止用该控制器停止 `xxr6at37` 或其他run。
- replacement以`549ynm08` Critic-381 contract为baseline，保留exact109、2048 env/rank、pure PPO、
  reward、DR、sampling、precomputed turn-then-forward command、0.005@0--2K后线性降至0@10K的
  entropy schedule、60K target和每1000 iteration原生PT+ONNX pair。唯一科学变化是：(1) exact
  `0mcqao8k` camera，torso offset=`[0.0576235,0.01753,0.41987]`、mount down=47.6 deg、residual
  pitch=0；(2) full XYZ `BadTracking`，五阈值=`[0.5,0.8,0.25,0.25,0.8]`并保留motion-end。
- immutable worker SHA256=`da39f6129f7d8b5bfcb0eb318c1722bc10a5dbd782a0c8660b0f2c8ca8160095`。
  canary验收会把完整`experiment_config`与`549ynm08`认证canary递归diff，只允许相机三项、
  termination函数/五阈值及run name/logger path共11个声明路径变化；其余任何漂移在创建W&B
  identity前fail closed。随后必须通过exact 4x8/e2048两步canary、ONNX checker、ORT load及
  PyTorch-vs-ORT parity，才动态分配fresh run ID并做Rule-90 prebind、formal preflight和launch。
- 触发后的fresh identity写入同一审计root的`wandb_identity.json`，最终32-GPU/W&B/Rule-90验收写入
  `formal_start_acceptance.json`；W&B URL由其中run ID确定。不存在这些文件时不得猜测run identity，
  也不得绕过canary手工启动。
## xxr6at37-derived 47.6-degree / BadTrackingZOnly-0.75x formal pair（2026-08-15）

- 用户要求使用8台机器启动两份fresh formal pure-RL：每臂4 nodes x 8 L40S、world size32、2048 env/rank、global65536、target60K。A为`xxr6at37`科学设置改成Rev-1.0物理47.6度Base相机；B除同一相机变化外，只把`BadTrackingZOnly`五个阈值整体乘0.75。两条都不是full-XYZ `BadTracking`。
- A为W&B `zihanw22/carry-any/tuhu3ghf`，节点`10.99.0.97/117/141/186`，tmux=`holosoma_formal_xxr47_tuhu3ghf`；B为`zihanw22/carry-any/11xg5p3k`，节点`10.99.0.24/39/54/183`，tmux=`holosoma_formal_bt075_11xg5p3k`。
- shared contract：immutable source=`src-d6abe8cd6a518af3754a8f543054e8c61d954e3be44cccbc17f3a6592a0c3ea4`；exact109/ws32 shard；Critic377；`precomputed_turn_then_forward`；pure PPO；entropy恒为0.005；相机offset=`[0.0576235,0.01753,0.42987]`、mount down=47.6度、residual pitch=0；每1000 update原子发布同iteration PT+ONNX，`training.export_onnx=true`。
- A的termination阈值`bad_ref_pos/ref_ori/motion_body_pos/object_pos/object_ori=[1.0,1.1,0.5,0.5,1.1]`；B精确为`[0.75,0.825,0.375,0.375,0.825]`。fresh W&B config已读回两边func均为`BadTrackingZOnly`。
- 两边4x8/e2048两步canary均自然退出并通过ONNX checker、ORT load与PyTorch-vs-ORT parity；A/B max-abs分别约`1.12e-7/9.69e-8`。两条Rule-90均本次重新录制`box_10`，H.264 1280x720@50fps、368帧/7.36秒，人工contact-sheet审核通过；每个W&B run最终只有一条history-backed `vis/replay`和一个对应MP4。
- 2026-08-15 06:53 UTC bounded acceptance：A/B remote completed iteration=`25/44`，W&B summary step=`21/41`且state均running；8/8 tmux、64/64 GPU apps、exit sidecar/fatal scan/volatile UECC均0，W&B读回`ppo_coeff=1`、`bc_loss=0`、entropy coefficient约0.005。审计root=`outputs/formal_xxr47p6_badtracking075_ws32_e2048_60k_20260815_060715/`。
- 首次A canary曾尝试节点`10.99.1.122`，因该节点source snapshot缺runtime `data` symlink，在simulator/checkpoint/W&B之前fail closed；旧尝试已停止且未创建正式identity，正式A改用已验证的`10.99.0.186`。不得把该pre-W&B尝试当训练run或checkpoint。

## exact109原始motion + Rev-1.0 URDF + broad physics privileged tracking teacher（2026-08-15）

- 用户要求用当前distill所依赖的109条原始motion、更新后的object/robot URDF和更宽DR，在一台8卡机器训练一份新的tracking teacher。fresh formal W&B run为`zihanw22/carry-any/4cnc2pjd`：
  <https://wandb.ai/zihanw22/carry-any/runs/4cnc2pjd>；节点=`ip-10-99-0-176`（`10.99.0.176`），8xL40S，tmux=`holosoma_teacher109_broad_4cnc2pjd`。目标40,000 updates、2048 env/rank、world size8、global16384、seed42；无resume、policy init、外部teacher、distill、BC、DAgger或offline contact-guidance reward。
- exact bank为
  `/data/holosoma_inputs/corl79_plus_debug30_decoupled_turn_forward_v1/by-source/307e9662d498bd507b9d17ca9abf74a3654f7bf66ac6ab989c6f19c3889bddef`，109条=`box25/ball9/barrel36/bin39`，原始motion numeric payload保持109/109；single-slot manifest SHA256=`2de9ee5ca188b70e877c32dd9f0d2975eea99d11aa077bb077cf06ea9ab897bb`，object map SHA256=`70b466aad04837a79f6dd0f4491cb345a73c687209981acd3eb7f4a0365d8f5c`。ws8 rank clip count=`[32,16,16,16,16,8,4,1]`、global exact-once且都整除2048，rank manifest SHA256=`092b6f89a3cf185d05ea9c22a1bec14614203be9c601d2299cc6e31a66b222a8`。
- 使用更新后的real-mesh category-mass object URDF：visual/depth为original mesh、collision为convex decomposition，COM与full inertia由mesh派生；不允许fallback geometry。机器人为G1 Rev-1.0 URDF，D435 nominal offset=`[0.0576235,0.01753,0.42987]`、物理俯角47.6度。相机不进入teacher actor/critic，但训练时仍真实生成并施加XYZ各`+-0.035 m`、每轴`+-3.5 deg`及noise/edge holes，以覆盖后续student观测域。
- policy是fresh privileged pure-PPO tracking teacher：actor=178D、critic=310D、action=29，MLP均`[512,256,128]`；24 steps/env、7 epochs、4 minibatches、adaptive actor/critic LR初始`1e-3`、desired KL=`0.01`、entropy=`0.005`。command为immutable `precomputed_turn_then_forward`，runtime pickup latch保留，drop active时root command清零；这不是distillation run。
- broad DR：motion-relative reset为dof pos/vel=`0.20 rad/0.35 rad/s`、root pos=`[0.08,0.08,0.025] m`、rot=`[0.15,0.15,0.30] rad`、linvel=`[0.20,0.20,0.10] m/s`、angvel=`[0.25,0.25,0.35] rad/s`、object xy=`+-0.08 m`；push interval=`[0.4,2.2] s`、max 6D velocity=`[0.8,0.8,0.3,0.8,0.8,1.1]`；robot static/dynamic friction=`[0.25,1.7]/[0.25,1.3]`、restitution=`[0,0.6]`、base COM x/y/z=`+-[0.065,0.095,0.12] m`、link mass scale=`[0.85,1.25]`、base added mass=`[-1.5,3.5] kg`；object static friction=`[0.08,0.8]`、dynamic/static=`[0.65,0.99]`、restitution=`[0,1]`、mass和完整inertia coupled log-uniform scale=`[0.20,5.0]`。action delay、torque RFI、PD gain和joint calibration bias保持关闭。
- preflight发现旧teacher object mass/full-inertia仍为`[0.30,3.25]`，未覆盖student的`[0.25,4.0]`；正式identity创建前已改成上述`[0.20,5.0]`并通过20项randomization/command测试。immutable source=`src-99c7b2632feb6edfe7c684a3b26bb9e612cf51c518a55ea2173b9608241d6656`，archive SHA256=`92f39dc60a152d66b74e04b8bc01eef6f25912f64ab51768c917207a72aba64d`，run contract SHA256=`f4edc11fc438c9ac25e928899cdab0834adee574f4e09e762e44d25711434774`。
- exact 1x8/e2048 canary完成2 updates，并原生生成same-iteration `model_00002.pt/.onnx`；PT/ONNX SHA256=`8e32b5713c083c98247694c3a8774dbf3b45dc01573024d0ec45f29bbdeb3d2e/12f9c2577b53319a166ec564d60aae6fc180ee5c4e540112d821a653fbcdc5a7`，ONNX checker、ORT CPU load、PyTorch-vs-ORT parity通过，max abs error=`8.195638656616211e-8`。formal固定`training.export_onnx=true`，每1000 update必须原子保存并上传同iteration PT+ONNX；任一导出/校验/发布失败则checkpoint boundary fail closed。
- Rule-90重新从exact输入录制canonical `box_10`，H.264 1280x720@50fps、368帧/7.36秒，视频SHA256=`eae57d0456f0e5b7bb863abef6eb4e6f1bbacd6da2ace3ca896d7d351aace051`；人工contact sheet确认approach/pickup/carry/drop/return连续且物体正确。最终fresh API复核为history step22恰好一行`vis/replay`、一个MP4，summary指向该history-backed文件；预绑定副本已删除。
- 首次formal wrapper在占GPU前因自定义prebind遗漏完整flattened metadata而被Rule-90 verifier fail closed，8卡训练没有启动；补齐同一immutable manifest对应的config/summary metadata后，attempt-2通过并启动。首次失败状态与日志保留，未改科学contract、run identity或checkpoint。2026-08-15 19:14 UTC bounded acceptance：local completed iteration>=48、W&B step>=56且state=running；8 apps/8 unique GPU，首个PPO iteration以后Traceback/RuntimeError/OOM/child failure/non-finite/PhysX capacity均0；W&B numeric finite、`ppo_coeff=1`、`bc_loss=distill_loss=0`。审计root=`outputs/formal_teacher109_rev1_broadphys_ws8_e2048_40k_20260815_183657/`，`formal_start_acceptance.json` SHA256=`5e7877e517db8d02ea300c29b7138ebbdcb77400287eea9a71513fc7aadb3cf4`。

## xxr6at37 model_33000 diverse12 pure-forward rollout（2026-08-16）

- 用户要求评估`zihanw22/carry-any/xxr6at37`最新checkpoint；请求时W&B summary step=`33887`，冻结的最新完整atomic PT+ONNX pair为`model_33000`（completed iteration32999/next33000），不得追逐录制期间出现的后续checkpoint。PT/ONNX SHA256分别为`d0606c0f8312c086916dcab8e314c07e01273d3de2d48a1e7f1cdd84d1d46089`和`dd6ea44de5a73c80d2a4b320a8e3f432fbc1734e8cbf1abaf47432aa5fd8e2b0`；pair中ONNX checker、ORT load及PyTorch-vs-ORT parity均通过。
- 使用与13K对比一致的diverse12（box/ball/barrel/bin各3条）、exact109 motion view digest=`307e9662d498bd507b9d17ca9abf74a3654f7bf66ac6ab989c6f19c3889bddef`、CORL相机effective down=37度、single env、seed42、timestep0、randomization disabled、initial-pose noise0。逐帧actor contract为gate前`[0,0,0,0]`，object相对初始z首次达到0.30m后持续robot-heading-frame`[0.15,0,0,0]`，consecutive=0、保留native pickup、drop全程0、无heading lock；501/501帧policy-I/O均通过审计。
- 9/12触发forward；`box_62`（max dz=0.270m）、`box_75`（0.269m）、`noscale__any_barrel_12`（0.076m）未达到0.30m，按`not_triggered`原样保留，未提前发command或降低阈值。`scale__any_barrel_53`到actor step497才触发，也如实保留。
- 本地交付=`/home/ubuntu/FAR/_check_vis/08-16-1914__xxr6at37_model33000__diverse12__post_lift_forward015__xy_trajectories/`：12 raw、12单条trajectory、4个category三联及1个4x3 master，共29个H.264 MP4，均为501帧@50fps并完成ffprobe和全解码；master SHA256=`d810d482caa2d108d87d22d91eb0fc2c0d661d27565a1b594133911f23e0a035`。审计root=`outputs/eval_xxr6at37_model33000_diverse12_forward015_20260816_191404/`。
- eval节点`10.99.1.122`的Isaac Sim在视频/metrics/policy-I/O全部落盘后卡在`simulation_app.close()`；每批仅在逐条验证501+501 rows及唯一501帧MP4后，才终止本次campaign的精确PID并释放GPU。未停止或修改任何training process，未向W&B写入。

## xxr6at37 model_21000 vs model_28000 diverse12 pure-forward（2026-08-16）

- 使用与13K/33K相同的diverse12、exact109 view、CORL effective37相机、seed42、timestep0、randomization disabled及lift-gated pure-forward协议，做同一run内21K/28K的干净checkpoint对比。两个pair的experiment/source/observation/command contract hash完全一致；21K PT/ONNX SHA256=`ba2ed8fc45bff471d3cb1d9ea5088d136290dcf1d3029788f07b613b428aea9c/fda7527e5f6f86961e67b3a2386ecd19accb73401e104cfa0f0259d4cdf68391`，28K=`1a2aeddc82fad8a41a71dd32d3c58c6e7a46dc6fd1df4384553e7ca36008dd29/2daba8afe9a76fc9cf22a6f2fd15608388247615620726bc136adb48f1bc86fd`；两者atomic pair及ONNX parity均通过。
- 逐帧actor input验收仍为gate前`[0,0,0,0]`，object dz达到0.30m后持续`[0.15,0,0,0]`，drop全0、consecutive=0、无heading lock。21K触发7/12；未触发为`box_62`、`box_75`、`noscale__any_barrel_12`、`scale__any_barrel_53`、`scale__any_barrel_80`。28K触发10/12；未触发仅`box_62`和`scale__any_barrel_53`，但`box_75/noscale__any_barrel_12/scale__any_barrel_80`分别较晚到actor step205/376/279才触发。
- 本地交付=`/home/ubuntu/FAR/_check_vis/08-16-1942__xxr6at37_model21000_vs_model28000__diverse12__post_lift_forward015__xy_trajectories/`：24 raw、24单条trajectory、12逐物体左右对照、4 category对照及1个4x6 master，共65个H.264 MP4，均501帧@50fps并完成ffprobe、全解码和contact-sheet人工检查；master SHA256=`10dbd8b4596a07e88e6fecd8e14596210f854451904b3919f07420c81e777178`。审计root=`outputs/eval_xxr6at37_model21000_vs_28000_diverse12_forward015_20260816_194214/`。
- Isaac Sim同样在完成媒体后可能卡cleanup；本次每条先由独立validator核验501 metrics、501 policy-I/O、唯一501帧H.264及完整command/gate契约，再释放本campaign精确PID。24/24均生成`COMPLETED`，未触碰training、未写W&B。

## xxr6at37 21K/28K post-lift forward 0.07 vs 0.11（2026-08-16）

- 用户在21K/28K同run对比后明确要求再试forward 0.11和0.07。保持相同diverse12、exact109 view、CORL effective37相机、seed42、timestep0、randomization disabled、object dz gate=0.30m、consecutive=0、native pickup保留、drop全0、无heading lock；唯一变化为触发后逐帧actor input分别严格`[0.07,0,0,0]`或`[0.11,0,0,0]`。共`2 checkpoints x 2 commands x 12 objects=48`条，48/48完成501-frame metrics/policy-I/O及唯一501帧H.264验收。
- trigger count与0.15基线一致且不受post-gate command影响：21K在0.07/0.11下均7/12，28K均10/12。21K未触发仍为`box_62/box_75/noscale__any_barrel_12/scale__any_barrel_53/scale__any_barrel_80`；28K仍为`box_62/scale__any_barrel_53`。
- 在triggered样本上，0.07明显优于0.11：21K从注入到终点的平均G1位移`1.930m vs 0.426m`、平均terminal object dz`0.253m vs 0.050m`；28K为`1.773m vs 0.702m`、`0.250m vs 0.062m`。0.11并未带来更远carry，反而更容易后段失稳/掉物；这是同seed同gate结果，但仍只代表该diverse12。
- 交付=`/home/ubuntu/FAR/_check_vis/08-16-2110__xxr6at37_model21000_vs28000__forward007_vs011__diverse12__xy_trajectories/`：48 raw、48单条trajectory、12逐object 2x2、4 category 3x4、21K/28K各1个4x6 master，共114个H.264 MP4，全部501帧@50fps并完成ffprobe、全解码和frame400人工抽帧。21K/28K master SHA256分别为`944ca5650d46af3313afe4be0c50d60226973dd20c0b6d9a72f255a548efe605`和`8bac71d5bea0a3692a5a9806374a419e4d0150ccb77ccce782834ffc622ff0c3`。
- 审计root=`outputs/eval_xxr6at37_model21000_vs_28000_forward007_vs011_diverse12_20260816_211026/`，远端同名root保留完整约55MB/rollout的full policy-I/O；本地为节省满盘空间保留command+action lossless compact副本，compact manifest SHA256=`7c32ffbaa3d777355ea2addcb6d4189d32f2de7639211666cf0e9ce535b24e85`。为释放空间，已删除本机33K、21K-vs-28K-0.15及本轮的重复`runs`缓存；前两次`_check_vis`交付未动，full audit仍在远端对应campaign root可恢复。

## kdw7jhze 40K vs oosdgi7q 40K original30-shared diverse12 0.07/0.11（2026-08-16）

- fresh W&B API只读复核两条run均`finished`、summary step39999，terminal均为`model_40000.pt`。KDW PT SHA256=`17d5ba030c3a7486258c874ea05013fd7142ec33ce042445e42431171b0e7942`，是original30 pure-BC depth student；OOS PT SHA256=`fb9ff738b56e43743adc357dc21584519d22d252d4e75407a22ee5828ff1540b`，是exact109 PPO+BC depth distillation student。OOS严格保持训练时D435 translation z=`0.41987m`、物理俯角47.6度，未改成workspace后来的0.42987m。
- 两条训练support不同，因此没有把xxr diverse12中的OOD box/额外barrel硬塞给KDW；公平对比从OOS完整包含的KDW original30中固定12条共同support：5 ball、唯一1 barrel、6个scaledown/unscale且尺寸不同的bin。single env、seed42、timestep0、randomization/reset disabled、501 steps@50Hz。
- 两条都执行同一语义：object相对初始world-z达到`+0.30m`前root command三维全零；达到后持续`[0.07,0,0]`或`[0.11,0,0]`，consecutive=0、无heading lock、drop逐帧为0。checkpoint observation layout不同并分别审计：KDW前缀为`[dx,dy,dyaw,pickup,drop]`，OOS不读取pickup button，前缀为`[dx,dy,dyaw,drop]`；不得再用同一固定slot把KDW pickup误判成drop。
- 48/48均触发lift gate，逐条501 metrics、501 policy-I/O、唯一501帧raw H.264及command/gate/action finite验收通过。strict-end-carry定义沿历史审计为末帧robot-z/object-z都>0.5m且robot-object XY<0.7m：KDW 0.07=`12/12`、0.11=`12/12`，平均post-trigger carry fraction均1.000，median object net XY分别`5.154/5.681m`；OOS 0.07=`5/12`、均值0.565、median XY1.787m，0.11=`2/12`、均值0.364、median XY1.183m。固定12条上KDW明显更稳，且OOS的0.11比0.07更差；不能外推为全bank总体成功率。
- 本地交付=`/home/ubuntu/FAR/_check_vis/08-16-2312__kdw7jhze_model40000_vs_oosdgi7q_model40000__forward007_vs011__original30_diverse12__xy_trajectories/`，含48 raw、48单条trajectory、12逐object 2x2、3 category、两个单run master和一个总8x6 master，共114个MP4。114/114 ffprobe、完整decode和SHA复核通过，三个master六时刻人工抽帧确认正确robot/object、连续动作和动态轨迹；真实失持/跌倒未裁掉。总master SHA256=`7e39dfd1ef7becd869bde4917edcd89b1ed306cc0cd7ff0d85962f47a2dfb870`，eval manifest SHA256=`d52028f2deb41a96ccc06fb07054c7eb7a0f447fd0c5c61f1de2344ff4277b4d`。
- 完整远端审计root=`/data/holosoma_eval_audits/kdw7jhze_vs_oosdgi7q_model40000_forward007_vs011_original30_diverse12_20260816_231251/`，保留full policy-I/O；本地compact manifest SHA256=`03a5768bd2ab14c0298285216f103beb8cb75c8295bd2c341595ab34cda79f10`。全程未写W&B、未触碰训练；结束后eval节点8张GPU均0MiB/0%，无残留campaign进程。

## 4cnc2pjd-derived control-chain DR privileged teacher（2026-08-17）

- 用户要求在另一台独立节点上，以`zihanw22/carry-any/4cnc2pjd`为科学基线启动8卡fresh
  formal teacher，并补齐当前student实际使用的actuator/control-chain DR。fresh W&B run为
  `zihanw22/carry-any/7hvy6x2i`：
  <https://wandb.ai/zihanw22/carry-any/runs/7hvy6x2i>；节点=`ip-10-99-1-122`
  （`10.99.1.122`，ap-northeast-2b），8xL40S，tmux=
  `holosoma_teacher109_controlchain_7hvy6x2i`。该节点与parent所在`10.99.0.176`不同且为单节点独占
  本job；world size8、2048 env/rank、global16384、seed42、target40K、save interval1000。
- 相对parent唯一科学增量为：per-joint zero/calibration bias=`U[-0.01,0.01] rad`，Kp/Kd scale
  各=`U[0.9,1.1]`。这与当前student runtime实际范围一致。student当前并未启用torque RFI或随机
  delay，因此teacher也明确保持`torque_rfi_enabled=false/rfi_limit=0`、
  `action_delay_enabled=false/steps=[0,0]`；不得把仓库旧测试里过时的RFI/delay预期误当成当前
  student contract。新preset=
  `g1_29dof_wbt_w_object_teacher_state_robust_control_chain_with_camera`，W&B fresh API已读回上述
  exact数值。
- 其余严格保持`4cnc2pjd`：exact109 bank与ws8 shard、Rev-1.0 G1和real-mesh category-mass object
  URDF、D435 47.6度及camera DR、broad reset/push/rigid-body/object DR、178D actor/310D critic/
  29D action、`[512,256,128]` MLP、24 steps/7 epochs/4 minibatches、adaptive PPO、reward、
  termination、`precomputed_turn_then_forward` command和sampling均未改。fresh actor/critic与
  optimizer，无resume、policy init、外部teacher、distill、BC、DAgger或offline contact-guidance
  reward；验收读回`ppo_coeff=1`且`bc_loss=distill_loss=dagger_weight=0`。
- 为避免dirty workspace漂移，从parent immutable source单独派生并双构建一致的只读source=
  `src-2d2891d0228f84d51c5d79d7ed6555a06638603ce134ff9e9c96fd8b0d2737bb`，archive SHA256=
  `986a9c8759af1aa563512af9f5b703c42f6102d01a3b31a4242d20526f995be5`，worker SHA256=
  `13c886418f8f3e7b0f3a4ce48da66fdcb4327a47503b01f87a95f36534c72fcc`。24项针对性测试通过；
  immutable run contract SHA256=
  `079b313e30bfe6c03832f78c78b5409e9ceebf296ba5ec2e49dbce22c2c93eab`。
- exact 1x8/e2048 canary自然完成2 PPO updates并原生生成same-iteration PT+ONNX；PT/ONNX SHA256=
  `3b5d46eb281c3bf20eb85f4b20760915b9e13cb7169736cb14a4c2a123328863/2d905c675016915432bf013ec2ad677f85ec15afa831d51cc266196882c3500f`。
  ONNX checker、ORT CPU load、6-row finite probe及PyTorch-vs-ORT parity均通过，max abs/rel=
  `1.4901161193847656e-7/2.9665843612747267e-5`。formal固定`training.export_onnx=true`，每1000
  iteration必须原子发布同iteration PT+ONNX；bounded acceptance在iteration40，尚无正式
  checkpoint，禁止把canary pair称为正式checkpoint。
- Rule-90从最终source/bank重新录制canonical `box_10` reference replay：H.264 1280x720@50fps、
  368帧/7.36s，视频SHA256=
  `1528a12b3ee081a593f507bb6fcee38a358879c53cdd2009ce925cd60a5ebc5b`；人工contact sheet确认
  approach/pickup/carry/drop/return连续且robot/object正确。fresh API最终为history step0恰好一行
  `vis/replay`和一个MP4，summary指向该history-backed文件，prebind副本已删除。
- 2026-08-17 01:04 UTC bounded acceptance：remote completed iteration>=40、W&B summary step>=39且
  state=`running`；tmux存活、8 compute apps/8 unique GPU UUID、volatile UECC=0、exit sidecar为空，
  首个PPO以后fatal/OOM/non-finite/PhysX-capacity及Vulkan/renderer新增均为0，W&B numeric finite。
  rank-visible headless启动期保留`No device could be created=21`、`gpu.foundation=600`、
  `renderer.plugin=28`日志命中，但8 ranks随后完成Gloo barrier、权重digest同步和连续PPO updates，
  不得隐瞒启动诊断或误报为训练failure。审计root=
  `outputs/formal_teacher109_controlchain_ws8_e2048_40k_20260817_002056/`，bounded acceptance SHA256=
  `001ca0cf5dd6893d2de727b4d883bb78ef8e7d47f4c5206fb63af3007d1c1e16`。

## oosdgi7q 40K diverse12 post-lift forward 0.07/0.15（2026-08-17）

- Fresh W&B API只读复核`zihanw22/carry-any/oosdgi7q`为`finished`、summary step39999，terminal atomic pair=`model_40000.pt/.onnx/.pair.json`。PT/ONNX SHA256分别为`fb9ff738b56e43743adc357dc21584519d22d252d4e75407a22ee5828ff1540b`和`f212b7ef10d7cb4613daecfa8ebbf23eecc4a55c11f93a20c7cfc6082cf3fb03`，pair声明PyTorch-vs-ORT parity通过。相机严格使用该run训练值：D435物理俯角47.6度、translation z=`0.41987m`，没有使用workspace的`0.42987m` override。
- 复用xxr评估的diverse12：box/ball/barrel/bin各3个；single env、seed42、timestep0、randomization/reset disabled、501 steps/frames@50Hz。默认0.15与额外诊断0.07都遵守统一lift gate：object相对初始world-z达到`+0.30m`前actor root command为零，下一step起直到结束严格`[dx,dy,dyaw,drop]=[0.07或0.15,0,0,0]`；consecutive=0、native pickup cue保留、drop全0、无heading lock。24/24均触发并通过501 metrics、501 policy-I/O、唯一501帧raw H.264、finite action与gate alignment验收。
- 历史strict-end-carry定义为末帧robot-z/object-z均>0.5m且robot-object XY<0.7m：0.07=`4/12`，平均post-trigger carry fraction=`0.446`、median object net XY=`0.576m`；0.15=`11/12`，分别为`0.959/5.422m`。0.07未命中为`box_75/noscale__any_ball_3/scaledown__any_ball_26/noscale__any_barrel_12/scale__any_barrel_53/scale__any_barrel_80/noscale__any_bin_32/scaledown__any_bin_21`；0.15仅`box_10`未命中。该同seed diverse12上0.15明显优于0.07，但不得外推为全bank成功率。
- 交付=`/home/ubuntu/FAR/_check_vis/08-17-0220__oosdgi7q_model40000__forward007_vs015__diverse12__xy_trajectories/`：24 raw、24单条trajectory、12逐object左右对照、4 category 2x3和1个4x6 master，共65个H.264 MP4。65/65均501帧@50fps并通过ffprobe、完整decode与master/category多时刻人工抽帧；左列0.07、右列0.15。master SHA256=`725972cbbda3be5d49a5b43e54b2846cbade4b03a4bab16bb9f09a1d240b59a7`，eval manifest SHA256=`e4d5441b168c8dfab13187799755bec1898602fd356e7bbf1306f062d0aeb5a3`。
- 完整远端审计root=`10.99.1.148:/data/holosoma_eval_audits/oosdgi7q_model40000_diverse12_forward007_vs015_20260817_022001/`，保留约55MB/rollout的full policy-I/O；本地精简审计root=`outputs/eval_oosdgi7q_model40000_diverse12_forward007_vs015_20260817_022001/`，compact manifest SHA256=`29da4297441c37cabc8ce56af5af582804000fa3e5d59231fb2e26b0c0064301`。W&B全程只读、未触碰训练；结束后eval节点GPU1--3无本campaign进程，GPU0原有Viser进程未触碰。失败节点上本任务创建的`20260817_022001`临时root已删除，可恢复审计仍在成功节点。

## kdw7jhze terminal actor -> original30 precomputed/drop-exclusive pure-RL fine-tuning（2026-08-17）

- 用户要求在一台独立8卡机器上取`zihanw22/carry-any/kdw7jhze`最新checkpoint，在30条数据上做
  RL fine-tuning，使用pre-computed command，并在drop button active时把其他command清零。fresh
  formal run为`zihanw22/carry-any/saoc6d5j`：
  <https://wandb.ai/zihanw22/carry-any/runs/saoc6d5j>；节点=`ip-10-99-1-134`
  （`ubuntu@10.99.1.134`，AWS name=`sky-zzzihanw-102-1452fa42-head`），8xL40S，tmux=
  `kdw_rlft_saoc6d5j`。world size8、2048 env/rank、global16384、seed42、target40K、save
  interval1000；正式console、rank logs、checkpoints与control均在
  `/data/holosoma_training/formal_kdw40000_rlft_original30_precomputed_dropzero_ws8_e2048_40k_20260817_025923/`。
- source checkpoint是KDW terminal `model_40000.pt`，SHA256=
  `17d5ba030c3a7486258c874ea05013fd7142ec33ce042445e42431171b0e7942`，completed/next
  iteration=`39999/40000`；terminal ONNX SHA256=
  `c42f79d4c2b3f200930d2a60a1f07111fcfb52ef4f14d8188ed89b5643e496a0`。新run只做
  actor policy-init，不是training resume：runtime日志明确忽略source iteration、critic、optimizers、
  critic normalizers和env state，actor normalizer无需恢复，fresh critic/optimizers/RNG/env/curriculum/
  W&B identity从iteration0开始。唯一允许的actor-contract migration为
  `tracking_error_to_precomputed_turn_then_forward_drop_exclusive_v1`；除此之外actor/perception/action
  contract exact匹配，任何其他漂移preflight fail closed。
- 数据是KDW original30的exact immutable view，不是109全bank：5 ball、1 barrel、24 bin、0 box；30/30
  原始motion numeric payload保持，另派生precomputed command/phase。single-slot source digest=
  `f8e3fde08b463b153fcaffd23c935d37dd2d6e41cc382e4597129f1a2154b963`，manifest SHA256=
  `c4939095b71853a814d333828f0ead71185932dfbd08d5b527e5ad1f1debac8f`；ws8 rank counts=
  `[4,4,4,4,4,4,4,2]`、global exact-once且均整除2048，rank manifest file SHA256=
  `8245dbd479c29cffffd018a591785c71c81431bef7a064e04f1ce315d9117d30`。运行时每rank只加载自己的
  immutable shard，并保留full contact-window coverage用于adaptive timestep sampling；offline contact-guidance
  reward weight严格为0。
- actor command mode严格为`precomputed_turn_then_forward`，runtime pickup latch保留；drop active时最后一道
  exclusivity强制实际actor输入为`[dx,dy,dyaw,drop]=[0,0,0,1]`，包括external override以后也不能重新注入
  root command。phase总数为zero/forward/yaw=`1776/7622/172`，`dy`全零且`dx/dyaw`不重叠。训练是pure
  PPO：teacher/distill/BC/DAgger/replay-BC均关闭，W&B读回`ppo_coeff=1`且上述loss/weight均0；24
  steps/env、1 epoch、64 minibatches、actor/critic adaptive LR初始`1e-3`、desired KL=`0.01`、entropy=
  `0.005`。初始7 epochs x4 minibatches真实canary在首个PPO backward OOM且未产checkpoint；KDW自身使用的
  1x64 geometry在相同2048 env/rank与原PhysX capacities下完成2 updates，因此正式采用1x64，没有降低
  env count或改actor batch contract。
- immutable source=`src-7351e6e7ab96b83cec8be948149dcde533e5b0803d9444ea0cb856640e679aef`，archive
  SHA256=`3ee018c40c45e82688bcf7dadca013c0ca90bb78843cd5d3511079b36105a33d`；run contract SHA256=
  `ba1bd89e472e89681207c6911d098d1372c1720c7ebb73b183a4f4d0175410c7`。exact 1x8/e2048/1x64
  canary自然完成2 PPO updates并原生生成same-iteration PT+ONNX；PT/ONNX SHA256=
  `8228665a2338016f2f64abba0c543b1592cdea023f86ae75ae8549522d4afac4/06da36a7a6c53f065dbe37536a7dc08b1a34a5b958762bf75768df45ebacd6a5`。
  ONNX checker、ORT CPU load、6-row finite probe及PyTorch-vs-ORT parity通过，max abs/rel=
  `0.00025177001953125/0.00016409829549957067`；formal固定`training.export_onnx=true`，每1000
  iteration必须原子发布同iteration PT+ONNX并fail closed。bounded acceptance尚未到iteration1000，
  不得把canary pair称为formal checkpoint。
- Rule-90从exact original30 canonical第一条`scaledown__any_ball_24`重新录制policy-free reference replay，
  H.264 1280x720@50fps、329帧/6.58s，视频SHA256=
  `253e3304f53ba61e3851df80c314c62fd3433469b09ff6a973cfa8157e5a0d21`；完整decode与人工contact sheet
  检查确认approach/pickup/elevated carry/drop/return连续且物体正确。fresh API最终为history step10恰好
  一行`vis/replay`、一个MP4，summary指向history-backed文件，prebind副本已删除。首次formal exact
  preflight因自定义prebind漏写完整flattened replay metadata而在GPU前fail closed；补同一immutable
  manifest对应metadata后通过。history promotion后public-API summary曾被live primary的旧内存值覆盖，随后
  直接attach其W&B service绑定history指针，并跨后续16+ history steps复核不再回退；两次均未改科学contract、
  video bytes或run identity。
- 2026-08-17 04:07 UTC bounded acceptance：remote completed iteration=50、W&B last history step=47且
  state=`running`；8 compute apps/8 unique GPU UUID、volatile UECC=0、tmux存活、exit sidecar为空，8个rank
  与console在进入PPO以后fatal/OOM/RuntimeError/non-finite/NCCL/PhysX-capacity均0；W&B 10,124个numeric
  values抽样全部finite。rank-visible headless启动期保留`No device could be created=21`、
  `gpu.foundation=599`、`renderer.plugin=21`日志命中，但8 ranks随后完成simulator/Gloo barrier、actor-load和
  连续PPO，不能把启动诊断误报为training failure。审计root=
  `outputs/formal_kdw40000_rlft_original30_precomputed_dropzero_ws8_e2048_40k_20260817_025923/`，bounded
  acceptance SHA256=`37f1d1406ad8b03efc6bba944c305b45553cd0b9a237caa4d1b908949cd0fac5`。

## 4cnc2pjd frozen-label teacher：PPO/BC contact A/B 与 pure-DAgger no-RL（2026-08-17）

- 用户要求以`zihanw22/carry-any/4cnc2pjd`最新可用的atomic checkpoint `model_23000`作为
  frozen label teacher，在三台新的单节点8xL40S上训练三个fresh student。teacher PT/ONNX/pair SHA256=
  `99bf62f3b310cbcb8fa5637819b83da3a66397c0269096648c745a588ee29d7b`/
  `539c01288a0c04058e063e94c341af39c322f18cf47380007680dfbbe0b7d614`/
  `cdbdd6f03a73d1b475f9fc6a7e21f8f9d195b3307f2a6ca42c0e8bfb85550a78`，completed/next=
  `22999/23000`。三条都不resume、不做policy init；teacher只产生state-action监督标签，student actor从随机
  初始化开始。immutable source=`src-2d2891d0228f84d51c5d79d7ed6555a06638603ce134ff9e9c96fd8b0d2737bb`，
  archive SHA256=`986a9c8759af1aa563512af9f5b703c42f6102d01a3b31a4242d20526f995be5`。
- 三条共享exact109 precomputed-turn-then-forward bank：box/ball/barrel/bin=`25/9/36/39`，ws8 rank
  counts=`[32,16,16,16,16,8,4,1]`，single-slot view digest=
  `42be48688b001bb21fa65b0229912e6316d3d2cc174dd67663c76b1c66413d52`。robot是G1 Rev-1.0 29DOF，
  real-mesh convex-decomposition objects；D435物理俯角47.6度、translation=`[0.0576235,0.01753,0.42987]m`。
  actor是SW-sized MLP base input126、hidden `[512,256,128]`、action29，加external small CNN：actor/perception
  observation=`94/5046`、embedding32。控制链DR固定joint bias `[-0.01,0.01]rad`、Kp/Kd `[0.9,1.1]`、
  delay `[0,1]`和torque RFI `0.01`，同时保留较宽的物理与camera DR。
- A/B两条都是40K、ws8、2048 env/rank、global16384、24 steps、7 epochs、4 minibatches、seed42、
  save interval1000；closed-loop student state collection、teacher action mix0、无replay。PPO/BC从`0.1/0.9`
  开始，每500 updates阶梯变化，最终`0.9/0.1`，DAgger监督到iteration4000。两条resolved config唯一科学
  delta是`reward.terms.offline_contact_guidance.weight`：A为0；B为1，B参数contact/wrist=`4/3`、force
  threshold=`1.4`、position/force sigma=`0.08/10`。两条物理robot-object collision都开启；“no-contact”仅指
  offline contact-guidance reward关闭，不是关闭碰撞。
- A/no-contact formal run=`zihanw22/carry-any/ko26vz7y`：
  <https://wandb.ai/zihanw22/carry-any/runs/ko26vz7y>；节点=`z1hanw-d4cn23-nocontact`
  （AWS `g6e.48xlarge`, ap-northeast-2a, private `172.31.6.100`），tmux=`formal_4cn23_nocontact`。
  B/contact formal run=`zihanw22/carry-any/t0f97aey`：
  <https://wandb.ai/zihanw22/carry-any/runs/t0f97aey>；节点=`z1hanw-d4cn23-contact`
  （AWS `g6e.48xlarge`, ap-northeast-2b, private `172.31.20.18`），tmux=`formal_4cn23_contact`。
  2026-08-17 08:37 UTC fresh API为A/B history step=`324/321`、均`running`；BC loss=
  `0.82931/0.83197`、actor loss=`0.74641/0.74886`、critic loss=`0.02719/0.02969`、KL=
  `0.01244/0.01210`、mean reward=`1.3601/1.4773`、FPS=`49802/48760`。B有
  `Episode/rew_offline_contact_guidance=0.01602`，A该reward metric缺失，符合A/B契约。
- 第三条是用户后加的真正closed-loop pure DAgger，而不是把PPO权重设小：formal run=
  `zihanw22/carry-any/0w65706p`：<https://wandb.ai/zihanw22/carry-any/runs/0w65706p>；节点=
  `z1hanw-d4cn23-puredagger`（AWS `g6e.48xlarge`, ap-northeast-2b, private `172.31.30.17`），tmux=
  `formal_4cn23_puredagger_0w65706p_retry2`。student自己闭环采state，teacher mix=0并只label；整个40K的
  PPO/RL coefficient严格为0、BC/DAgger weight=1，actor-only optimizer step，critic objective/optimizer均关闭，
  replay关闭，offline contact-guidance weight=0；物理碰撞仍开启，contact sidecar只用于sampling与预计算command。
  2026-08-17 08:37 UTC fresh API为17个training rows、state=`running`：BC=actor=`0.305738`、critic=0、
  KL=0、PPO=0、replay BC=0、teacher mix=0、teacher BC mask=`0.92979`、mean reward=`2.4461`、FPS=`49525`。
- A/B exact full-scale 2-update canary的唯一科学diff已机器比对，resolved config其余一致；两臂都原生生成
  same-iteration PT+ONNX，ONNX checker、ORT load/inference与PyTorch parity通过。canary acceptance SHA256=
  `c894dd456fc88fbf216c8a3aa805c1b12ab5393f4418dee7a66ccd524a08ec1a`。pure-DAgger exact full-scale
  canary也完成2个actor-only updates，观察到BC=actor=`[0.6516,0.5314]`、PPO/critic/KL全0、actor/critic
  optimizer state entries=`16/0`；canary PT/ONNX SHA256=
  `9cffa0be6a9019615843c995c8f4d0da1fa63c6b387682b1cffbe4fc72b14090`/
  `f8c8922b5ce19ad71855b82c6c780805d2b70fed5aabdb047040a2af1aa03237`，parity max abs/rel=
  `4.76837158203125e-7/0.00016330232028849423`，acceptance SHA256=
  `b8e69e58fd812be3d5fd8e17b46426f70fa47e9d3daa82b391bac62e98f690f6`。
- 三条formal均固定`training.export_onnx=true`，每1000 iteration必须原子发布同iteration PT+ONNX pair并完成
  checker/ORT/parity，失败则checkpoint boundary整体fail closed。当前尚未到formal iteration1000；上述
  `model_00002` pair都只是canary，禁止称为正式checkpoint。三条共享的Rule-90是policy-free canonical
  `box_10` reference replay，H.264 1280x720@50fps、368帧/7.36s，视频SHA256=
  `b8f2facfbc10c09fbfbdab556ea44e7627ca38158b49e5f299287b42cfe55f55`；每个formal W&B run均复核恰好一个
  history-backed `vis/replay`行和一个MP4。
- pure节点canary前两次分别因dataset parent lock ownership、缺少exact editable IsaacLab tree而在sim/checkpoint前
  fail closed；补齐后第三次通过。pure formal第一次因新节点缺W&B credential，在environment creation、optimizer
  update和checkpoint前退出；失败现场保留在remote
  `failed_formal_attempt1_missing_wandb_credentials/`。从同账号已认证A节点安全复制credential并验证API后，以
  同一immutable contract、同一W&B identity和`resume=must`重启成功；没有丢弃或伪造训练进度。08:37 UTC三台
  tmux均存活、各8 compute apps/8 unique GPU UUID、24张卡volatile UECC均0、active exit sidecar均为空。
- 完整本地审计root=
  `outputs/formal_distill_4cn23k_swab_ws8x2_e2048_40k_20260817_061827/`。A/B contract SHA256=
  `40fd8f3cad6fcd476db6d2c2bec2c75105e7f13e00470dbf325b24f7f658055f`/
  `58e714904962cd7a578353144dc115d08703dbf831cbaa6f6a56ad8e0790fe0b`，pure contract SHA256=
  `c7ccda2d61b00449704c2b444f6b54936645700cd0e9bdaa58272a3493ac70c6`；A/B initial health与pure initial
  health acceptance SHA256=`7cf1c7f6d7f786fabfdd90e184b034dd850d7bd4e365bf0383259c2df6f15bff`/
  `6487868263644ad13b183a602b55efff08c52b6419649e2f20e1e3ef09b060b1`。

## 0mcqao8k 最小 Spatial Softmax depth readout 单卡实验（2026-08-18）

- 用户要求以`zihanw22/carry-any/0mcqao8k`为base做真正最小的depth位置感知实验。formal没有使用当前dirty
  worktree，而是从0mc immutable source `src-309efb4303690cea9960b10a2e459041608243842e284f2732cd27f0cccbca57`
  解包；新source archive与base逐文件比对后恰好只改
  `src/holosoma/holosoma/agents/modules/modules.py`，无新增/删除文件。新immutable source=
  `src-c4f9efa5aba1b7e68a7d275da1462ff827f366b2e7071e9b2936507d56248a59`，archive SHA256=
  `03a5e4b441838cedcc115dee8ee0d2df9b3f32f522b29c689c754f9c1bf15a8e`；old/new `modules.py` SHA256=
  `6fe40663a1618175003cc69f49af93ff435cae3126cb67e6574bdfbc6a25cbec`/
  `6b870085faefc3c693b13a097ce1b7301fe5fca96f444a1db8cca778d50ba799`。
- depth结构固定为`[B,1,58,87] -> Conv(1,16,k5,s2,p2)+ELU -> [B,16,29,44] ->
  Conv(16,32,k3,s2,p1)+ELU -> [B,32,15,22] -> Conv(32,64,k3,s2,p1)+ELU -> [B,64,8,11]`；
  唯一模型readout delta是原`AdaptiveAvgPool2d(1,1)->Linear(64,32)+ELU`改成每channel固定temperature=1.0、
  `[-1,1]`坐标、`[x1,y1,...,x64,y64]`交错顺序的Spatial Softmax `[B,128] -> Linear(128,32)+ELU`。
  scalar obs=94、actor input=126、actor hidden=`[512,256,128]`、action=29以及外部ONNX I/O均不变；卷积参数完全
  不变，depth encoder只因projection增加2048参数（25,632 -> 27,680），actor MLP不变。
- 其余科学设置沿用0mc：fresh pure PPO、无resume/init/teacher/distill/BC/contact reward，exact109 bank
  box/ball/barrel/bin=`25/9/36/39`，precomputed-turn-then-forward+runtime pickup latch、24 steps、7 epochs、
  4 minibatches、相同reward/termination/DR，D435 z=`0.41987m`、物理俯角47.6度、58x87 processed depth、
  real-mesh depth与convex-decomposition collision。单卡拓扑不是原0mc的ws32/global65536，必须作为confound：
  L40S上2048不能被109整除而被assignment gate拒绝；2071/1962/1853/1635分别在真实初始化或Spatial Softmax
  forward中OOM并全部fail closed、未创建formal W&B identity。最终采用ws1/e1090=`109x10`，保持109条每条精确
  10 env以及4 minibatches，但global rollout batch显著小于0mc；禁止把它描述成除depth外连训练拓扑也完全相同。
- ws1/e1090真实两update canary完成，same-iteration PT/ONNX SHA256=
  `bfbad2389abfff88888fddc9a224a834f9e7ea2d0d835f0159127095c09e86b7`/
  `7e270979b7a41b24f8b7663fb2b9a9ea57d7936b8e5bf56aacfa17ce9eab0d36`。ONNX checker、ORT CPU load和
  PyTorch-vs-ORT parity通过，max abs/rel=`7.82310962677002e-8/1.7575808669789694e-5`；外部I/O是
  `actor_obs[batch,94]+perception_obs[batch,5046] -> action[batch,29]`。canary acceptance SHA256=
  `3e007b5d37f9860057a0f04b96df0635a0ec17803935a3b96836f40e2778a46c`。
- formal run=`zihanw22/carry-any/4v4jv0xv`：
  <https://wandb.ai/zihanw22/carry-any/runs/4v4jv0xv>，name=
  `pure_rl_0mc_spatial_softmax_ws1_e1090_onnx_20260818_011240`；节点=`zzzihanw-e`、private IP=
  `10.99.1.21`、GPU0 L40S、tmux=`formal_0mc_spatial_4v4jv0xv`。run contract SHA256=
  `a331eb14e4940c4e87eb18eff58c23f37c93a216ef3f6dec4eb52b76addabb7c`，Rule-90 manifest SHA256=
  `024973c209404950650d4fbc76a26de8c713ae6651477959807910847d642d4e`。fresh canonical `box_10`
  policy-free replay为H.264 1280x720@50fps、368帧/7.36s，视频SHA256=
  `dc6383c7eadf1f48e79482736ccc7e08cea7e405c027724fa3b66905a5078e38`；人工contact sheet与fresh API
  均通过，W&B `vis/replay`已绑定且保留。
- 2026-08-18 01:41 UTC bounded acceptance：remote completed iteration=18，W&B state=`running`、18行history、
  last history step=17，最后一行145个numeric values全部finite；GPU UECC=0、tmux存活、exit sidecar为空，formal log
  中OOM/Traceback/training exception/FATAL均0。`training.export_onnx=true`、save interval=1000，每个boundary必须
  原子发布同iteration PT+ONNX并完成checker/ORT/parity，失败则整体fail closed。验收时尚未到iteration1000，
  因此没有formal checkpoint；上述`model_00002`只属于canary，禁止称为formal checkpoint。initial acceptance
  SHA256=`1cc0b007ad77fd7d2054e5d02e8ee4488d1cf5916504773e4fcf4a9be3639a72`。remote audit root=
  `/data/holosoma_training/formal_0mc_spatial_softmax_ws1_e1090_40k_20260818_011240/`；本机因root filesystem已满，
  临时控制副本位于`/dev/shm/formal_0mc_spatial_softmax_ws1_e1090_40k_20260818_011240/`，长期审计以remote root为准。

## 0mcqao8k ball y=+0.15与depth/command反事实诊断（2026-08-18）

- 用户要求在已失败的`noscale__any_ball_3`左侧`y=+0.236m`附近补录`y=+0.15m`，并验证depth、command与
  proprio shortcut。使用exact `0mcqao8k/model_40000.pt`，SHA256=
  `7a2084e726c183c2345b6bfd0848be4baceaceb51335814fdb798da0223cf760`；目标robot-heading-frame初始物体
  `xy=[0.2142336869,+0.15]m`，对应world reset offset=
  `[-0.1079441252,+0.2002203810,0]m`。single env、seed42、timestep0、randomization/init noise/reset disabled；
  actor在lift前严格`[0,0,0,0]`，object world-z相对初始达到`+0.30m`后从actor step92持续
  `[0.15,0,0,0]`，consecutive=0、无heading lock、drop全程0。
- 物理rollout通过501 metrics、501 full policy-I/O和唯一H.264 640x360@50fps、501帧/10.02s验证。lift gate
  metric step91触发，最大object dz=`0.54248m`；末帧object/robot z=`0.68901/0.75642m`、两者XY距离=
  `0.22498m`，末帧仍抱持。对照中`y=+0.131m`虽触发但最终丢球，`y=+0.236m`未触发；由于前者x也从
  `0.214m`变为`0.193m`，三者不能被误报成严格单变量成功边界。
- 固定上述每个真实物理state，使用同iteration `model_40000.onnx`做offline单次actor forward反事实；这不是
  闭环rollout。ONNX SHA256=`b2eb5206e255efb7a8974def2aa533f3ad493378affd351829521d11c38a4483`，相对logged
  PyTorch action最大绝对差=`5.50e-4`。在`t1±10`的21帧内，水平镜像depth、换成右侧`y=-0.135m` depth、
  逐帧保留均值但抹去depth空间结构、换成右侧rollout proprio/previous-action（保留原command/depth）的29D
  action平均L2变化分别为`0.354/1.017/2.972/9.360`，相对baseline action norm为
  `2.41%/6.87%/19.94%/63.45%`。整个pre-lift对应为`0.610/1.106/2.707/6.090`和
  `6.04%/10.43%/29.57%/55.00%`。pre-lift command本来严格为零，所以zero/reverse command的action差也严格为0。
- post-lift把forward command从`+0.15`清零/反成`-0.15`，action平均L2变化为`1.376/2.713`（baseline norm
  的`10.19%/20.25%`）；同阶段镜像/右侧depth仅为`0.455/0.696`（`3.38%/5.08%`）。结论是depth pathway并非
  完全没学：去除空间内容有显著影响；但左右镜像响应尤其在t1附近很弱，actor更依赖proprio/previous-action。
  command不是pre-lift左右抓取的直接泄漏，却在lift后形成明显carry shortcut。此结果支持“弱左右视觉使用 +
  open-loop/proprio主导 + post-lift command主导”的组合解释，不能只归因于GAP或所谓SW uniform-T1 boost。
- 交付目录=`/home/ubuntu/FAR/_check_vis/08-18-0249__0mcqao8k_model40000__ball_y+0150__depth_command_counterfactual/`，
  包含raw、带XY轨迹单条、`y=+0.131/+0.150/+0.236`三列对照、反事实曲线/depth montage、逐帧CSV、JSON和
  manifest；3/3 MP4均H.264@50fps、501帧/10.02s并完整decode。远端full审计root=
  `/data/holosoma_eval_audits/0mcqao8k_model40000_ball_y015_depth_command_counterfactual_20260818_024914/`。

## 3m8lkcxf/model_23000 CORL79 diverse12 post-lift forward inference（2026-08-18）

- 用户要求infer `zihanw22/carry-any/3m8lkcxf`。fresh W&B冻结的最新完整checkpoint是`model_23000`，不是把
  run summary `_step=23079`误当checkpoint；PT/ONNX大小为12,948,989/1,227,731 bytes，SHA256=
  `c6fab2035194da6deeaa565f007d7d3025e8f30facb4b00afa9e88f5e4456fd1`/
  `161d23871e574294e0c86ea4003819f496ddc8b4dff20d189175f82932260b62`。checkpoint内部completed/next
  iteration=`22999/23000`，ONNX checker通过，外部I/O=`actor_obs[batch,94]+perception_obs[batch,5046]
  -> action[batch,29]`。run已经`finished`；本次只读W&B并下载artifact，没有向原run写media或改变生命周期。
- 这是fresh pure PPO depth student：无teacher/BC/DAgger/init/resume，actor=`root_contact_aware + drop_button +
  proprio_with_actions_no_linvel + depth`，29D action。评估使用它的immutable source
  `src-a8879814e247e78b0e686fb8bdf94b985d008b2960aa187b482ef8de7b644399`与exact CORL79 training support
  view digest=`6209b4742cce3b2989c7ea1f96a55a27d57bcf91eeb90699d409747187ca2cca`；从四类各取三条：
  box=`10/62/75`，ball=`noscale 3/6/84`，barrel=`noscale 12, scale 53/80`，bin=`noscale 32,
  scale 5/85`。single env、seed42、timestep0、zero init noise、physical DR与所有auto/motion/clip/bad-tracking reset关闭。
- checkpoint原始perception契约完整保留：`far_tracking_warp`、raw `60x106`、crop/resize `58x87`、sensor offset
  `[0.01,0.01,0.44]m`、mount quaternion xyzw=`[0.00644801,0.23350163,0.00644801,0.97231365]`，edge/hole/
  sensor noise active，hole p=`0.2`、additive/depth-offset std=`0.03m`、latency=`[3,4]`，并绑定训练时hole
  reference batch=`1024`。旧source早于通用lift-gate helper，因此只在exact-source inference loop中、actor forward
  前覆盖原始command group：lift前严格`[0,0,0,0]`；首次object dz达到`+0.30m`后立即且持续
  `[0.15,0,0,0]`；consecutive=0、robot-heading frame、无heading lock、native pickup不改、drop全程0。
- 12/12均有501 metrics与501 full policy-I/O，逐帧验收全部通过，且12/12均触发而无`not_triggered`。trigger
  actor step范围85--152（1.70--3.04s）；最大object dz=`0.449--0.650m`，末帧仍比初始高
  `0.391--0.619m`；末帧robot-object XY距离=`0.205--0.463m`，robot root全程最低=`0.602--0.638m`。
  因此这12条确定性样本都完成pickup并在10.02s末仍保持物体，不能外推成79条总体成功率。
- 虽然actor的dy/dyaw逐帧严格为0，运动并不总是直线：robot净位移`0.414--8.252m`；例如ball6路径
  `6.27m`但净位移仅`0.41m`、box10路径`6.42m`但净位移`1.00m`，轨迹出现明显转弯/绕圈。这里证明的是
  command input纯forward，不证明policy会把它稳定映射成无yaw漂移的world直线。
- 交付目录=`/home/ubuntu/FAR/_check_vis/08-18-0323__3m8lkcxf_model23000__corl79_diverse12__post_lift_forward015__xy_trajectories/`；
  含12条individual trajectory MP4、4x3 master、contact sheet与`eval_manifest.json`。所有individual与master均为
  H.264/yuv420p@50fps、501帧/10.02s；master=`2560x1860`、SHA256=
  `d00bf690e9ab8e739a8724e16f2fd1c86f0cdef152754d099f669d3616703a23`。四时刻contact sheet人工确认正确
  box/ball/barrel/bin资产、G1与物体连续可见、轨迹面板正常；启动阶段因root盘满产生的3条零帧attempt未计入结果，
  清理本次新生成且可重建的robot USD conversion caches后均已完整重跑通过。

## 0mcqao8k 最小 Spatial Softmax depth readout 32卡正式实验（2026-08-18）

- 用户明确否决此前 `4v4jv0xv` 的 ws1/e1090 拓扑妥协，要求保持原 0mc 的 32 卡训练规模。新 formal
  run=`zihanw22/carry-any/4ujnckj9`：
  <https://wandb.ai/zihanw22/carry-any/runs/4ujnckj9>，name=
  `pure_rl_0mc_spatial_softmax_ws32_e2048_onnx_20260818_041727`。这是 fresh pure PPO：无 resume、policy init、
  teacher、distill、BC 和 offline contact-guidance reward；exact109、command、camera、DR、PPO 与 0mc 对齐，科学
  delta 仍只有 depth readout 从 GAP 换为 `SpatialSoftmax[64x(x,y)]->Linear(128,32)+ELU`。immutable source=
  `src-c4f9efa5aba1b7e68a7d275da1462ff827f366b2e7071e9b2936507d56248a59`，run contract SHA256=
  `99d8b970d412a6c02a2dde1a6ac5400032b62349d691671795956161a0d309bf`。
- 正式集群为同一个 AWS region/AZ 的四台完整 8 卡节点，不是每台取 4 卡：region=`ap-northeast-2`、AZ=
  `ap-northeast-2b`，均为 `g6e.48xlarge`、8xL40S。rank0--3 分别为 `z1hanw/10.99.1.60`、
  `zzzihanw-z/10.99.1.69`、`zzzihanw-72/10.99.1.89`、`zzzihanw-79/10.99.1.154`；world size=32、
  2048 env/rank、global env=65536。节点间实测平均 RTT 范围 `0.148--0.170ms`；32-rank 256MiB/rank NCCL
  all-reduce probe通过，worst-rank median/max=`354.830/786.006ms`、algorithmic/bus bandwidth=
  `0.757/1.466GB/s`。四台没有显式 placement group，因此只能陈述同 AZ 与实测通信，不得声称 placement-group
  保证。
- exact-topology 两更新 canary成功；same-iteration PT/ONNX SHA256=
  `9403d3fdf2eda9699597a8b015099ffa92d37da82800d8ea5ba1c9b565050e68`/
  `c856e9eb350bbc654552d46ed167490e7ff2d7404ba68d7d2da6a531ef75dc42`，checker、ORT与PyTorch parity通过，
  max abs=`1.2665987014770508e-7`，canary acceptance SHA256=
  `fe482f93285e1599069af954055f4ee36371edf31f916f7a63a1006db2703e55`。formal固定
  `training.export_onnx=true`、save interval=1000；每个checkpoint boundary必须原子发布同iteration PT+ONNX并
  完成parity，否则fail closed。canary pair不是formal checkpoint。
- 2026-08-18 04:42 UTC bounded acceptance：W&B state=`running`，23个numeric rows、last step=22，全部数值
  finite；latest FPS=`97056`，distributed loss weight sum=32，PPO coefficient=1，BC/DAgger/distill/replay-BC均0。
  四个tmux均存活、每台8 compute apps与8 unique GPU UUID、32张卡volatile UECC全0、exit sidecar均pending，
  OOM/NCCL/Gloo/Traceback/Xid/non-finite匹配均0。Rule-90 fresh canonical `box_10` replay是H.264
  1280x720@50fps、368帧/7.36s，视频SHA256=
  `fce634838a253619e53cc3884ebce719678b316052dbcf5473a8b20f33f217c2`；W&B已复核恰好一个history-backed
  `vis/replay`行和一个远端MP4。initial acceptance=
  `outputs/formal_0mc_spatial_softmax_ws32_e2048_40k_20260818/formal_initial_acceptance.json`。
- 被用户否决的单卡 run `4v4jv0xv` 在 `10.99.1.21/GPU0` 仍保持运行；本次未获得停止它的明确授权，所以没有
  中断。后续若用户确认，可单独停止该旧 ws1 run；禁止把它和新的 ws32 formal 混为同一实验。

## 11-checkpoint random/depth/carry/forward 多轴评估（2026-08-18）

- 用户要求用数小时系统比较 randomized objects、depth响应、搬运稳定性、合适forward command及teacher-student、
  RL、bad-tracking、DR、depth camera等变量。campaign=`multiaxis_policy_ablation_eval_20260818_064433`，冻结11个
  checkpoint：`0mcqao8k/40000`、`oosdgi7q/40000`、`swl41n4x/39999`、`kdw7jhze/40000`、
  `z9e7vxcv/24000`、`2xmp4whp/24000`、`tuhu3ghf/30000`、`11xg5p3k/30000`、`xxr6at37/45000`、
  `ob0odtkx/45000`及response-only的`4ujnckj9/01000`。checkpoint inventory SHA256=
  `df5ccd2497cba470d2e9c4804b7f250e6cffa2089291fa1e5fc3a0038d783043`。
- 共620条：每policy random-lateral16、5-command×4-object=20、DR 4-object×4-seed=16；另对6个policy做
  mirror/constant-mean closed-loop depth各8。全部single-env、timestep0、501 steps@50Hz；lift前command严格0，
  object dz达到+0.30m后持续指定pure-forward，drop全程0。620/620有501 metrics+501 full policy-I/O+唯一
  H.264 640x360/501-frame视频，command/drop/depth-transform验收全通过；scheduler两阶段均failed=0。
- 最平衡的是OOS：random lateral terminal75%、DR terminal100%/completion75%；0.15与0.20固定四物体均
  terminal4/4、completion3/4，但0.15 yaw较小，部署优先0.15。tuhu的0.15是唯一固定四物体4/4 completion；
  xxr DR terminal100%/completion62.5%，但lateral completion31.25%。
- SW在random lateral16/16及command sweep20/20都末帧抱持，但random completion18.75%、DR completion0；
  0.05/0.07/0.11物体中位forward progress为负，0.15 yaw约5.84rad。因此它是carry-prior robust，不是
  command-following robust。KDW pure-BC random/DR terminal仅25%/31.25%，个别失败帧action爆到千量级；
  teacher label/T1 boost本身不充分，OOS收益只能归于joint PPO+BC/data/actor/sampling recipe。
- 严格threshold A/B显示`11xg5p3k(0.75x)-tuhu3ghf(1.0x)`在forward sweep trigger=-25pp，paired bootstrap
  95%CI=[-45,-10]pp；terminal/completion、DR、yaw没有可靠收益。当前保留1.0x，不建议直接0.75x。
- 严格camera 24k A/B两边都处于成功率floor：37deg在DR completion及lateral有小幅方向性优势、forward
  terminal反而-5pp，不能决定37或47.6。`xxr vs ob`同时变camera、ZOnly/FullXYZ、critic和world size，禁止
  当camera-only解释。
- Offline pre-lift mirror action relative-L2：0mc7.5%、OOS3.7%、SW8.5%、KDW12.2%、tuhu6.2%、11x7.1%、
  xxr8.6%、ob3.8%、Spatial1k7.5%；constant-same-mean普遍更大。闭环OOS/SW baseline均lift4/4，constant后
  均0/4；SW mirror后lift1/4、terminal0/4。结论是depth确实被使用，但数值sensitivity不是representation质量
  排名；SW最依赖正确左右depth，OOS闭环质量最好。SpatialSoftmax仅1k，禁止进入最终质量排名。
- 本地交付=`/home/ubuntu/FAR/_check_vis/08-18-1036__11checkpoints__620rollouts__random_depth_carry_forward_ablation/`
  （指向`/data/holosoma_eval_deliveries/...`），主报告=`EVALUATION_REPORT_ZH.md`，含37条代表视频、3张人工抽帧
  contact sheet、6张图、全部CSV/JSON/config/log/manifest。delivery manifest SHA256=
  `0f906e7d1b6fc15551ba2568520019a26e6746da12f9232f4c7a155a37c48821`；evaluation summary SHA256=
  `0ee2af34e949b123a2cbf86b6c84bcb030818b80e17a524b98305c1db21a87ab`。
- 全部620视频与full policy-I/O保留在
  `10.99.1.21:/data/holosoma_eval_audits/multiaxis_policy_ablation_eval_20260818_064433/`。W&B全程只读；eval只用
  GPU1--7，GPU0既有`4v4jv0xv`未停止、未修改。

## tuhu3ghf/model_35000 四类代表物体 pure-forward rollout（2026-08-18）

- 用户要求录制`zihanw22/carry-any/tuhu3ghf`当时最新checkpoint；请求时run仍为`running`、W&B history
  step约35262，冻结的最新完整atomic pair为`model_35000`（completed iteration34999/next35000），未追逐录制期间
  后续checkpoint。PT/ONNX SHA256=`ebdbd22e413f768a88530f02e89acca6269920c7aa31edafe9168ebae5b7fcbf`/
  `74fa526f6d04d06538b45da350a7a8cd1fe57e2c0d9e21fe3e6dfcf3313ea516`；pair manifest记录PyTorch-vs-ORT
  parity通过，fresh `onnx.checker`与ORT CPU加载也通过，actor input=`94+5046`、action=`29`。
- 固定campaign同一组四类代表物体`box_28/noscale__any_ball_3/scale__any_barrel_84/unscale__any_bin_22`，single env、
  seed42、timestep0、randomization disabled、501 steps@50Hz。actor在object相对初始world-z达到+0.30m前严格
  `[0,0,0,0]`，达到后持续robot-heading-frame`[0.15,0,0,0]`，consecutive=0、保留native pickup、drop全程0、
  无heading lock。4/4均触发，首次forward actor step为`98/90/144/80`；每条501 metrics、501 full policy-I/O、
  唯一501帧H.264及command/drop/action finite均通过，4/4 `any_done=false`。
- model35000末端object相对高度为box/ball/barrel/bin=`0.703/0.516/0.516/0.423m`，object净XY位移=
  `3.261/4.744/4.123/5.292m`，末端robot-object XY距离=`0.329/0.213/0.227/0.232m`。同clip同seed的30K
  对照也一并打包；35K在box/ball/bin上大致相近，barrel净XY由30K的`7.312m`降至35K的`4.123m`，因此不能称
  35K整体优于30K，但四类均完成抬起并保持到末帧。
- 本地交付=`/home/ubuntu/FAR/_check_vis/08-18-1830__tuhu3ghf_model35000__vs_model30000__box_ball_barrel_bin__post_lift_forward015__xy_trajectories/`，
  含35K四宫格master、30K-vs-35K 4x2 master、8条individual trajectory、4条逐物体对照、4条35K raw、manifest
  与人工contact sheet。18/18 MP4均H.264@50fps/501帧，完成ffprobe和全解码；35K master/比较master SHA256=
  `8f050f7a813203e099edc6f0273f23b85cedc40fd0cd2011aaeca3369e7ac3e7`/
  `36c3edf8d3bce579149e659faf2f2ad6d00748239b2745b36c0096f63baf31ea`。full audit保留在
  `10.99.1.21:/data/holosoma_eval_audits/tuhu3ghf_model35000_diverse4_forward015_20260818_183056/`；W&B只读，未修改训练，
  eval使用GPU1--4且结束后均已释放，GPU0原有任务未触碰。
