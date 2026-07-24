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
