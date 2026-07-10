# HOI Tracking Playbook

这份文档记录我们当前 HOI tracking 应该怎么跑、怎么检查、怎么看 W&B，以及后续如果改 per-joint token 要改什么。默认以后所有相关实验都放在 W&B project `holosomatest`，不要再放到 `carry-any`。

## 当前默认方案

- 任务：object interaction / HOI tracking expert，然后用 expert 做 depth student distillation。
- Expert 架构：`TokenHSI` transformer，不是 MLP。
- 当前 token 方式：modality-level token，不是 per-joint token。
- 默认数据：OMOMO carry data，约 62 条 clip。
- 默认训练规模：`4096 env / GPU`。
- 推荐 W&B project：`zihanw22/holosomatest`。

当前主要 expert run：

- W&B: `https://wandb.ai/zihanw22/holosomatest/runs/rrfbwfnq`
- run name: `omomo62_object_tokenhsi_nexttarget_2x8_16gpu_4096envpergpu_65536env_40000iter_20260706_225349`
- group: `omomo62-object-tokenhsi-nexttarget-4096envpergpu-2x8-20260706_225349`
- exp: `g1-29dof-wbt-w-object-height-scan-tokenhsi-next-target`
- session: `object_tokenhsi_nexttarget_omomo62_2x8_4096envpergpu_20260706_225349`
- nodes: `10.0.90.122`, `10.0.123.134`
- scale: 2 nodes x 8 GPUs = 16 GPUs, `4096 env/GPU`, total `65536 env`
- iterations: `40000`
- data: `/nfs/zzzihanw/amass/converted_res/object_interaction/omomo_carry`
- object map: `/nfs/zzzihanw/amass/converted_res/object_interaction/omomo_carry/_clip_object_urdf_map.json`
- object URDF: `holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf`

## Data Contract

训练 HOI expert 时要用带 object 的 motion data。当前 canonical data 就是 `/nfs/zzzihanw` 里的这份 OMOMO largebox carry set：

```bash
/nfs/zzzihanw/amass/converted_res/object_interaction/omomo_carry
```

这是当前成功 HOI tracking run 使用的数据，不要用本地 `data/` 里的 debug bank 替代，除非只是做 smoke test。

这个目录应该满足：

- 目录下有 62 个 `.npz` motion clips。
- 有 `_clip_object_urdf_map.json`。
- map schema 是 top-level `clips` dict，每个 key 是 clip stem。
- 当前 object 是 `largebox`。
- URDF 指向 `holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf`。
- 例子：`sub10_largebox_032_mj_w_obj.npz` 对应 map key `sub10_largebox_032_mj_w_obj`。

启动前先检查：

```bash
find /nfs/zzzihanw/amass/converted_res/object_interaction/omomo_carry -maxdepth 1 -name '*.npz' | wc -l
test -f /nfs/zzzihanw/amass/converted_res/object_interaction/omomo_carry/_clip_object_urdf_map.json
test -f holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf
```

多机训练前，每台 node 都要能看到同一个 NFS data path 和同一个 repo 里的 URDF。否则会出现只加载 body motion、没有 object mesh / object target 的问题。

## Expert Training

启动脚本：

```bash
./object_tokenhsi_multinode.sh
```

当前推荐启动方式是两台机器一起跑同一组实验，不是每台机器分开跑独立实验：

```bash
TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)

NODE_HOSTS="10.0.90.122 10.0.123.134" \
NNODES=2 \
GPUS_PER_NODE=8 \
ENVS_PER_GPU=4096 \
EXP_NAME=g1-29dof-wbt-w-object-height-scan-tokenhsi-next-target \
MOTION_DIR=/nfs/zzzihanw/amass/converted_res/object_interaction/omomo_carry \
OBJECT_URDF_MAP=/nfs/zzzihanw/amass/converted_res/object_interaction/omomo_carry/_clip_object_urdf_map.json \
WANDB_PROJECT=holosomatest \
WANDB_ENTITY=zihanw22 \
RUN_NAME="omomo62_object_tokenhsi_nexttarget_2x8_16gpu_4096envpergpu_65536env_40000iter_${TIMESTAMP}" \
LOGGER_GROUP="omomo62-object-tokenhsi-nexttarget-4096envpergpu-2x8-${TIMESTAMP}" \
SESSION="object_tokenhsi_nexttarget_omomo62_2x8_4096envpergpu_${TIMESTAMP}" \
REMOTE_REPO=/home/ubuntu/FAR/holosoma_object_tokenhsi_20260704_211351 \
SYNC_SCRIPT=1 \
KILL_EXISTING=0 \
./object_tokenhsi_multinode.sh
```

如果要 4 台机器一起跑同一组实验：

```bash
TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)

NODE_HOSTS="10.0.100.200 10.0.72.226 10.0.90.122 10.0.123.134" \
NNODES=4 \
GPUS_PER_NODE=8 \
ENVS_PER_GPU=4096 \
EXP_NAME=g1-29dof-wbt-w-object-height-scan-tokenhsi-next-target \
MOTION_DIR=/nfs/zzzihanw/amass/converted_res/object_interaction/omomo_carry \
OBJECT_URDF_MAP=/nfs/zzzihanw/amass/converted_res/object_interaction/omomo_carry/_clip_object_urdf_map.json \
WANDB_PROJECT=holosomatest \
WANDB_ENTITY=zihanw22 \
RUN_NAME="omomo62_object_tokenhsi_nexttarget_4x8_32gpu_4096envpergpu_131072env_40000iter_${TIMESTAMP}" \
LOGGER_GROUP="omomo62-object-tokenhsi-nexttarget-4096envpergpu-4x8-${TIMESTAMP}" \
SESSION="object_tokenhsi_nexttarget_omomo62_4x8_4096envpergpu_${TIMESTAMP}" \
REMOTE_REPO=/home/ubuntu/FAR/holosoma_object_tokenhsi_20260704_211351 \
SYNC_SCRIPT=1 \
KILL_EXISTING=0 \
./object_tokenhsi_multinode.sh
```

注意：4 nodes x 8 GPUs x 4096 env/GPU = `131072 env`，吞吐高但更容易遇到显存、network 或 node stability 问题。确认所有 node 都可达之后再开。

## W&B 规则

以后 HOI tracking 相关实验统一放：

```bash
WANDB_ENTITY=zihanw22
WANDB_PROJECT=holosomatest
```

不要放到：

```bash
carry-any
```

本地启动后，脚本会记录 run metadata：

```bash
cat logs/run_commands/${SESSION}.run_name
cat logs/run_commands/${SESSION}.group
cat logs/run_commands/${SESSION}.nodes
cat logs/run_commands/${SESSION}.remote_repo
```

如果忘了 W&B run 在哪里，先看 `${SESSION}.run_name`，再去 `https://wandb.ai/zihanw22/holosomatest` 搜 run name。

## Log Checks

启动后先看 tmux 和 process：

```bash
for h in 10.0.90.122 10.0.123.134; do
  ssh "$h" "tmux ls | grep ${SESSION} || true"
  ssh "$h" "pgrep -af 'train_agent|torchrun' || true"
  ssh "$h" "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits"
done
```

看 remote logs：

```bash
REMOTE_REPO=$(cat logs/run_commands/${SESSION}.remote_repo)

for h in $(cat logs/run_commands/${SESSION}.nodes); do
  ssh "$h" "tail -80 ${REMOTE_REPO}/logs/run_commands/${SESSION}_node*_${h//./-}.log"
done
```

必须确认 log 里有这些信号：

- `Loaded object URDF`
- `MultiMotionLoader: found 62 .npz files`
- `Loaded clip-object metadata map ... (62 entries)`
- `MultiMotionLoader: 62 motions, 21134 total frames`
- `Fixed env-to-clip assignment`
- 正常的 iteration / FPS / reward logging
- W&B URL 或 W&B run id

如果没有 `Loaded object URDF` 或没有 object map，就不要继续等，先修 data path / object map / remote repo sync。

## Policy Inputs

当前 expert 不是 MLP，是 `TokenHSI` transformer。相关代码：

- `src/holosoma/holosoma/agents/modules/modules.py`
- `src/holosoma/holosoma/config_values/wbt/g1/experiment.py`
- `src/holosoma/holosoma/config_values/wbt/g1/observation.py`

当前 `g1-29dof-wbt-w-object-height-scan-tokenhsi-next-target` 的 actor 输入：

```python
["actor_obs", "actor_height_scan", "actor_object_obs"]
```

critic 输入：

```python
["critic_obs", "critic_height_scan", "critic_object_obs"]
```

TokenHSI 当前是 modality token：

- output token
- robot/proprio token: `actor_obs` / `critic_obs`
- height scan token: `actor_height_scan` / `critic_height_scan`
- object token: `actor_object_obs` / `critic_object_obs`

当前不是 per-joint token。29 个 joints 仍然主要在 flat robot/proprio observation 里。

## Object Observations

当前 actor 能看到 object 当前状态、当前 target，以及下一帧 target。关键 observation group 是：

- `actor_object_obs_with_next_target`
- `critic_object_obs_with_next_target`

包含：

- current object: `obj_pos_b`, `obj_ori_b`, `obj_lin_vel_b`
- current target: `obj_ref_pos_b`, `obj_ref_ori_b`, `obj_ref_lin_vel_b`
- next target: `obj_ref_pos_next_b`, `obj_ref_ori_next_b`, `obj_ref_lin_vel_next_b`

所以答案是：policy 可以看到当前 frame 的 target，也可以看到下一 frame 的 object target；但目前不是任意 future horizon，只是 next target。

## Rewards And Termination

当前 reward 里有明确 object tracking 项，不只是 body tracking：

- `Episode/rew_object_global_ref_position_error_exp`
- `Episode/rew_object_global_ref_orientation_error_exp`

实现位置：

- `src/holosoma/holosoma/managers/reward/terms/wbt.py`

对应函数：

- `object_global_ref_position_error_exp`
- `object_global_ref_orientation_error_exp`

termination 也包含 object tracking 失败条件：

- `src/holosoma/holosoma/managers/termination/terms/wbt.py`

重点检查：

- object position reward 是否正向增长或保持稳定。
- object orientation reward 是否有信号。
- bad object tracking termination 是否过高。
- 如果 body reward 好但 object reward 不动，通常说明 object target 没进 obs、object map 没加载、URDF 没加载、或者 reward config 用错了 exp。

## From Default Pose

如果训练看起来像从 default pose 开始，要先区分两件事：

- reset 初始状态是否来自 motion reference。
- policy rollout 前几帧是否因为 bad tracking / reset 看起来退回 default pose。

检查方向：

- log 里是否成功加载 motion clips。
- env-to-clip assignment 是否固定并覆盖所有 clips。
- rollout visualizer 里 reference body 和 object reference 是否在动。
- termination 是否一开始就频繁触发。

如果 motion loaded 正常，但 agent 仍像 default pose，多半是 policy 还没学会或 reset/initialization 配置需要进一步查。

## Failure Modes

常见失败和处理：

- `SIGHUP` / `SIGTERM`：通常是外部 session、node、launcher 或 infra kill。若发生在第一次 save 之前，不会有可用 checkpoint。
- `No route to host`：node 不在线、私网路由断了、跨 region 不能走 10.x private IP。
- `NNODES does not match NODE_HOSTS count`：launcher 没把完整 `NODE_HOSTS` 传到 remote。当前脚本已修过 remote env，要用最新脚本重跑。
- 找不到 object map：`MOTION_DIR` 不对，或者 remote node 看不到 NFS。
- 找不到 object URDF：remote repo 没同步，或者 URDF path 相对 repo 不存在。
- W&B project 错了：确认 `WANDB_PROJECT=holosomatest`。
- 没有 object reward：确认 exp 是 `g1-29dof-wbt-w-object-height-scan-tokenhsi-next-target`，不是没有 object/next-target 的旧 exp。

## Depth Student Distillation

Expert 稳定后再做 depth distill。当前主要 teacher 应该从 `holosomatest/rrfbwfnq` 选 checkpoint，例如：

```bash
wandb://zihanw22/holosomatest/rrfbwfnq/model_08000.pt
```

单机调试启动：

```bash
TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)

TEACHER_CHECKPOINT=wandb://zihanw22/holosomatest/rrfbwfnq/model_08000.pt \
WANDB_PROJECT=holosomatest \
WANDB_ENTITY=zihanw22 \
DISTILL_TAG=hoi_omomo62_rrfbwfnq_model08000_objdepth \
NUM_GPUS=1 \
ENVS_PER_GPU=1024 \
NUM_ITERATIONS=20000 \
SAVE_INTERVAL=1000 \
RUN_NAME="hoi_depth_student_rrfbwfnq_model08000_objdepth_1gpu_${TIMESTAMP}" \
SESSION="hoi_depth_student_rrfbwfnq_model08000_objdepth_1gpu_${TIMESTAMP}" \
./csp_depth_distill.sh
```

多机 distill 启动：

```bash
TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)

NODE_HOSTS="10.0.100.200 10.0.72.226" \
NNODES=2 \
GPUS_PER_NODE=8 \
ENVS_PER_GPU=4096 \
TEACHER_CHECKPOINT=wandb://zihanw22/holosomatest/rrfbwfnq/model_08000.pt \
WANDB_PROJECT=holosomatest \
WANDB_ENTITY=zihanw22 \
DISTILL_TAG=hoi_omomo62_rrfbwfnq_model08000_objdepth \
RUN_NAME="hoi_depth_student_rrfbwfnq_model08000_objdepth_2x8_4096env_${TIMESTAMP}" \
SESSION="hoi_depth_student_rrfbwfnq_model08000_objdepth_2x8_${TIMESTAMP}" \
SYNC_REPO=0 \
./csp_multinode_depth_distill.sh
```

注意：`csp_multinode_depth_distill.sh` 如果 `SYNC_REPO=1`，通常会要求 clean git tree。当前 repo 经常有未提交改动，所以要么先整理 commit，要么明确 `SYNC_REPO=0` 并确认 remote repo 已经是正确代码。

当前 distill checkpoint 可以 save student 和 optimizer，但代码里还没有稳定的 `--resume-student-checkpoint` 入口。如果 distill 在 save interval 前死掉，就没有可恢复 student；如果有 `student_*.pt`，后续要补 resume loader 才能严格接着训练。

## Depth Student Eval

可视化 depth student：

```bash
source scripts/source_inference_setup.sh

python scripts/viser_depth_student_physics_eval.py \
  --checkpoint logs/.../student_0005000.pt \
  --port 2106 \
  --env-id 0 \
  --num-envs 1 \
  --gui-command \
  --depth-hits \
  --no-red-points \
  --no-motion-ref \
  --disable-randomization \
  --log-every 100
```

已有参考 command log：

- `logs/run_commands/holosoma_depth_student_vj7urlp6_student5000_joystick_20260707_163708.cmd`
- `logs/run_commands/holosoma_depth_student_vj7urlp6_student5000_guicmd_20260707_164152.cmd`

## Machines And Connectivity

当前本机：

```bash
10.0.73.59
```

之前同 region 私网节点曾用过：

- `10.0.100.200`
- `10.0.72.226`
- `10.0.90.122`
- `10.0.123.134`
- `10.0.74.86`

如果这些节点 `No route to host`，不要直接假设训练还在。先用 SSH / tmux / process / W&B 同时确认。

跨 region 的机器通常不能直接用 10.x private IP 做同一个 torch distributed job，除非 VPC routing / peering 已经配好。比如 `zzzihanw-112`：

- AWS name: `sky-zzzihanw-112-1452fa42-head`
- region: `eu-north-1`
- private IP: `10.99.1.118`
- public IP: `16.171.139.176`
- public DNS: `ec2-16-171-139-176.eu-north-1.compute.amazonaws.com`

这个机器可能 ping 不通，因为 ICMP 被挡；但 SSH public IP 可以通才有意义：

```bash
ssh ubuntu@16.171.139.176
```

查所有 running `zzzihanw` nodes：

```bash
regions=$(aws ec2 describe-regions --query 'Regions[].RegionName' --output text)

for r in $regions; do
  aws ec2 describe-instances --region "$r" \
    --filters 'Name=instance-state-name,Values=running' 'Name=tag:Name,Values=*zzzihanw*' \
    --query 'Reservations[].Instances[].{Name:Tags[?Key==`Name`]|[0].Value,PrivateIp:PrivateIpAddress,PublicIp:PublicIpAddress,Type:InstanceType}' \
    --output table
done
```

## Comparing To Old Carry-Any Runs

旧的 `carry-any` run，例如：

```bash
https://wandb.ai/zihanw22/carry-any/runs/u8udzw0u
```

和当前 OMOMO 62-clips / object TokenHSI / next-target setup 不能直接只看最终曲线比较。主要差异可能来自：

- 数据不同：旧 run 是 39 条 data，当前是 OMOMO carry 约 62 条。
- W&B project 和 run config 可能不同。
- 当前用 object mesh、object target、next target；旧 run 是否完全一致需要核对 config。
- 当前总 env 数更大，batch/optimization dynamics 不一定等价。
- termination 和 reset 分布变了，可能让早期 learning curve 看起来更差。
- 如果旧 run object reward 高很多，要先核对它是否真的加载 object mesh/object target，而不是只看 body tracking。

对比时优先比这些字段：

- exp name
- motion data path
- number of clips / frames
- object map path
- object URDF load log
- observation groups
- reward terms
- termination thresholds
- total envs, minibatches, learning rate, save interval

## Future: Per-Joint Token

当前 TokenHSI 是 modality token。以后如果要改成 per-joint token，主要变化是：

- 把 29 个 joints 的 `dof_pos`, `dof_vel`, previous action 等拆成 29 个 joint tokens。
- 额外保留 root/base token。
- 保留 object token，包含 object current / target / next target。
- 保留 terrain/depth token。
- 加 joint id / joint type / left-right / kinematic parent 等 embedding。
- action head 可以先保持 pooled latent -> 29 dof action，之后再改成 per-joint action decoder。

推荐迭代顺序：

1. 先做最小 per-joint actor：PPO storage 仍用 flat obs，只在 module 里按 index split 成 29 个 tokens。
2. 再做 per-joint action decoder：每个 joint token 直接预测对应 action。
3. 再加入 kinematic embedding 或 graph bias。

预期收益：

- 更容易让 policy 学 joint-object attention。
- 对手、臂、躯干和 object tracking 的角色分配更明确。
- 对复杂 HOI 可能比单个 flat robot token 更稳。

代价：

- token 数显著增加，训练更慢。
- checkpoint 不兼容当前 TokenHSI。
- observation packing 和 action decoding 更容易出错。
- 需要更仔细地做 ablation，确认提升来自 tokenization，而不是 batch size / target obs / reward config。

## Run Ledger

重要 run 记录：

- `rrfbwfnq`: 当前主要 OMOMO62 object TokenHSI next-target expert, project `holosomatest`。
- `ke1fhl6w`: 早期 4-node OMOMO run，曾在 iter 464 左右被外部 SIGHUP/SIGTERM 停止；由于 save interval 1000，基本没有可用 checkpoint。
- `vj7urlp6`: depth student multi-node run，project `holosomatest`，last seen around step 7025，W&B state crashed；本地有 `student_0005000.pt`。
- `vw34tvkv`: temporary local object-depth distill，1 GPU，teacher `rrfbwfnq/model_08000.pt`，在 iter 975 左右 SIGHUP，未到 save interval。

后续新 run 要记录：

- W&B URL
- run name
- group
- session
- nodes
- envs per GPU
- total envs
- data path
- object map
- teacher checkpoint
- code commit 或 remote repo path
