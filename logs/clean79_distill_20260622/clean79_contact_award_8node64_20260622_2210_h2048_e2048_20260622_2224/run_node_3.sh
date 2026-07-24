#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/FAR/holosoma
mkdir -p /home/ubuntu/FAR/holosoma/logs/clean79_distill_20260622/clean79_contact_award_8node64_20260622_2210_h2048_e2048_20260622_2224
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PYTHON_BIN=/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python3
export NPROC=8
export NNODES=8
export NODE_RANK=3
export MASTER_ADDR=10.99.0.141
export MASTER_PORT=29690
export PER_GPU_ENVS=2048
export TOTAL_NUM_ENVS=131072
export AS_SUCCESS133_FINAL0P5=1
export AS_SUCCESS133_BANK_NAME=carryany_filter_scale_noscale_keep169_20260513_plus_box_teacher_rollout_success155_bcleb5oi58000_final0p5_solid80_clean_box_bin_barrel_ball_meshphys_v1
export OMOMO_EXPECTED_TOTAL=79
export RESUME_FROM_BOX=1
export BOX_RESUME_CKPT=/home/ubuntu/.cache/holosoma/checkpoints/as_ch51_convex_pretrain_model_22000.pt
export RESUME_FROM_BOX_CKPT=/home/ubuntu/.cache/holosoma/checkpoints/as_ch51_convex_pretrain_model_22000.pt
export POLICY_INIT_CKPT=/home/ubuntu/.cache/holosoma/checkpoints/as_ch51_convex_pretrain_model_22000.pt
export POLICY_INIT_CHECKPOINT=/home/ubuntu/.cache/holosoma/checkpoints/as_ch51_convex_pretrain_model_22000.pt
export AS_CONTACT_AWARE=1
export ROOT_COMMAND_MODE=contact-aware
export STUDENT_ACTOR_INPUTS="['actor_obs_root_contact_aware','actor_obs_drop_button','actor_obs_proprio_with_actions_no_linvel']"
export STUDENT_ACTOR_HIDDEN_DIMS="[2048,1024,512,256,128]"
export AS_CONTACT_EXPORT_ROOT=/home/ubuntu/FAR/holosoma/data/ds_as_data/carryany_filter_scale_noscale_keep169_20260513_plus_box_teacher_rollout_success155_bcleb5oi58000_final0p5_solid80_clean_box_bin_barrel_ball_meshphys_v1/contact_export_from_teacher_meshphys58000_20260622_043839_4gpu_all79
export ENABLE_OFFLINE_CONTACT_GUIDANCE=True
export RUN_NAME=g1_w_object_distill_as_button_clean79_contact_award_m58000_64gpu_e2048_h2048_1024_512_256_128
export TRAINING_NAME=g1_29dof_wbt_w_object_distill_as_button_clean79_contact_award_m58000_64gpu_e2048_h2048_1024_512_256_128_depth
export TRAINING_PROJECT=carry-any
export WANDB_PROJECT=carry-any
export SCHEDULE_NAME=as_clean79_contact_award_drop_button_ppo_first_m58000_64gpu_e2048_h2048
export SCHEDULE_NOTES=clean79\ contact-award\ all79\ meshphys\ m58000\,\ 8\ nodes\ x\ 8\ GPUs\,\ PER_GPU_ENVS=2048\,\ actor\ hidden\ dims\ \[2048\,1024\,512\,256\,128\]\,\ mesh-surface\ contact\ export.
export NUM_LEARNING_ITERATIONS=40000
export SAVE_INTERVAL=100
export EXPORT_ONNX=False
export WANDB_RESUME_SAME_RUN=0
export OBJECT_GEOMETRY_MODE=mesh
export HOLOSOMA_OBJECT_SPAWN_MODE=single_slot_multi_urdf
export HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK=1
export TORCH_DIST_TIMEOUT_SEC=3600
export MAX_RESTARTS=0
export NCCL_SOCKET_IFNAME=enp135s0
export GLOO_SOCKET_IFNAME=enp135s0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_RETRY_CNT=${NCCL_SOCKET_RETRY_CNT:-120}
export NCCL_SOCKET_RETRY_SLEEP_MSEC=${NCCL_SOCKET_RETRY_SLEEP_MSEC:-1000}
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_LIB_DIR=/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/lib/python3.11/site-packages/nvidia/nccl/lib
export LD_LIBRARY_PATH=/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/lib/python3.11/site-packages/nvidia/nccl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY=301989888
export PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY=436207616
export PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY=134217728
export PHYSX_GPU_COLLISION_STACK_SIZE=268435456
echo "[INFO][10.99.1.60] session=clean79_contact_award_8node64_20260622_2210_h2048_e2048 node_rank=3/8 nproc=8 per_gpu_envs=2048 total_envs=131072"
echo "[INFO][10.99.1.60] master=10.99.0.141:29690"
echo "[INFO][10.99.1.60] bank=/home/ubuntu/FAR/holosoma/data/ds_as_data/carryany_filter_scale_noscale_keep169_20260513_plus_box_teacher_rollout_success155_bcleb5oi58000_final0p5_solid80_clean_box_bin_barrel_ball_meshphys_v1"
echo "[INFO][10.99.1.60] contact_root=/home/ubuntu/FAR/holosoma/data/ds_as_data/carryany_filter_scale_noscale_keep169_20260513_plus_box_teacher_rollout_success155_bcleb5oi58000_final0p5_solid80_clean_box_bin_barrel_ball_meshphys_v1/contact_export_from_teacher_meshphys58000_20260622_043839_4gpu_all79"
echo "[INFO][10.99.1.60] actor_hidden_dims=${STUDENT_ACTOR_HIDDEN_DIMS}"
echo "[INFO][10.99.1.60] nccl_if=${NCCL_SOCKET_IFNAME} gloo_if=${GLOO_SOCKET_IFNAME} nccl_ib_disable=${NCCL_IB_DISABLE}"
echo "[INFO][10.99.1.60] nccl_lib_dir=${NCCL_LIB_DIR}"
exec bash ./distill_as_button.sh contact-aware wandb://zihanw22/carry-any/bcleb5oi/model_58000.pt
