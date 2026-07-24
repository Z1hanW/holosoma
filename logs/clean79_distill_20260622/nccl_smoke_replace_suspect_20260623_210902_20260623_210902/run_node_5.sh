#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/FAR/holosoma
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_SOCKET_IFNAME=enp135s0
export GLOO_SOCKET_IFNAME=enp135s0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export TORCH_DIST_TIMEOUT_SEC=${TORCH_DIST_TIMEOUT_SEC:-120}
export NCCL_LIB_DIR=/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/lib/python3.11/site-packages/nvidia/nccl/lib
export LD_LIBRARY_PATH=/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/lib/python3.11/site-packages/nvidia/nccl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
exec /home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python3 -m torch.distributed.run   --nnodes=8   --node_rank=5   --master_addr=10.99.1.134   --nproc_per_node=8   --max_restarts=0   --master_port=30117   scripts/distributed_barrier_smoke.py
