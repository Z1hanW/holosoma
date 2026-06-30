#!/usr/bin/env python3
"""Small multi-node NCCL all-reduce smoke test."""

from __future__ import annotations

import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or value == "" else int(value)


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")


def _build_hierarchical_groups(rank: int, world_size: int, local_world_size: int):
    if local_world_size <= 1 or world_size <= local_world_size or world_size % local_world_size != 0:
        raise ValueError(
            f"hierarchical test requires world_size divisible by LOCAL_WORLD_SIZE; "
            f"got world_size={world_size}, local_world_size={local_world_size}"
        )
    node_count = world_size // local_world_size
    local_node_idx = rank // local_world_size
    local_leader_rank = local_node_idx * local_world_size
    local_ranks = list(range(local_leader_rank, local_leader_rank + local_world_size))
    local_group = dist.new_group(ranks=local_ranks, use_local_synchronization=True)
    local_barrier_group = dist.new_group(ranks=local_ranks, backend="gloo", use_local_synchronization=True)
    leader_ranks = list(range(0, world_size, local_world_size))
    is_leader_rank = rank in leader_ranks
    if is_leader_rank:
        leader_group = dist.new_group(ranks=leader_ranks, use_local_synchronization=True)
        leader_gloo_group = dist.new_group(ranks=leader_ranks, backend="gloo", use_local_synchronization=True)
    else:
        leader_group = None
        leader_gloo_group = None
    return local_group, local_barrier_group, local_leader_rank, leader_group, leader_gloo_group, is_leader_rank


def _hierarchical_all_reduce(
    tensor: torch.Tensor,
    *,
    local_group,
    local_barrier_group,
    local_leader_rank: int,
    leader_group,
    leader_gloo_group,
    is_leader_rank: bool,
    cpu_leader: bool,
) -> None:
    dist.reduce(tensor, dst=local_leader_rank, op=dist.ReduceOp.SUM, group=local_group)
    if is_leader_rank:
        if cpu_leader:
            cpu_tensor = tensor.detach().cpu()
            dist.all_reduce(cpu_tensor, op=dist.ReduceOp.SUM, group=leader_gloo_group)
            tensor.copy_(cpu_tensor.to(device=tensor.device, dtype=tensor.dtype))
        else:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=leader_group)
    dist.barrier(group=local_barrier_group)
    dist.broadcast(tensor, src=local_leader_rank, group=local_group)


def main() -> None:
    rank = _env_int("RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)
    local_rank = _env_int("LOCAL_RANK", 0)
    local_world_size = _env_int("LOCAL_WORLD_SIZE", world_size)
    numel = _env_int("HOLOSOMA_NCCL_TEST_NUMEL", 3_076_719)
    iterations = _env_int("HOLOSOMA_NCCL_TEST_ITERS", 200)
    timeout_sec = _env_int("HOLOSOMA_NCCL_TEST_TIMEOUT_SEC", 600)
    backend = os.environ.get("HOLOSOMA_NCCL_TEST_BACKEND", "nccl").strip().lower()
    hierarchical = _env_bool("HOLOSOMA_NCCL_TEST_HIERARCHICAL")
    cpu_leader = _env_bool("HOLOSOMA_NCCL_TEST_CPU_LEADER")
    gloo_payload = _env_bool("HOLOSOMA_NCCL_TEST_GLOO_PAYLOAD")

    if backend == "nccl":
        torch.cuda.set_device(local_rank)
        device_id = torch.device(f"cuda:{local_rank}")
        device = torch.device("cuda", local_rank)
    elif backend == "gloo":
        device_id = None
        device = torch.device("cpu")
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    init_kwargs = {
        "backend": backend,
        "rank": rank,
        "world_size": world_size,
        "timeout": timedelta(seconds=timeout_sec),
    }
    if device_id is not None:
        init_kwargs["device_id"] = device_id
    dist.init_process_group(
        **init_kwargs,
    )
    tensor = torch.empty(numel, device=device, dtype=torch.float32)
    expected = world_size * (world_size - 1) / 2.0
    if gloo_payload:
        gloo_payload_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
    else:
        gloo_payload_group = None
    if hierarchical:
        (
            local_group,
            local_barrier_group,
            local_leader_rank,
            leader_group,
            leader_gloo_group,
            is_leader_rank,
        ) = _build_hierarchical_groups(rank, world_size, local_world_size)
    else:
        local_group = None
        local_barrier_group = None
        local_leader_rank = 0
        leader_group = None
        leader_gloo_group = None
        is_leader_rank = False

    if backend == "nccl":
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()
    start = time.perf_counter()
    for idx in range(iterations):
        tensor.fill_(float(rank))
        op_start = time.perf_counter()
        if gloo_payload:
            cpu_tensor = tensor.detach().cpu()
            dist.all_reduce(cpu_tensor, op=dist.ReduceOp.SUM, group=gloo_payload_group)
            tensor.copy_(cpu_tensor.to(device=tensor.device, dtype=tensor.dtype))
        elif hierarchical:
            _hierarchical_all_reduce(
                tensor,
                local_group=local_group,
                local_barrier_group=local_barrier_group,
                local_leader_rank=local_leader_rank,
                leader_group=leader_group,
                leader_gloo_group=leader_gloo_group,
                is_leader_rank=is_leader_rank,
                cpu_leader=cpu_leader,
            )
        else:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if backend == "nccl":
            torch.cuda.synchronize(device)
        if idx == 0 or (idx + 1) % 25 == 0:
            value = float(tensor[0].item())
            if abs(value - expected) > 1e-3:
                raise RuntimeError(f"rank {rank} iter {idx}: got {value}, expected {expected}")
            if rank == 0:
                elapsed_ms = (time.perf_counter() - op_start) * 1000.0
                total_s = time.perf_counter() - start
                print(
                    f"NCCL smoke iter={idx + 1}/{iterations} hierarchical={hierarchical} "
                    f"gloo_payload={gloo_payload} numel={numel} cpu_leader={cpu_leader} "
                    f"last_allreduce_ms={elapsed_ms:.3f} elapsed_s={total_s:.3f}",
                    flush=True,
                )

    if backend == "nccl":
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()
    if rank == 0:
        print("NCCL smoke test passed", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
