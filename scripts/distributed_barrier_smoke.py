from __future__ import annotations

import inspect
import os
from datetime import timedelta

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    kwargs = {
        "backend": "nccl",
        "rank": global_rank,
        "world_size": world_size,
        "timeout": timedelta(seconds=int(os.environ.get("TORCH_DIST_TIMEOUT_SEC", "300"))),
    }
    if "device_id" in inspect.signature(dist.init_process_group).parameters:
        kwargs["device_id"] = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(**kwargs)

    value = torch.tensor([float(global_rank + 1)], device=f"cuda:{local_rank}")
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    dist.barrier(device_ids=[local_rank])
    if global_rank == 0:
        expected = world_size * (world_size + 1) / 2.0
        print(f"[SMOKE] world_size={world_size} all_reduce={value.item()} expected={expected}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
