from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=240))
    rank = dist.get_rank()
    world = dist.get_world_size()
    x = torch.tensor([rank + 1.0], device=f"cuda:{local_rank}")
    dist.all_reduce(x)
    expected = world * (world + 1) / 2
    torch.cuda.synchronize()
    if abs(float(x.item()) - expected) > 1e-3:
        raise RuntimeError(f"bad all_reduce result rank={rank} got={x.item()} expected={expected}")
    if rank == 0:
        print(f"NCCL_PROBE_OK world={world} sum={x.item()}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
