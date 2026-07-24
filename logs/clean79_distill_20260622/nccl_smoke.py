import os
import socket

import torch
import torch.distributed as dist


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    x = torch.tensor([float(rank + 1)], device="cuda")
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    expected = world_size * (world_size + 1) / 2
    if rank == 0:
        print(
            f"NCCL_SMOKE_OK host={socket.gethostname()} world={world_size} "
            f"value={float(x.item())} expected={expected}",
            flush=True,
        )
    if abs(float(x.item()) - expected) > 0.5:
        raise RuntimeError(f"Unexpected all_reduce result: {float(x.item())} != {expected}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
