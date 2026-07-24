#!/usr/bin/env python3
"""NCCL/Gloo canary matching HoloSoma's gradient-reduction protocol.

Legacy HOLOSOMA_NCCL_TEST_* environment variables remain the interface.  In
hierarchical mode every payload follows the training order exactly:

  local NCCL reduce -> leader all-reduce -> local Gloo barrier
  -> local NCCL broadcast

Collective errors are never retried on another backend.  torchrun and the
configured process-group timeout own peer termination.
"""

from __future__ import annotations

import math
import os
import sys
import time
from collections.abc import Mapping
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist


_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _integer(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name, "")
    try:
        return default if raw == "" else int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _boolean(env: Mapping[str, str], name: str, default: bool = False) -> bool:
    raw = env.get(name, "")
    if raw == "":
        return default
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}")


def _parse_config(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    env = os.environ if env is None else env
    world = _integer(env, "WORLD_SIZE", 1)
    rank = _integer(env, "RANK", 0)
    device_rank = _integer(env, "LOCAL_RANK", 0)
    local_world = _integer(
        env,
        "HOLOSOMA_ORIGINAL_LOCAL_WORLD_SIZE",
        _integer(env, "LOCAL_WORLD_SIZE", world),
    )
    local_rank = _integer(env, "HOLOSOMA_ORIGINAL_LOCAL_RANK", device_rank)
    rounds = _integer(env, "HOLOSOMA_NCCL_TEST_ITERS", 200)
    control_ops = tuple(
        item.strip().lower()
        for item in env.get("HOLOSOMA_NCCL_TEST_CONTROL_OPS", "sum,min,max").split(",")
        if item.strip()
    )
    config: dict[str, Any] = {
        "rank": rank,
        "world": world,
        "device_rank": device_rank,
        "local_rank": local_rank,
        "local_world": local_world,
        "nodes": world // local_world if local_world else 0,
        "numel": _integer(env, "HOLOSOMA_NCCL_TEST_NUMEL", 3_076_719),
        "rounds": rounds,
        "ops_per_round": _integer(env, "HOLOSOMA_NCCL_TEST_OPS_PER_ROUND", 1),
        "warmup": _integer(
            env,
            "HOLOSOMA_NCCL_TEST_WARMUP_ITERS",
            min(10, max(0, rounds - 1)),
        ),
        "timeout": _integer(env, "HOLOSOMA_NCCL_TEST_TIMEOUT_SEC", 600),
        "backend": env.get("HOLOSOMA_NCCL_TEST_BACKEND", "nccl").strip().lower(),
        "hierarchical": _boolean(env, "HOLOSOMA_NCCL_TEST_HIERARCHICAL"),
        "cpu_leader": _boolean(env, "HOLOSOMA_NCCL_TEST_CPU_LEADER"),
        "gloo_payload": _boolean(env, "HOLOSOMA_NCCL_TEST_GLOO_PAYLOAD"),
        "control_mode": env.get("HOLOSOMA_NCCL_TEST_CONTROL_MODE", "none").strip().lower(),
        "control_ops": control_ops,
        "control_numel": _integer(env, "HOLOSOMA_NCCL_TEST_CONTROL_NUMEL", 4),
        "control_every": _integer(env, "HOLOSOMA_NCCL_TEST_CONTROL_EVERY", 1),
        "check_every": _integer(env, "HOLOSOMA_NCCL_TEST_CHECK_EVERY", 25),
        "report_every": _integer(env, "HOLOSOMA_NCCL_TEST_REPORT_EVERY", 25),
    }
    positive = (
        "world", "local_world", "numel", "rounds", "ops_per_round", "timeout",
        "control_numel", "control_every", "check_every", "report_every",
    )
    if any(config[name] <= 0 for name in positive):
        raise ValueError(f"Canary values must be positive: {config}")
    if device_rank < 0:
        raise ValueError(f"LOCAL_RANK must be non-negative, got {device_rank}")
    if not 0 <= rank < world or not 0 <= local_rank < local_world:
        raise ValueError(f"Invalid rank topology: rank={rank}/{world}, local_rank={local_rank}/{local_world}")
    if world % local_world or local_rank != rank % local_world:
        raise ValueError("Canary requires contiguous torchrun ranks and WORLD_SIZE divisible by LOCAL_WORLD_SIZE")
    if not 0 <= config["warmup"] < rounds:
        raise ValueError("HOLOSOMA_NCCL_TEST_WARMUP_ITERS must be in [0, ITERS)")
    if config["backend"] not in {"gloo", "nccl"}:
        raise ValueError(f"Unsupported backend {config['backend']!r}")
    if config["control_mode"] not in {"none", "flat", "hierarchical"}:
        raise ValueError(f"Unsupported control mode {config['control_mode']!r}")
    if not control_ops or len(control_ops) != len(set(control_ops)) or set(control_ops) - {"sum", "min", "max"}:
        raise ValueError("HOLOSOMA_NCCL_TEST_CONTROL_OPS must be a unique subset of sum,min,max")
    if config["cpu_leader"] and not config["hierarchical"]:
        raise ValueError("HOLOSOMA_NCCL_TEST_CPU_LEADER requires hierarchical mode")
    if config["gloo_payload"] and config["hierarchical"]:
        raise ValueError("Gloo payload and hierarchical payload are mutually exclusive")
    if config["control_mode"] == "hierarchical" and not config["hierarchical"]:
        raise ValueError("Hierarchical controls require hierarchical payload topology")
    if config["hierarchical"] and (local_world <= 1 or world <= local_world):
        raise ValueError("Hierarchical mode requires multiple ranks per node and multiple nodes")
    expected = {
        "world": _integer(env, "HOLOSOMA_NCCL_TEST_EXPECTED_WORLD_SIZE", 0),
        "local_world": _integer(env, "HOLOSOMA_NCCL_TEST_EXPECTED_LOCAL_WORLD_SIZE", 0),
        "nodes": _integer(env, "HOLOSOMA_NCCL_TEST_EXPECTED_NODE_COUNT", 0),
    }
    for name, value in expected.items():
        if value and value != config[name]:
            raise ValueError(f"Expected {name}={value}, got {config[name]}")
    return config


def _group_plan(config: Mapping[str, Any]) -> list[tuple[str, list[int], str]]:
    if not config["hierarchical"]:
        return []
    plan = []
    for node in range(config["nodes"]):
        ranks = list(range(node * config["local_world"], (node + 1) * config["local_world"]))
        plan += [(f"local_payload_{node}", ranks, "nccl"), (f"local_barrier_{node}", ranks, "gloo")]
    leaders = list(range(0, config["world"], config["local_world"]))
    plan.append(("leader_payload", leaders, "gloo" if config["cpu_leader"] else "nccl"))
    if config["control_mode"] == "hierarchical" and not config["cpu_leader"]:
        plan.append(("leader_control", leaders, "gloo"))
    return plan


def _build_groups(config: Mapping[str, Any]) -> dict[str, Any]:
    timeout = timedelta(seconds=config["timeout"])
    handles = {
        name: dist.new_group(ranks=ranks, backend=backend, timeout=timeout)
        for name, ranks, backend in _group_plan(config)
    }
    node = config["rank"] // config["local_world"]
    result = {
        "local_payload": handles.get(f"local_payload_{node}"),
        "local_barrier": handles.get(f"local_barrier_{node}"),
        "local_leader": node * config["local_world"],
        "leader_payload": handles.get("leader_payload"),
        "leader_control": handles.get("leader_control", handles.get("leader_payload") if config["cpu_leader"] else None),
        "is_leader": config["local_rank"] == 0,
    }
    result["all_gloo"] = dist.new_group(
        ranks=list(range(config["world"])), backend="gloo", timeout=timeout
    )
    return result


def _sync(tensor: torch.Tensor) -> None:
    if tensor.device.type == "cuda":
        torch.cuda.synchronize(tensor.device)


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _payload_reduce(tensor: torch.Tensor, config: Mapping[str, Any], groups: Mapping[str, Any]) -> dict[str, float]:
    total = time.perf_counter()
    if not config["hierarchical"]:
        if config["gloo_payload"]:
            cpu = tensor.detach().cpu()
            dist.all_reduce(cpu, op=dist.ReduceOp.SUM, group=groups["all_gloo"])
            tensor.copy_(cpu.to(tensor.device))
        else:
            _sync(tensor)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        _sync(tensor)
        elapsed = _ms(total)
        return {"payload_ms": elapsed, "flat_all_reduce_ms": elapsed}

    phases: dict[str, float] = {}
    start = time.perf_counter()
    _sync(tensor)
    phases["pack_sync_ms"] = _ms(start)
    start = time.perf_counter()
    dist.reduce(tensor, dst=groups["local_leader"], op=dist.ReduceOp.SUM, group=groups["local_payload"])
    _sync(tensor)
    phases["local_reduce_ms"] = _ms(start)
    if groups["is_leader"]:
        start = time.perf_counter()
        if config["cpu_leader"]:
            cpu = tensor.detach().cpu()
            dist.all_reduce(cpu, op=dist.ReduceOp.SUM, group=groups["leader_payload"])
            tensor.copy_(cpu.to(tensor.device))
        else:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=groups["leader_payload"])
        _sync(tensor)
        phases["leader_all_reduce_ms"] = _ms(start)
    else:
        phases["leader_all_reduce_ms"] = math.nan
    start = time.perf_counter()
    dist.barrier(group=groups["local_barrier"])
    _sync(tensor)
    phases["local_barrier_ms"] = _ms(start)
    start = time.perf_counter()
    dist.broadcast(tensor, src=groups["local_leader"], group=groups["local_payload"])
    _sync(tensor)
    phases["local_broadcast_ms"] = _ms(start)
    phases["payload_ms"] = _ms(total)
    return phases


def _control_reduce(tensor: torch.Tensor, name: str, config: Mapping[str, Any], groups: Mapping[str, Any]) -> None:
    op = {"sum": dist.ReduceOp.SUM, "min": dist.ReduceOp.MIN, "max": dist.ReduceOp.MAX}[name]
    if config["control_mode"] == "flat":
        dist.all_reduce(tensor, op=op, group=groups["all_gloo"])
        return
    dist.reduce(tensor, dst=groups["local_leader"], op=op, group=groups["local_barrier"])
    if groups["is_leader"]:
        dist.all_reduce(tensor, op=op, group=groups["leader_control"])
    dist.broadcast(tensor, src=groups["local_leader"], group=groups["local_barrier"])


def _expected_control(name: str, base: int, world: int) -> int:
    return {
        "sum": world * base + world * (world - 1) // 2,
        "min": base,
        "max": base + world - 1,
    }[name]


def _check_payload(
    tensor: torch.Tensor,
    expected: float,
    control_failed: bool,
    config: Mapping[str, Any],
    groups: Mapping[str, Any],
) -> None:
    finite = bool(torch.isfinite(tensor).all().item())
    error = float(torch.amax(torch.abs(tensor - expected)).item()) if finite else math.inf
    status = torch.tensor([not finite, error > 1e-3, control_failed], dtype=torch.int64)
    dist.all_reduce(status, op=dist.ReduceOp.SUM, group=groups["all_gloo"])
    if bool(status.any().item()):
        raise RuntimeError(
            f"Correctness failure rank={config['rank']} expected={expected} error={error} "
            f"global_nonfinite,payload,control={status.tolist()}"
        )


def _percentiles(values: torch.Tensor) -> dict[str, float | int]:
    values = values[torch.isfinite(values)].double()
    if not values.numel():
        return {"samples": 0}
    q = torch.quantile(values, torch.tensor([0.50, 0.95, 0.99], dtype=torch.float64))
    return {
        "samples": values.numel(), "p50": q[0].item(), "p95": q[1].item(),
        "p99": q[2].item(), "max": values.max().item(),
    }


def _print_latency(rows: list[dict[str, float]], config: Mapping[str, Any], groups: Mapping[str, Any]) -> None:
    names = sorted({name for row in rows for name in row})
    local = torch.tensor([[row.get(name, math.nan) for name in names] for row in rows], dtype=torch.float64)
    gathered = [torch.empty_like(local) for _ in range(config["world"])] if config["rank"] == 0 else None
    dist.gather(local, gather_list=gathered, dst=0, group=groups["all_gloo"])
    if config["rank"]:
        return
    assert gathered is not None
    merged = torch.cat(gathered)
    for index, name in enumerate(names):
        stats = _percentiles(merged[:, index])
        print("Canary latency " + name + " " + " ".join(f"{key}={value:.3f}" if key != "samples" else f"samples={value}" for key, value in stats.items()), flush=True)


def _version() -> str:
    try:
        value = torch.cuda.nccl.version()
        return ".".join(map(str, value)) if isinstance(value, tuple) else str(value)
    except Exception:
        return "unavailable"


def _run(config: Mapping[str, Any]) -> None:
    cuda = config["backend"] == "nccl" or config["hierarchical"]
    if cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("This canary mode requires CUDA")
        torch.cuda.set_device(config["device_rank"])
        device = torch.device("cuda", config["device_rank"])
    else:
        device = torch.device("cpu")
    kwargs = {
        "backend": config["backend"], "rank": config["rank"], "world_size": config["world"],
        "timeout": timedelta(seconds=config["timeout"]),
    }
    if config["backend"] == "nccl":
        kwargs["device_id"] = device
    dist.init_process_group(**kwargs)
    groups = _build_groups(config)
    versions = [None] * config["world"]
    dist.all_gather_object(
        versions,
        (str(torch.__version__), str(torch.version.cuda), _version(), os.environ.get("NCCL_LIB_SHA256", "")),
        group=groups["all_gloo"],
    )
    if config["rank"] == 0:
        leaders = list(range(0, config["world"], config["local_world"]))
        print(
            f"Canary protocol=v2 backend={config['backend']} hierarchical={config['hierarchical']} "
            f"cpu_leader={config['cpu_leader']} world={config['world']} local_world={config['local_world']} "
            f"nodes={config['nodes']} leaders={leaders} numel={config['numel']} "
            f"rounds={config['rounds']} ops_per_round={config['ops_per_round']} warmup={config['warmup']} "
            f"control={config['control_mode']} versions={sorted(set(versions))}", flush=True,
        )

    payload = torch.empty(config["numel"], dtype=torch.float32, device=device)
    control = torch.empty(config["control_numel"], dtype=torch.int64) if config["control_mode"] != "none" else None
    rows: list[dict[str, float]] = []
    control_failed = False
    total_ops = config["rounds"] * config["ops_per_round"]
    for round_index in range(config["rounds"]):
        for inner in range(config["ops_per_round"]):
            op_index = round_index * config["ops_per_round"] + inner
            offset = (op_index % 17) * 0.25
            payload.fill_(config["rank"] + offset)
            total = time.perf_counter()
            timings = _payload_reduce(payload, config, groups)
            if control is not None and (op_index + 1) % config["control_every"] == 0:
                base = (op_index % 17) * config["world"] * 2
                for name in config["control_ops"]:
                    control.fill_(base + config["rank"])
                    start = time.perf_counter()
                    _control_reduce(control, name, config, groups)
                    timings[f"control_{name}_ms"] = _ms(start)
                    control_failed |= not bool(torch.eq(control, _expected_control(name, base, config["world"])).all())
            timings["end_to_end_ms"] = _ms(total)
            if op_index == 0 or (op_index + 1) % config["check_every"] == 0 or op_index + 1 == total_ops:
                expected = config["world"] * (config["world"] - 1) / 2 + config["world"] * offset
                _check_payload(payload, expected, control_failed, config, groups)
                control_failed = False
            if round_index >= config["warmup"]:
                rows.append(timings)
        if config["rank"] == 0 and ((round_index + 1) % config["report_every"] == 0 or round_index + 1 == config["rounds"]):
            print(f"NCCL smoke iter={round_index + 1}/{config['rounds']} last_allreduce_ms={timings['payload_ms']:.3f}", flush=True)

    dist.barrier(group=groups["all_gloo"])
    _print_latency(rows, config, groups)
    if config["rank"] == 0:
        print("Canary correctness=passed finite=passed", flush=True)
        print("NCCL smoke test passed", flush=True)
    dist.destroy_process_group()


def main() -> None:
    config = _parse_config()
    try:
        _run(config)
    except BaseException as exc:
        print(f"NCCL canary failed rank={config['rank']}: {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
