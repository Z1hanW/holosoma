from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

_STATE: "DistributedDebugState | None" = None


class DistributedDebugState:
    def __init__(self, *, rank: int, world_size: int):
        import torch.distributed as dist
        from torch.distributed.distributed_c10d import _get_default_store

        self.rank = rank
        self.world_size = world_size
        self._dist = dist
        self._store = _get_default_store()
        self._heartbeat_dir = _resolve_heartbeat_dir()

    def set(self, key: str, value: dict[str, Any]) -> None:
        payload = json.dumps(value, ensure_ascii=True, sort_keys=True)
        self._store.set(key, payload)

    def get(self, key: str) -> dict[str, Any] | None:
        try:
            payload = self._store.get(key)
        except Exception:
            return None

        if isinstance(payload, bytes):
            payload = payload.decode("utf-8", errors="replace")
        try:
            return json.loads(payload)
        except Exception:
            return {"raw": str(payload)}

    def write_heartbeat(self, payload: dict[str, Any]) -> None:
        self._heartbeat_dir = _resolve_heartbeat_dir()
        if self._heartbeat_dir is None:
            return
        self._heartbeat_dir.mkdir(parents=True, exist_ok=True)
        target = self._heartbeat_dir / f"rank_{self.rank:02d}.json"
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2), encoding="utf-8")
        tmp.replace(target)


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def _resolve_heartbeat_dir() -> Path | None:
    base = os.environ.get("HOLOSOMA_DISTRIBUTED_DEBUG_DIR", "").strip()
    if not base:
        return None
    return Path(base)


def init_distributed_debug_state(rank: int, world_size: int) -> None:
    global _STATE
    _STATE = DistributedDebugState(rank=rank, world_size=world_size)
    mark_distributed_stage("process_group_initialized")


def clear_distributed_debug_state() -> None:
    global _STATE
    _STATE = None


def mark_distributed_stage(stage: str, **extra: Any) -> None:
    if _STATE is None:
        return

    payload: dict[str, Any] = {
        "kind": "stage",
        "rank": _STATE.rank,
        "pid": os.getpid(),
        "stage": stage,
        "time_utc": _utc_now(),
    }
    if extra:
        payload["extra"] = extra
    _STATE.set(f"holosoma_debug_rank_{_STATE.rank}", payload)
    _STATE.write_heartbeat(payload)
    logger.debug("Distributed stage: {}", payload)


def mark_distributed_exception(exc: BaseException, traceback_text: str, *, stage: str | None = None) -> None:
    if _STATE is None:
        return

    payload: dict[str, Any] = {
        "kind": "exception",
        "rank": _STATE.rank,
        "pid": os.getpid(),
        "time_utc": _utc_now(),
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback_text,
    }
    if stage:
        payload["stage"] = stage
    _STATE.set(f"holosoma_debug_rank_{_STATE.rank}", payload)
    _STATE.write_heartbeat(payload)
    logger.error("Distributed exception marker: {}", payload)


def dump_distributed_debug_state(header: str = "Distributed debug snapshot") -> None:
    if _STATE is None:
        return

    lines = [header]
    for rank in range(_STATE.world_size):
        snapshot = _STATE.get(f"holosoma_debug_rank_{rank}")
        if snapshot is None:
            lines.append(f"rank {rank}: <missing>")
            continue

        kind = snapshot.get("kind", "unknown")
        stage = snapshot.get("stage", "<unknown>")
        timestamp = snapshot.get("time_utc", "<unknown>")
        pid = snapshot.get("pid", "<unknown>")
        msg = f"rank {rank}: kind={kind} stage={stage} pid={pid} time={timestamp}"
        if kind == "exception":
            exc_type = snapshot.get("exception_type", "Exception")
            exc_msg = snapshot.get("message", "")
            msg += f" exception={exc_type}: {exc_msg}"
        elif snapshot.get("extra") is not None:
            msg += f" extra={snapshot['extra']}"
        lines.append(msg)

    logger.error("\n".join(lines))
