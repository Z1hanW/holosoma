from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in _TRUE_VALUES:
        return True
    if raw in _FALSE_VALUES:
        return False
    return default


def _split_cuda_visible_devices(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _remap_rank_to_single_visible_gpu() -> None:
    if not _env_flag("HOLOSOMA_RANK_VISIBLE_DEVICES", default=True):
        return

    original_local_rank_raw = os.environ.get("LOCAL_RANK", "0")
    try:
        original_local_rank = int(original_local_rank_raw)
    except ValueError as exc:
        raise SystemExit(f"Invalid LOCAL_RANK for rank-visible launch: {original_local_rank_raw!r}") from exc

    original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if original_cuda_visible and original_cuda_visible.lower() != "all":
        visible_devices = _split_cuda_visible_devices(original_cuda_visible)
        if original_local_rank >= len(visible_devices):
            raise SystemExit(
                "LOCAL_RANK={} is out of range for CUDA_VISIBLE_DEVICES={!r}".format(
                    original_local_rank,
                    original_cuda_visible,
                )
            )
        physical_device = visible_devices[original_local_rank]
    else:
        physical_device = str(original_local_rank)

    os.environ.setdefault("HOLOSOMA_ORIGINAL_LOCAL_RANK", str(original_local_rank))
    os.environ.setdefault("HOLOSOMA_ORIGINAL_LOCAL_WORLD_SIZE", os.environ.get("LOCAL_WORLD_SIZE", ""))
    os.environ.setdefault("HOLOSOMA_ORIGINAL_CUDA_VISIBLE_DEVICES", original_cuda_visible)
    os.environ["HOLOSOMA_RANK_VISIBLE_PHYSICAL_DEVICE"] = physical_device
    os.environ["CUDA_VISIBLE_DEVICES"] = physical_device
    os.environ["LOCAL_RANK"] = "0"
    os.environ["LOCAL_WORLD_SIZE"] = "1"

    if os.environ.get("RANK", "0") == "0":
        print(
            "[INFO] rank-visible GPU remap enabled: "
            f"original CUDA_VISIBLE_DEVICES={original_cuda_visible!r}, "
            f"rank0 physical device={physical_device!r}",
            flush=True,
        )


def main() -> None:
    _remap_rank_to_single_visible_gpu()
    target = Path(__file__).with_name("train_agent.py")
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
