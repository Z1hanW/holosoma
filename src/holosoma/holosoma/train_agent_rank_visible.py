from __future__ import annotations

import os
import re
import runpy
import subprocess
import sys
from pathlib import Path


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}
_RANK_LOCAL_CPU_AFFINITY_ENV = "HOLOSOMA_RANK_LOCAL_CPU_AFFINITY"
_PCI_BUS_ID_RE = re.compile(
    r"^(?P<domain>[0-9a-fA-F]{4}|[0-9a-fA-F]{8}):"
    r"(?P<bus>[0-9a-fA-F]{2}):(?P<device>[0-9a-fA-F]{2})\."
    r"(?P<function>[0-7])$"
)


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in _TRUE_VALUES:
        return True
    if raw in _FALSE_VALUES:
        return False
    raise SystemExit(
        f"{name} must be a boolean (0/1/false/true/no/yes/off/on), got {os.environ.get(name)!r}"
    )


def _split_cuda_visible_devices(raw: str) -> list[str]:
    parts = [part.strip() for part in raw.split(",")]
    if not parts or any(not part or part == "-1" for part in parts):
        raise SystemExit(
            "CUDA_VISIBLE_DEVICES must be a non-empty comma-separated GPU list "
            f"for rank-visible launch, got {raw!r}"
        )
    if len(set(parts)) != len(parts):
        raise SystemExit(
            f"CUDA_VISIBLE_DEVICES contains duplicate GPU tokens for rank-visible launch: {raw!r}"
        )
    return parts


def _affinity_warning(message: object) -> None:
    print(
        f"[WARN] rank-local CPU affinity was not applied (fail-open): {message}",
        file=sys.stderr,
        flush=True,
    )


def _parse_cpu_list(raw: str) -> set[int]:
    cpus: set[int] = set()
    for item in raw.strip().split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            first_raw, last_raw = item.split("-", 1)
            first = int(first_raw)
            last = int(last_raw)
            if first < 0 or last < first:
                raise ValueError(f"invalid CPU range {item!r}")
            cpus.update(range(first, last + 1))
        else:
            cpu = int(item)
            if cpu < 0:
                raise ValueError(f"invalid CPU id {item!r}")
            cpus.add(cpu)
    if not cpus:
        raise ValueError(f"empty CPU list {raw!r}")
    return cpus


def _format_cpu_list(cpus: set[int]) -> str:
    ordered = sorted(cpus)
    if not ordered:
        return ""
    ranges: list[str] = []
    first = previous = ordered[0]
    for cpu in ordered[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(str(first) if first == previous else f"{first}-{previous}")
        first = previous = cpu
    ranges.append(str(first) if first == previous else f"{first}-{previous}")
    return ",".join(ranges)


def _normalize_pci_bus_id(raw: str) -> str:
    match = _PCI_BUS_ID_RE.fullmatch(raw.strip())
    if match is None:
        raise ValueError(f"invalid NVIDIA PCI bus id {raw!r}")
    domain = int(match.group("domain"), 16)
    if domain > 0xFFFF:
        raise ValueError(f"PCI domain is outside Linux sysfs range: {raw!r}")
    return (
        f"{domain:04x}:{match.group('bus').lower()}:"
        f"{match.group('device').lower()}.{match.group('function')}"
    )


def _query_gpu_inventory() -> list[tuple[str, str, str]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise RuntimeError(f"nvidia-smi topology query failed: {detail}")

    inventory: list[tuple[str, str, str]] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3 or not all(fields):
            raise RuntimeError(f"malformed nvidia-smi topology row: {line!r}")
        index, uuid, pci_bus_id = fields
        inventory.append((index, uuid, _normalize_pci_bus_id(pci_bus_id)))
    if not inventory:
        raise RuntimeError("nvidia-smi returned no GPU topology rows")
    return inventory


def _gpu_inventory_row(
    token: str,
    inventory: list[tuple[str, str, str]],
) -> tuple[str, str, str]:
    exact = [row for row in inventory if token in row[:2]]
    if len(exact) == 1:
        return exact[0]
    # CUDA accepts an unambiguous UUID prefix in CUDA_VISIBLE_DEVICES.  Do not
    # guess if a prefix matches more than one physical GPU.
    prefix = [row for row in inventory if row[1].startswith(token)]
    if len(prefix) == 1:
        return prefix[0]
    raise RuntimeError(
        f"CUDA_VISIBLE_DEVICES token {token!r} did not resolve uniquely in nvidia-smi"
    )


def _numa_node_for_pci_device(sysfs_root: Path, pci_bus_id: str) -> int:
    numa_path = sysfs_root / "bus" / "pci" / "devices" / pci_bus_id / "numa_node"
    numa_node = int(numa_path.read_text(encoding="utf-8").strip())
    if numa_node < 0:
        raise RuntimeError(f"kernel reports no NUMA node for PCI device {pci_bus_id}")
    return numa_node


def _cpu_sibling_groups(sysfs_root: Path, cpus: set[int]) -> list[set[int]]:
    groups: list[set[int]] = []
    for cpu in sorted(cpus):
        sibling_path = (
            sysfs_root
            / "devices"
            / "system"
            / "cpu"
            / f"cpu{cpu}"
            / "topology"
            / "thread_siblings_list"
        )
        try:
            siblings = _parse_cpu_list(sibling_path.read_text(encoding="utf-8")) & cpus
        except (OSError, ValueError):
            siblings = {cpu}
        if not siblings:
            siblings = {cpu}
        # Merge overlapping declarations instead of trusting every sysfs file
        # to be simultaneously readable.  This prevents an SMT sibling from
        # appearing in two rank partitions if one CPU's topology read races a
        # hotplug event or is individually unavailable.
        merged = set(siblings)
        disjoint: list[set[int]] = []
        for group in groups:
            if group & merged:
                merged.update(group)
            else:
                disjoint.append(group)
        disjoint.append(merged)
        groups = disjoint
    return sorted(groups, key=lambda group: (min(group), tuple(sorted(group))))


def _partition_cpu_groups(
    groups: list[set[int]],
    *,
    slot: int,
    slots: int,
) -> set[int]:
    if slots <= 0 or slot < 0 or slot >= slots:
        raise ValueError(f"invalid affinity partition slot={slot} slots={slots}")
    groups_per_slot, remainder = divmod(len(groups), slots)
    count = groups_per_slot + (1 if slot < remainder else 0)
    start = slot * groups_per_slot + min(slot, remainder)
    if count == 0:
        raise RuntimeError(
            f"NUMA node has only {len(groups)} physical CPU groups for {slots} local ranks"
        )
    selected: set[int] = set()
    for group in groups[start : start + count]:
        selected.update(group)
    return selected


def _apply_rank_local_cpu_affinity(*, sysfs_root: Path = Path("/sys")) -> bool:
    """Optionally bind this torchrun child to CPUs local to its selected GPU.

    This is deliberately best-effort.  Any missing/malformed topology data or
    denied sched_setaffinity call leaves the inherited affinity untouched.
    """

    try:
        enabled = _env_flag(_RANK_LOCAL_CPU_AFFINITY_ENV, default=False)
    except SystemExit as exc:
        _affinity_warning(exc)
        return False
    if not enabled:
        return False

    try:
        local_rank_raw = os.environ.get("LOCAL_RANK", "0")
        local_world_size_raw = os.environ.get("LOCAL_WORLD_SIZE", "1")
        local_rank = int(local_rank_raw)
        local_world_size = int(local_world_size_raw)
        if local_rank < 0 or local_world_size <= 0 or local_rank >= local_world_size:
            raise ValueError(
                f"invalid torchrun topology LOCAL_RANK={local_rank_raw!r} "
                f"LOCAL_WORLD_SIZE={local_world_size_raw!r}"
            )

        cuda_visible_raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if cuda_visible_raw and cuda_visible_raw.lower() != "all":
            visible_tokens = _split_cuda_visible_devices(cuda_visible_raw)
            if len(visible_tokens) < local_world_size:
                raise ValueError(
                    f"CUDA_VISIBLE_DEVICES={cuda_visible_raw!r} has fewer entries than "
                    f"LOCAL_WORLD_SIZE={local_world_size}"
                )
            rank_tokens = visible_tokens[:local_world_size]
        else:
            rank_tokens = [str(rank) for rank in range(local_world_size)]

        inventory = _query_gpu_inventory()
        rank_topology: list[tuple[str, str, int]] = []
        for token in rank_tokens:
            _, _, pci_bus_id = _gpu_inventory_row(token, inventory)
            rank_topology.append(
                (token, pci_bus_id, _numa_node_for_pci_device(sysfs_root, pci_bus_id))
            )

        token, pci_bus_id, numa_node = rank_topology[local_rank]
        ranks_on_numa = [
            rank for rank, (_, _, rank_numa) in enumerate(rank_topology) if rank_numa == numa_node
        ]
        slot = ranks_on_numa.index(local_rank)

        node_cpu_path = (
            sysfs_root / "devices" / "system" / "node" / f"node{numa_node}" / "cpulist"
        )
        node_cpus = _parse_cpu_list(node_cpu_path.read_text(encoding="utf-8"))
        online_path = sysfs_root / "devices" / "system" / "cpu" / "online"
        try:
            node_cpus &= _parse_cpu_list(online_path.read_text(encoding="utf-8"))
        except OSError:
            pass
        if not node_cpus:
            raise RuntimeError(f"NUMA node {numa_node} has no online CPUs")

        cpu_groups = _cpu_sibling_groups(sysfs_root, node_cpus)
        partition = _partition_cpu_groups(
            cpu_groups,
            slot=slot,
            slots=len(ranks_on_numa),
        )
        inherited_affinity = set(os.sched_getaffinity(0))
        selected_cpus = partition & inherited_affinity
        if not selected_cpus:
            raise RuntimeError(
                f"GPU-local CPU partition {_format_cpu_list(partition)} does not intersect "
                f"the inherited affinity {_format_cpu_list(inherited_affinity)}"
            )

        os.sched_setaffinity(0, selected_cpus)
        selected_cpu_list = _format_cpu_list(selected_cpus)
        os.environ["HOLOSOMA_RANK_LOCAL_CPU_AFFINITY_APPLIED"] = "1"
        os.environ["HOLOSOMA_RANK_LOCAL_CPU_AFFINITY_CPUS"] = selected_cpu_list
        os.environ["HOLOSOMA_RANK_LOCAL_CPU_AFFINITY_NUMA_NODE"] = str(numa_node)
        os.environ["HOLOSOMA_RANK_LOCAL_CPU_AFFINITY_GPU"] = token
        os.environ["HOLOSOMA_RANK_LOCAL_CPU_AFFINITY_PCI_BUS_ID"] = pci_bus_id
        try:
            print(
                "[INFO] rank-local CPU affinity enabled: "
                f"local_rank={local_rank}/{local_world_size} gpu={token!r} "
                f"pci={pci_bus_id} numa={numa_node} cpus={selected_cpu_list}",
                flush=True,
            )
        except OSError:
            # Logging failure after a successful atomic sched_setaffinity call
            # is not an affinity failure and must not report a false fallback.
            pass
        return True
    except (Exception, SystemExit) as exc:
        _affinity_warning(exc)
        return False


def _remap_rank_to_single_visible_gpu() -> None:
    if not _env_flag("HOLOSOMA_RANK_VISIBLE_DEVICES", default=True):
        return

    original_local_rank_raw = os.environ.get("LOCAL_RANK", "0")
    try:
        original_local_rank = int(original_local_rank_raw)
    except ValueError as exc:
        raise SystemExit(f"Invalid LOCAL_RANK for rank-visible launch: {original_local_rank_raw!r}") from exc
    if original_local_rank < 0:
        raise SystemExit(
            f"LOCAL_RANK must be non-negative for rank-visible launch: {original_local_rank}"
        )

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

    # These values describe the current torchrun child, not inherited user
    # configuration.  Preserving stale HOLOSOMA_ORIGINAL_* variables from a
    # parent shell can make the simulator bind rank 0 to another worker's GPU
    # and can corrupt hierarchical local-rank/leader selection.  Always derive
    # and overwrite the aliases from torchrun's authoritative environment
    # before narrowing CUDA visibility.
    os.environ["HOLOSOMA_ORIGINAL_LOCAL_RANK"] = str(original_local_rank)
    os.environ["HOLOSOMA_ORIGINAL_LOCAL_WORLD_SIZE"] = os.environ.get("LOCAL_WORLD_SIZE", "")
    os.environ["HOLOSOMA_ORIGINAL_CUDA_VISIBLE_DEVICES"] = original_cuda_visible
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
    # Apply affinity while torchrun's authoritative LOCAL_RANK and original
    # CUDA_VISIBLE_DEVICES list are still intact, and before importing
    # torch/Isaac/PhysX through train_agent.py.
    _apply_rank_local_cpu_affinity()
    _remap_rank_to_single_visible_gpu()
    target = Path(__file__).with_name("train_agent.py")
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
