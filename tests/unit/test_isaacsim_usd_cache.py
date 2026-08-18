from __future__ import annotations

from pathlib import Path

import pytest

from holosoma.simulator.isaacsim.usd_cache import resolve_robot_usd_conversion_dir


def test_robot_usd_cache_preserves_legacy_asset_root_default(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"

    result = resolve_robot_usd_conversion_dir(asset_root, 3, environ={})

    assert result == asset_root.resolve() / "converted_rank3"


def test_robot_usd_cache_uses_explicit_rank_private_runtime_root(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime" / "robot_usd"

    rank_zero = resolve_robot_usd_conversion_dir(
        tmp_path / "immutable-assets",
        0,
        environ={"HOLOSOMA_ROBOT_USD_CACHE_DIR": str(runtime_root)},
    )
    rank_seven = resolve_robot_usd_conversion_dir(
        tmp_path / "immutable-assets",
        7,
        environ={"HOLOSOMA_ROBOT_USD_CACHE_DIR": str(runtime_root)},
    )

    assert rank_zero == runtime_root.resolve() / "converted_rank0"
    assert rank_seven == runtime_root.resolve() / "converted_rank7"
    assert rank_zero.parent == rank_seven.parent
    assert rank_zero != rank_seven


@pytest.mark.parametrize(
    ("root", "error"),
    [
        ("relative/cache", "absolute path"),
        ("/", "filesystem root"),
    ],
)
def test_robot_usd_cache_rejects_unsafe_explicit_roots(root: str, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        resolve_robot_usd_conversion_dir(
            "/immutable-assets",
            0,
            environ={"HOLOSOMA_ROBOT_USD_CACHE_DIR": root},
        )


@pytest.mark.parametrize("local_rank", [-1, True, 1.5, "1"])
def test_robot_usd_cache_rejects_invalid_local_rank(local_rank: object) -> None:
    with pytest.raises(ValueError, match="local_rank"):
        resolve_robot_usd_conversion_dir(
            "/immutable-assets",
            local_rank,  # type: ignore[arg-type]
            environ={},
        )
