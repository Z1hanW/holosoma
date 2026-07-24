from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import holosoma.utils.runtime_asset_manifest as runtime_asset_manifest


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFM_ROOT = REPO_ROOT / "submodules" / "defm"
MODULES_SOURCE = (
    REPO_ROOT / "src" / "holosoma" / "holosoma" / "agents" / "modules" / "modules.py"
)


def test_holosoma_defm_factory_does_not_import_upstream_model_factory() -> None:
    source = MODULES_SOURCE.read_text(encoding="utf-8")
    assert 'import_module("defm.model_factory")' not in source
    assert 'import_module("defm.models.vision_transformer")' in source


def test_holosoma_local_defm_runtime_does_not_import_huggingface() -> None:
    script = r"""
import importlib.abc
import sys

class BlockHuggingFace(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "huggingface_hub" or fullname.startswith("huggingface_hub."):
            raise ModuleNotFoundError("test blocked huggingface_hub")
        return None

sys.meta_path.insert(0, BlockHuggingFace())
from holosoma.agents.modules import modules
factory, preprocess = modules._load_defm_runtime()
assert callable(factory)
assert callable(preprocess)
model = factory("defm_vit_s14", pretrained=False, pretrained_path=None)
assert model.__class__.__name__ == "DinoVisionTransformer"
try:
    factory("defm_vit_s14", pretrained=True, pretrained_path=None)
except ValueError:
    pass
else:
    raise AssertionError("HoloSoma DeFM factory accepted upstream weight loading")
assert "defm.model_factory" not in sys.modules
assert not any(
    name == "huggingface_hub" or name.startswith("huggingface_hub.")
    for name in sys.modules
)
"""
    env = os.environ.copy()
    env["HOLOSOMA_DEFM_ROOT"] = str(DEFM_ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        (
            str(REPO_ROOT / "src" / "holosoma"),
            str(REPO_ROOT / "src" / "holosoma_inference"),
            str(REPO_ROOT / "src"),
        )
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_local_defm_manifest_excludes_network_client(monkeypatch) -> None:
    observed: list[str] = []

    def fake_manifest(name: str) -> dict[str, str]:
        observed.append(name)
        return {"distribution": name}

    monkeypatch.setattr(
        runtime_asset_manifest,
        "_distribution_content_manifest",
        fake_manifest,
    )

    records = runtime_asset_manifest._defm_python_dependency_manifest()

    assert observed == ["torchvision", "omegaconf", "Pillow"]
    assert [record["distribution"] for record in records] == observed
    assert "huggingface-hub" not in observed
