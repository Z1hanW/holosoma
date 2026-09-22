"""Fast-FoundationStereo depth predictor.

Drop-in alternative to `holosoma.models.gum.infer.GUM`: same `predict()` contract,
so `image_server.py` can produce the `depth_gum` channel from Fast-FoundationStereo
(NVlabs, CVPR 2026) instead of GUM without any change to the policy side.

Enable by setting `HOLOSOMA_DEPTH_PREDICTOR=ffs` alongside an image server config
that already has `enable_gum_depth_prediction=True` (e.g. `real_depth_gum_d435i`).

Requires, none of which ship with this repo:
  - the Fast-FoundationStereo checkout on PYTHONPATH (HOLOSOMA_FFS_REPO)
  - a model checkpoint plus its sibling `cfg.yaml` (HOLOSOMA_FFS_MODEL)
  - a CUDA device; the upstream model is not usable at frame rate on CPU

NOTE: this has not been run end to end — no machine here has a working CUDA
device or the upstream weights. The disparity->depth conversion follows
`scripts/run_demo.py` upstream (`depth = fx * baseline / disparity`).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FFSConfig:
    """Configuration for the Fast-FoundationStereo depth predictor."""

    repo_dir: str = ""
    """Fast-FoundationStereo checkout. Defaults to $HOLOSOMA_FFS_REPO."""

    model_checkpoint: str = ""
    """Checkpoint path; its directory must also contain cfg.yaml. Defaults to $HOLOSOMA_FFS_MODEL."""

    device: str = "cuda"
    valid_iters: int = 16
    """Refinement iterations. Lower is faster; upstream demo defaults to 32."""

    max_disp: int = 192
    hierarchical: bool = False
    """Use run_hierachical() (upstream spelling) for high-resolution inputs."""

    depth_min: float = 0.2
    depth_max: float = 10.0


class FastFoundationStereo:
    """Predict metric depth from a side-by-side stereo pair."""

    def __init__(self, cfg: FFSConfig | None = None, dtype=None):
        import torch

        self.cfg = cfg or FFSConfig()
        self.torch = torch

        repo_dir = self.cfg.repo_dir or os.environ.get("HOLOSOMA_FFS_REPO", "")
        model_path = self.cfg.model_checkpoint or os.environ.get("HOLOSOMA_FFS_MODEL", "")
        if not repo_dir:
            raise ValueError("Fast-FoundationStereo repo not set. Set HOLOSOMA_FFS_REPO or FFSConfig.repo_dir.")
        if not model_path:
            raise ValueError("Fast-FoundationStereo checkpoint not set. Set HOLOSOMA_FFS_MODEL.")

        repo_dir = str(Path(repo_dir).expanduser().resolve())
        if repo_dir not in sys.path:
            # Upstream modules import each other as top-level packages (core.*, Utils).
            sys.path.insert(0, repo_dir)

        model_path = Path(model_path).expanduser().resolve()
        cfg_path = model_path.parent / "cfg.yaml"
        if not model_path.exists():
            raise FileNotFoundError(f"Fast-FoundationStereo checkpoint not found: {model_path}")
        if not cfg_path.exists():
            raise FileNotFoundError(f"Expected cfg.yaml next to the checkpoint: {cfg_path}")

        self.device = torch.device(self.cfg.device)
        self.dtype = dtype

        # Upstream ships the whole model pickled, not a state dict.
        self.model = torch.load(str(model_path), map_location="cpu", weights_only=False)
        self.model.args.max_disp = self.cfg.max_disp
        self.model = self.model.to(self.device).eval()

        from Utils import InputPadder  # noqa: E402  (only importable once repo_dir is on sys.path)

        self._InputPadder = InputPadder

        print(f"[FFS] Initialized on {self.device} from {model_path}")
        print(f"[FFS] valid_iters={self.cfg.valid_iters} max_disp={self.cfg.max_disp}")
        print(f"[FFS] Depth range: [{self.cfg.depth_min}, {self.cfg.depth_max}] meters")

    @staticmethod
    def _baseline_from_extrinsics(camera_extrinsics: np.ndarray) -> float:
        """Stereo baseline in meters from the (2, 4, 4) extrinsics pair.

        Matches the convention in holosoma.sensors.zed, where the right eye's
        extrinsics carry the horizontal offset in the translation column.
        """
        extr = np.asarray(camera_extrinsics, dtype=np.float64)
        if extr.shape != (2, 4, 4):
            raise ValueError(f"camera_extrinsics must be (2, 4, 4), got {extr.shape}")
        baseline = abs(float(extr[1][0, 3]) - float(extr[0][0, 3]))
        if baseline <= 0.0:
            raise ValueError(f"Non-positive stereo baseline derived from extrinsics: {baseline}")
        return baseline

    def predict(
        self,
        side_by_side_image: np.ndarray,
        camera_intrinsics: np.ndarray,  # (2, 3, 3)
        camera_extrinsics: np.ndarray,  # (2, 4, 4)
    ) -> np.ndarray:
        """Return metric depth (H, W) in meters for one side-by-side stereo image."""
        torch = self.torch

        img = np.asarray(side_by_side_image)
        if img.ndim != 3 or img.shape[1] % 2 != 0:
            raise ValueError(f"expected a side-by-side HxWx3 image with even width, got {img.shape}")
        half = img.shape[1] // 2
        left, right = img[:, :half], img[:, half:]

        intr = np.asarray(camera_intrinsics, dtype=np.float64)
        if intr.shape != (2, 3, 3):
            raise ValueError(f"camera_intrinsics must be (2, 3, 3), got {intr.shape}")
        fx = float(intr[0][0, 0])
        baseline = self._baseline_from_extrinsics(camera_extrinsics)

        with torch.no_grad():
            t0 = torch.from_numpy(np.ascontiguousarray(left)).to(self.device).float()[None].permute(0, 3, 1, 2)
            t1 = torch.from_numpy(np.ascontiguousarray(right)).to(self.device).float()[None].permute(0, 3, 1, 2)
            padder = self._InputPadder(t0.shape, divis_by=32, force_square=False)
            t0, t1 = padder.pad(t0, t1)

            if self.cfg.hierarchical:
                disp = self.model.run_hierachical(t0, t1, iters=self.cfg.valid_iters, test_mode=True, small_ratio=0.5)
            else:
                disp = self.model.forward(t0, t1, iters=self.cfg.valid_iters, test_mode=True)

            disp = padder.unpad(disp.float())

        h, w = left.shape[:2]
        disp = disp.detach().cpu().numpy().reshape(h, w)

        # disparity -> metric depth; guard the division so zero/negative disparity
        # becomes far-plane rather than inf/NaN, which the policy cannot consume.
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = (fx * baseline) / disp
        depth[~np.isfinite(depth)] = self.cfg.depth_max
        depth[disp <= 0] = self.cfg.depth_max

        return np.clip(depth, self.cfg.depth_min, self.cfg.depth_max).astype(np.float32)
