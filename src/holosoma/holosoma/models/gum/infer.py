import torch
import numpy as np
from pathlib import Path
from typing import Any


class GUM:
    """GUM (Geometric Understanding Model) for depth prediction from RGB images.
    
    This class wraps the GUM model for depth prediction. It handles model loading,
    torch operations initialization, and provides a simple predict interface.
    """
    
    def __init__(
        self,
        cfg: Any,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize GUM model.
        
        Args:
            cfg: Config object with fields compatible with
                        `GUMConfig` in `image_server.py`.
            dtype: Data type for model (torch.bfloat16 or torch.float32)
        """
        if cfg is None:
            raise ValueError("cfg must be provided.")
        self.cfg = cfg
        self.device = torch.device(self.cfg.device)
        self.dtype = dtype

        if self.cfg.target_h % 32 != 0 or self.cfg.target_w % 32 != 0:
            raise ValueError(
                "cfg.target_h and cfg.target_w must be multiples of 32, "
                f"got target_h={self.cfg.target_h}, target_w={self.cfg.target_w}."
            )

        self.root_folder = Path(self.cfg.root_folder).expanduser().resolve() if self.cfg.root_folder else None
        # Load torch operations
        torch_ops_dir = self.cfg.torch_ops_dir
        if not torch_ops_dir:
            torch_ops_dir = "torch_ops/build"
        self._load_torch_ops(str(self._resolve_gum_relative_path(torch_ops_dir)))
        
        # Load model
        self._load_model(str(self._resolve_gum_relative_path(self.cfg.model_checkpoint)))

        print(f"[GUM] Initialized successfully on {self.device}")
        print(f"[GUM] Model checkpoint: {self._resolve_gum_relative_path(self.cfg.model_checkpoint)}")
        print(f"[GUM] Target inference size (HxW): {self.cfg.target_h}x{self.cfg.target_w}")
        print(f"[GUM] Depth range: [{self.cfg.depth_min}, {self.cfg.depth_max}] meters")

    def _resolve_gum_dir(self) -> Path:
        """Resolve the models/gum directory."""
        if self.root_folder is not None:
            return self.root_folder / "src" / "holosoma" / "holosoma" / "models" / "gum"
        return Path(__file__).resolve().parent

    def _resolve_gum_relative_path(self, input_path: str | Path) -> Path:
        """Resolve a path relative to models/gum when input path is relative."""
        path = Path(input_path).expanduser()
        if path.is_absolute():
            return path

        # Primary behavior: paths are relative to models/gum under root_folder.
        if self.root_folder is not None:
            primary = self.root_folder / "src" / "holosoma" / "holosoma" / "models" / "gum" / path
        else:
            primary = Path(__file__).resolve().parent / path
        if primary.exists():
            return primary

        # Fallback: allow relative-to-root paths if root_folder was provided.
        if self.root_folder is not None:
            fallback = self.root_folder / path
            if fallback.exists():
                return fallback

        return primary
    
    def _load_torch_ops(self, torch_ops_dir: str):
        """Load torch custom operations."""
        ops_dir = Path(torch_ops_dir)
        if not ops_dir.exists():
            raise FileNotFoundError(f"Torch ops directory not found: {torch_ops_dir}")
        
        ops_files = [
            "patch_outerprod_cuda.so",
            "sym3eig_cuda.so",
            "soft_bin_ops_cuda.so",
        ]
        
        for op_file in ops_files:
            op_path = ops_dir / op_file
            if op_path.exists():
                torch.ops.load_library(str(op_path))
                print(f"[GUM] Loaded torch op: {op_file}")
            else:
                raise FileNotFoundError(f"Torch op not found: {op_path}")
    
    def _load_model(self, checkpoint_path: str):
        """Load GUM model from checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

        # Some TorchScript checkpoints reference torchvision custom ops
        # (e.g. torchvision::nms), which are registered on torchvision import.
        try:
            import torchvision  # noqa: F401  # pyright: ignore[reportMissingImports]
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "torchvision is required to load this GUM checkpoint because it uses "
                "TorchVision custom ops (e.g. torchvision::nms)."
            ) from exc

        self.model = torch.jit.load(str(checkpoint_path), map_location='cpu')
        self.model = self.model.eval().to(device=self.device, dtype=self.dtype)
        print(f"[GUM] Loaded model from: {checkpoint_path}")
    
    def _split_side_by_side_image(self, side_by_side_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split side-by-side image into left and right images."""
        img_height, double_img_width, _ = side_by_side_image.shape
        if double_img_width % 2 != 0:
            raise ValueError(f"Side-by-side width must be even, got {double_img_width}.")
        img_width = double_img_width // 2
        left = side_by_side_image[:, :img_width, :]
        right = side_by_side_image[:, img_width:, :]

        left_tensor = torch.from_numpy(left).float().permute(2, 0, 1).to(self.device) / 255.0
        right_tensor = torch.from_numpy(right).float().permute(2, 0, 1).to(self.device) / 255.0
        return left_tensor, right_tensor
    
    def _resize_to_target_size(self, left_tensor: torch.Tensor, right_tensor: torch.Tensor, camera_intrinsics: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Resize image to target size and adjust intrinsics to target size."""
        img_height, img_width = left_tensor.shape[-2:]
        target_height, target_width = self.cfg.target_h, self.cfg.target_w
        intrinsics = torch.from_numpy(camera_intrinsics).to(self.device)

        left_tensor = torch.nn.functional.interpolate(
            left_tensor.unsqueeze(0),
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        right_tensor = torch.nn.functional.interpolate(
            right_tensor.unsqueeze(0),
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0) # (C, H, W)

        # adjust intrinsics and to target size
        intrinsics[:, 0, :] *= float(target_width) / float(img_width)
        intrinsics[:, 1, :] *= float(target_height) / float(img_height)

        return left_tensor, right_tensor, intrinsics
    

    def predict(
        self,
        side_by_side_image: np.ndarray,
        camera_intrinsics: np.ndarray, # shape (2, 3, 3)
        camera_extrinsics: np.ndarray, # shape (2, 4, 4)
    ) -> np.ndarray:
        """Predict depth for one side-by-side stereo camera image.
        
        Args:
            side_by_side_image: Side-by-side concatenated RGB image, shape (H, 2*W, 3).
                Left/right stereo images are concatenated horizontally.
            camera_intrinsics: Camera intrinsics array with shape (2, 3, 3).
            camera_extrinsics: Camera extrinsics array with shape (2, 4, 4).
        
        Returns:
            Depth prediction (H, W) in meters.
        """
        depth_min = torch.tensor(self.cfg.depth_min, device=self.device)
        depth_max = torch.tensor(self.cfg.depth_max, device=self.device)

        with torch.no_grad():

            img_left_tensor, img_right_tensor = self._split_side_by_side_image(side_by_side_image)
            img_height, img_width = img_left_tensor.shape[-2:]

            left_tensor, right_tensor, intrinsics = self._resize_to_target_size(img_left_tensor, img_right_tensor, camera_intrinsics)
            extrinsics = torch.from_numpy(camera_extrinsics).to(self.device)

            # Predict depth [target_height, target_width]
            rgbs = torch.stack([left_tensor, right_tensor], dim=0) # (2, C, H, W)
            backbone_out = self.model.predict_backbone(rgbs)
            im_size = torch.tensor([self.cfg.target_h, self.cfg.target_w], device=self.device)
            out = self.model.predict_depth(
                rgbs,
                im_size,
                intrinsics,
                intrinsics.inverse(),
                extrinsics,
                depth_min,
                depth_max,
                backbone_out,
            )

            # Resize the predicted depth back to the original image size
            # out["depth_pred"] (target_height, target_width) -> (img_height, img_width)
            out["depth_pred"] = torch.nn.functional.interpolate(
                out["depth_pred"].unsqueeze(0).unsqueeze(0),
                size=(img_height, img_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

            return out["depth_pred"].cpu().numpy().astype(np.float32)