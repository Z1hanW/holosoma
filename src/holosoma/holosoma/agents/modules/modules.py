from __future__ import annotations

import inspect
import os
import sys
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from holosoma.config_types.algo import LayerConfig, ModuleConfig


class ImgChLayerNorm(nn.Module):
    """Image channel-wise layer normalization."""

    def __init__(self, num_channels, eps: float = 1e-5):
        """Initialize ImgChLayerNorm module.

        Parameters
        ----------
        num_channels: int
            Number of channels in the input tensor
        eps: float, optional
            Small value to prevent division by zero, by default 1e-5
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        """Forward pass for image channel-wise layer normalization.

        Normalizes each channel of the input tensor independently.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape [B, C, H, W]

        Returns
        -------
        torch.Tensor
            Output tensor of shape [B, C, H, W]
        """
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class CNNWrapper(nn.Module):
    """Wrapper module that handles reshaping for CNN layers when working with flattened inputs."""

    def __init__(self, cnn_layers, input_channels, input_height, input_width, flatten_output=True):
        """Initialize CNNWrapper module.

        Wraps CNN layers to handle reshaping for CNN layers when working with flattened inputs.
        For instance, this is useful when giving the flattened output of a CNN layer to an MLP layer.

        Parameters
        ----------
        cnn_layers: nn.Module
            CNN layers to wrap
        input_channels: int
            Number of input channels
        input_height: int
            Height of input feature maps
        input_width: int
            Width of input feature maps
        flatten_output: bool, optional
            Whether to flatten the output, by default True
        """
        super().__init__()
        self.cnn_layers = cnn_layers
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.expected_input_size = input_channels * input_height * input_width
        self.flatten_output = flatten_output

    @property
    def output_size(self):
        """Computes the output size of the CNN layers by doing a forward pass with dummy data."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels * self.input_height * self.input_width)
            dummy_output = self.forward(dummy_input)
            return dummy_output.shape[-1]

    def forward(self, x):
        """Forward pass for CNNWrapper module.

        Reshapes the input tensor to (batch_size, channels, height, width) and applies the CNN layers.
        If flatten_output is True, flattens the output back to (batch_size, -1).

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape [B, C, H, W]

        Returns
        -------
        torch.Tensor
            Output tensor of shape [B, -1] if flatten_output is True, otherwise [B, C, H, W]
        """
        # Validate input size
        batch_size = x.shape[0]
        if x.shape[1] != self.expected_input_size:
            raise ValueError(
                f"Input size mismatch: expected {self.expected_input_size} "
                f"(channels={self.input_channels}, height={self.input_height}, width={self.input_width}), "
                f"but got {x.shape[1]}"
            )

        # Reshape from flattened input to (batch_size, channels, height, width)
        x = x.view(batch_size, self.input_channels, self.input_height, self.input_width)

        # Apply CNN layers
        x = self.cnn_layers(x)

        if self.flatten_output:
            # Flatten back to (batch_size, -1)
            x = x.view(batch_size, -1)
        else:
            # x is currently [batch_size, channels, height, width]
            # Reshape to [batch_size, height * width, channels]
            x = x.view(batch_size, x.shape[1], -1)
            x = x.permute(0, 2, 1)

        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer inputs."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1], : x.shape[2]]


def _resolve_transformer_activation(name: str):
    if not isinstance(name, str):
        return "gelu"
    lower = name.lower()
    if lower in ("relu", "gelu"):
        return lower
    if lower == "elu":
        return F.elu
    if lower == "tanh":
        return torch.tanh
    return "gelu"


class TargetPoseTransformer(nn.Module):
    """Transformer encoder for flattened target-pose sequences."""

    def __init__(
        self,
        obs_dim: int,
        num_steps: int,
        latent_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        activation: str,
        pooling: str,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_steps = num_steps
        self.pooling = pooling
        self.input_proj = nn.Linear(obs_dim, latent_dim)
        self.positional_encoding = PositionalEncoding(latent_dim, max_len=num_steps)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation=_resolve_transformer_activation(activation),
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.view(batch_size, self.num_steps, self.obs_dim)
        x = self.input_proj(x)
        x = self.positional_encoding(x)
        x = self.encoder(x)
        if self.pooling == "mean":
            return x.mean(dim=1)
        return x[:, 0, :]


class ObsTargetPoseTransformer(nn.Module):
    """Transformer encoder over current-obs token + future target-pose tokens."""

    def __init__(
        self,
        obs_dim: int,
        target_obs_dim: int,
        num_steps: int,
        latent_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        activation: str,
        pooling: str,
        layer_config: LayerConfig,
        encoder_hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.target_obs_dim = target_obs_dim
        self.num_steps = num_steps
        self.latent_dim = latent_dim
        self.pooling = pooling

        hidden_dims = encoder_hidden_dims or []
        self.obs_encoder = build_mlp_layer(obs_dim, hidden_dims, latent_dim, layer_config)
        self.target_encoder = build_mlp_layer(target_obs_dim, hidden_dims, latent_dim, layer_config)

        self.positional_encoding = PositionalEncoding(latent_dim, max_len=num_steps + 1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation=_resolve_transformer_activation(activation),
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, obs_token: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
        batch_size = obs_token.shape[0]
        if target_tokens.shape[-1] != self.num_steps * self.target_obs_dim:
            raise ValueError(
                f"Expected target_tokens shape (*,{self.num_steps * self.target_obs_dim}), got {target_tokens.shape}"
            )

        target_tokens = target_tokens.view(batch_size, self.num_steps, self.target_obs_dim)

        obs_embed = self.obs_encoder(obs_token).unsqueeze(1)
        target_flat = target_tokens.reshape(-1, self.target_obs_dim)
        target_embed = self.target_encoder(target_flat).view(batch_size, self.num_steps, self.latent_dim)

        seq = torch.cat([obs_embed, target_embed], dim=1)
        seq = self.positional_encoding(seq)
        seq = self.encoder(seq)

        if self.pooling == "mean":
            return seq.mean(dim=1)
        return seq[:, 0, :]


class TerrainObsTokenTransformer(nn.Module):
    """Transformer encoder over output token + proprio token + depth token + optional target tokens."""

    def __init__(
        self,
        proprio_dim: int,
        depth_dim: int,
        target_obs_dim: int | None,
        num_steps: int,
        latent_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        activation: str,
        pooling: str,
        layer_config: LayerConfig,
        encoder_hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        self.proprio_dim = proprio_dim
        self.depth_dim = depth_dim
        self.target_obs_dim = target_obs_dim
        self.num_steps = num_steps
        self.latent_dim = latent_dim
        self.pooling = pooling

        hidden_dims = encoder_hidden_dims or []
        self.output_token = nn.Parameter(torch.zeros(1, 1, latent_dim))
        self.proprio_encoder = build_mlp_layer(proprio_dim, hidden_dims, latent_dim, layer_config)
        self.depth_encoder = build_mlp_layer(depth_dim, hidden_dims, latent_dim, layer_config)
        self.target_encoder = None
        if self.target_obs_dim is not None and self.num_steps > 0:
            self.target_encoder = build_mlp_layer(target_obs_dim, hidden_dims, latent_dim, layer_config)

        max_len = 3 + (self.num_steps if self.target_encoder is not None else 0)
        self.positional_encoding = PositionalEncoding(latent_dim, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation=_resolve_transformer_activation(activation),
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        nn.init.trunc_normal_(self.output_token, std=0.02)

    def forward(
        self,
        proprio_token: torch.Tensor,
        depth_token: torch.Tensor,
        target_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = proprio_token.shape[0]

        seq_parts = [
            self.output_token.expand(batch_size, -1, -1),
            self.proprio_encoder(proprio_token).unsqueeze(1),
            self.depth_encoder(depth_token).unsqueeze(1),
        ]

        if self.target_encoder is not None:
            if target_tokens is None:
                raise ValueError("target_tokens must be provided when target_encoder is configured.")
            expected_dim = self.num_steps * self.target_obs_dim
            if target_tokens.shape[-1] != expected_dim:
                raise ValueError(f"Expected target_tokens shape (*,{expected_dim}), got {target_tokens.shape}")
            target_tokens = target_tokens.view(batch_size, self.num_steps, self.target_obs_dim)
            target_flat = target_tokens.reshape(-1, self.target_obs_dim)
            target_embed = self.target_encoder(target_flat).view(batch_size, self.num_steps, self.latent_dim)
            seq_parts.append(target_embed)

        seq = torch.cat(seq_parts, dim=1)
        seq = self.positional_encoding(seq)
        seq = self.encoder(seq)

        if self.pooling == "mean":
            return seq.mean(dim=1)
        return seq[:, 0, :]


class MLPWithExtraProj(nn.Module):
    """MLP that adds a projected extra input to the first hidden layer."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        output_dim: int,
        layer_config: LayerConfig,
    ):
        super().__init__()
        if hidden_dims is None or len(hidden_dims) == 0:
            raise ValueError("extra_input_to_hidden requires at least one hidden layer.")

        self.supports_extra_input = True
        self.first = nn.Linear(input_dim, hidden_dims[0])
        self.extra_proj = nn.LazyLinear(hidden_dims[0])
        self.activation = getattr(nn, layer_config.activation)()
        self.dropout = nn.Dropout(p=layer_config.dropout_prob) if layer_config.dropout_prob > 0 else None

        layers = []
        for layer_idx in range(len(hidden_dims)):
            if layer_idx == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[layer_idx], output_dim))
            else:
                layers.append(nn.Linear(hidden_dims[layer_idx], hidden_dims[layer_idx + 1]))
                layers.append(getattr(nn, layer_config.activation)())
                if layer_config.dropout_prob > 0:
                    layers.append(nn.Dropout(p=layer_config.dropout_prob))
        self.rest = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, extra_input: torch.Tensor | None = None) -> torch.Tensor:
        x = self.first(x)
        if extra_input is not None:
            if extra_input.is_inference():
                # Final safety: ensure extra_input is a normal tensor for autograd.
                extra_input = extra_input.clone()
            x = x + self.extra_proj(extra_input)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.rest(x)


def build_mlp_layer(
    input_dim,
    hidden_dims,
    output_dim,
    layer_config,
):
    """Builds a multi-layer perceptron (MLP) layer.

    Parameters
    ----------
    input_dim: int
        Number of input dimensions
    hidden_dims: tuple[int, ...]
        Tuple of hidden dimensions
    output_dim: int
        Number of output dimensions
    layer_config: dict
        Dictionary containing:
        - activation: Activation function name (e.g., "ReLU")
        - dropout_prob: Dropout probability (default: 0)

    Returns
    -------
    nn.Sequential
        The constructed MLP layer
    """
    use_extra_proj = bool(getattr(layer_config, "extra_input_to_hidden", False))
    if hidden_dims is None:
        if use_extra_proj:
            raise ValueError("extra_input_to_hidden requires hidden_dims to be set.")
        return None
    if use_extra_proj:
        return MLPWithExtraProj(input_dim, tuple(hidden_dims), output_dim, layer_config)

    layers = []
    activation = getattr(nn, layer_config.activation)()
    dropout = layer_config.dropout_prob

    if len(hidden_dims) == 0:
        # No hidden layer, just one linear layer
        layers.append(nn.Linear(input_dim, output_dim))
    else:
        # First hidden layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(activation)
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))

        # Additional hidden layers
        for layer_idx in range(len(hidden_dims)):
            if layer_idx == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[layer_idx], output_dim))
            else:
                layers.append(nn.Linear(hidden_dims[layer_idx], hidden_dims[layer_idx + 1]))
                layers.append(activation)
                if dropout > 0:
                    layers.append(nn.Dropout(p=dropout))

    return nn.Sequential(*layers)


class ConditionalFlowMLP(nn.Module):
    """Conditional flow-matching network over action space."""

    supports_flow_matching = True

    def __init__(self, condition_dim: int, action_dim: int, layer_config: LayerConfig):
        super().__init__()
        if layer_config.hidden_dims is None:
            raise ValueError("ConditionalFlowMLP requires hidden_dims to be set.")
        self.condition_dim = int(condition_dim)
        self.action_dim = int(action_dim)
        self.integration_steps = max(1, int(getattr(layer_config, "flow_integration_steps", 4)))
        self.train_noise_std = float(getattr(layer_config, "flow_train_noise_std", 1.0))
        self.time_epsilon = min(max(float(getattr(layer_config, "flow_time_epsilon", 1e-4)), 0.0), 0.49)
        self.inference_noise_std = max(0.0, float(getattr(layer_config, "flow_inference_noise_std", 0.0)))
        self.net = build_mlp_layer(
            self.condition_dim + self.action_dim + 1,
            layer_config.hidden_dims,
            self.action_dim,
            layer_config,
        )

    @property
    def supports_extra_input(self) -> bool:
        return bool(getattr(self.net, "supports_extra_input", False))

    def _time_column(self, t: torch.Tensor | float, batch_size: int, ref: torch.Tensor) -> torch.Tensor:
        if not isinstance(t, torch.Tensor):
            return torch.full((batch_size, 1), float(t), device=ref.device, dtype=ref.dtype)
        t = t.to(device=ref.device, dtype=ref.dtype)
        if t.ndim == 0:
            return t.view(1, 1).expand(batch_size, 1)
        if t.ndim == 1:
            return t.view(-1, 1)
        return t.view(batch_size, 1)

    def velocity(
        self,
        condition: torch.Tensor,
        action_t: torch.Tensor,
        t: torch.Tensor | float,
        extra_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        time_col = self._time_column(t, action_t.shape[0], action_t)
        flow_input = torch.cat([condition, action_t, time_col], dim=-1)
        if extra_input is not None:
            if not self.supports_extra_input:
                raise ValueError("extra_input provided but flow network is not configured for extra_input_to_hidden.")
            if hasattr(extra_input, "is_inference") and extra_input.is_inference():
                extra_input = extra_input.clone()
            return self.net(flow_input, extra_input=extra_input)
        return self.net(flow_input)

    def forward(self, condition: torch.Tensor, extra_input: torch.Tensor | None = None) -> torch.Tensor:
        if self.inference_noise_std > 0.0 and not torch.onnx.is_in_onnx_export():
            action = torch.randn(condition.shape[0], self.action_dim, device=condition.device, dtype=condition.dtype)
            action = action * self.inference_noise_std
        else:
            action = torch.zeros(condition.shape[0], self.action_dim, device=condition.device, dtype=condition.dtype)
        dt = 1.0 / float(self.integration_steps)
        for step in range(self.integration_steps):
            t = float(step) * dt
            action = action + dt * self.velocity(condition, action, t, extra_input=extra_input)
        return action

    def flow_matching_loss(
        self,
        condition: torch.Tensor,
        target_action: torch.Tensor,
        *,
        extra_input: torch.Tensor | None = None,
        loss_fn=F.mse_loss,
    ) -> torch.Tensor:
        noise = torch.randn_like(target_action) * self.train_noise_std
        eps = self.time_epsilon
        t = torch.rand(target_action.shape[0], 1, device=target_action.device, dtype=target_action.dtype)
        if eps > 0.0:
            t = t * (1.0 - 2.0 * eps) + eps
        action_t = (1.0 - t) * noise + t * target_action
        target_velocity = target_action - noise
        pred_velocity = self.velocity(condition, action_t, t, extra_input=extra_input)
        per_elem = loss_fn(pred_velocity, target_velocity, reduction="none")
        return per_elem.mean(dim=-1)


class GatedLinearEncoder(nn.Module):
    """Flatten + linear projection gated by a learned sigmoid."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.view(x.shape[0], -1)
        return self.proj(flat) * torch.sigmoid(self.gate(flat))


class AttentionLinearEncoder(nn.Module):
    """Flatten + linear projection scaled by a learned per-channel gate."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self.attention = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.view(x.shape[0], -1)
        scale = 1.0 + torch.tanh(self.attention)
        return self.proj(flat) * scale


class FarTrackingDepthSmallEncoder(nn.Module):
    """Small far-tracking-style depth CNN: HxW depth -> 32d-style latent."""

    def __init__(self, input_height: int, input_width: int, output_dim: int):
        super().__init__()
        if input_height <= 0 or input_width <= 0:
            raise ValueError(
                f"FarTrackingDepthSmallEncoder expects positive input size, got {(input_height, input_width)}."
            )
        self.input_height = int(input_height)
        self.input_width = int(input_width)
        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=2),
            activation,
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            activation,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, output_dim),
        )
        self.output_activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.view(x.shape[0], -1)
        expected_dim = self.input_height * self.input_width
        if flat.shape[-1] != expected_dim:
            raise ValueError(
                f"FarTrackingDepthSmallEncoder expected flattened input dim {expected_dim}, got {flat.shape[-1]}."
            )
        images = flat.view(x.shape[0], 1, self.input_height, self.input_width)
        latent = self.image_compression(images)
        return self.output_activation(latent)


@lru_cache(maxsize=1)
def _resolve_defm_repo_root() -> Path:
    env_root = os.environ.get("HOLOSOMA_DEFM_ROOT", "").strip()
    candidates: list[Path] = []
    if env_root:
        candidates.append(Path(env_root).expanduser())
    this_file = Path(__file__).resolve()
    for parent in this_file.parents:
        candidates.append(parent / "defm")
        candidates.append(parent / "submodules" / "defm")
    for candidate in candidates:
        if (candidate / "defm" / "model_factory.py").is_file():
            return candidate
    raise FileNotFoundError(
        "Unable to locate the local DeFM source tree. Set HOLOSOMA_DEFM_ROOT to a directory containing "
        "'defm/model_factory.py'."
    )


@lru_cache(maxsize=1)
def _load_defm_runtime():
    repo_root = _resolve_defm_repo_root()
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    try:
        from defm.model_factory import create_defm_model
        from defm.utils.utils import preprocess_depth_batch
    except Exception as exc:
        raise RuntimeError(
            f"Failed to import DeFM from local source tree at {repo_root}. "
            "Make sure its Python dependencies are installed."
        ) from exc
    return create_defm_model, preprocess_depth_batch


def _resolve_defm_forward_batch_size(model_name: str, freeze_backbone: bool) -> int:
    raw_value = os.environ.get("HOLOSOMA_DEFM_FORWARD_BATCH_SIZE", "").strip()
    if raw_value:
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise ValueError("HOLOSOMA_DEFM_FORWARD_BATCH_SIZE must be an integer.") from exc
        if value < 0:
            raise ValueError("HOLOSOMA_DEFM_FORWARD_BATCH_SIZE must be >= 0.")
        return value
    if freeze_backbone and "vit" in model_name:
        return 256
    if freeze_backbone and model_name.startswith("defm_"):
        return 512
    return 0


class DeFMEncoder(nn.Module):
    """Frozen-or-trainable DeFM depth encoder with metric-aware preprocessing."""

    def __init__(
        self,
        input_height: int,
        input_width: int,
        output_dim: int,
        *,
        model_name: str,
        backbone_dim: int,
        output_key: str | None = None,
        pretrained: bool = True,
        pretrained_path: str | None = None,
        freeze_backbone: bool = True,
        target_size: int | tuple[int, int] | None = 224,
        patch_size: int | None = 14,
        use_no_bifpn: bool = False,
    ):
        super().__init__()
        if input_height <= 0 or input_width <= 0:
            raise ValueError(f"DeFMViTS14Encoder expects positive input size, got {(input_height, input_width)}.")
        self.input_height = int(input_height)
        self.input_width = int(input_width)
        self.expected_input_size = self.input_height * self.input_width
        self.model_name = model_name
        self.output_key = output_key
        self.pretrained = bool(pretrained)
        self.pretrained_path = pretrained_path
        self.freeze_backbone = bool(freeze_backbone)
        self.target_size = target_size
        self.patch_size = patch_size
        self.use_no_bifpn = bool(use_no_bifpn)
        self.backbone_dim = int(backbone_dim)
        self.backbone: nn.Module | None = None
        self._preprocess_depth_batch = None
        self.forward_batch_size = _resolve_defm_forward_batch_size(self.model_name, self.freeze_backbone)
        self.register_buffer("_defm_mean", torch.tensor([0.248880, 0.495620, 0.492858]).view(1, 3, 1, 1))
        self.register_buffer("_defm_std", torch.tensor([0.139357, 0.271314, 0.297177]).view(1, 3, 1, 1))
        self.proj = nn.Identity() if output_dim == self.backbone_dim else nn.Linear(self.backbone_dim, output_dim)

    def _ensure_backbone(self, device: torch.device) -> None:
        if self.backbone is None:
            create_defm_model, preprocess_depth_batch = _load_defm_runtime()
            backbone = create_defm_model(
                self.model_name,
                pretrained=self.pretrained,
                pretrained_path=self.pretrained_path,
            )
            if self.freeze_backbone:
                backbone.eval()
                for param in backbone.parameters():
                    param.requires_grad_(False)
            self.backbone = backbone.to(device)
            self._preprocess_depth_batch = preprocess_depth_batch
        elif next(self.backbone.parameters()).device != device:
            self.backbone = self.backbone.to(device)
        if self.freeze_backbone and self.backbone is not None:
            self.backbone.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.view(x.shape[0], -1)
        if flat.shape[-1] != self.expected_input_size:
            raise ValueError(
                f"DeFMViTS14Encoder expected flattened input dim {self.expected_input_size}, got {flat.shape[-1]}."
            )
        device = flat.device
        self._ensure_backbone(device)
        assert self.backbone is not None

        backbone_forward = self.backbone
        if self.use_no_bifpn:
            if not hasattr(self.backbone, "forward_no_bifpn"):
                raise ValueError(f"DeFM model {self.model_name} does not expose forward_no_bifpn().")
            backbone_forward = self.backbone.forward_no_bifpn
        if self.forward_batch_size > 0 and flat.shape[0] > self.forward_batch_size:
            features = torch.cat(
                [
                    self._forward_backbone_chunk(
                        flat[start : start + self.forward_batch_size],
                        device,
                        backbone_forward,
                    )
                    for start in range(0, flat.shape[0], self.forward_batch_size)
                ],
                dim=0,
            )
        else:
            features = self._forward_backbone_chunk(flat, device, backbone_forward)
        return self.proj(features)

    def _forward_backbone_chunk(self, flat: torch.Tensor, device: torch.device, backbone_forward) -> torch.Tensor:
        depth = flat.view(flat.shape[0], self.input_height, self.input_width)
        depth_batch = self._preprocess_depth_batch_onnx_safe(depth, device=device)
        if self.freeze_backbone:
            with torch.no_grad():
                features = backbone_forward(depth_batch)
        else:
            features = backbone_forward(depth_batch)
        if self.output_key is not None:
            if not isinstance(features, dict) or self.output_key not in features:
                raise ValueError(
                    f"DeFM model {self.model_name} did not return expected output key {self.output_key!r}."
                )
            features = features[self.output_key]
        if self.freeze_backbone:
            features = features.detach()
        features = features.to(dtype=flat.dtype)
        return features

    def _resolve_target_hw(self) -> tuple[int, int]:
        if self.target_size is None:
            target_h, target_w = self.input_height, self.input_width
        elif isinstance(self.target_size, int):
            target_h, target_w = self.target_size, self.target_size
        else:
            target_h, target_w = self.target_size
        if self.patch_size is not None:
            target_h = (int(target_h) // self.patch_size) * self.patch_size
            target_w = (int(target_w) // self.patch_size) * self.patch_size
        return int(target_h), int(target_w)

    def _preprocess_depth_batch_onnx_safe(self, depth: torch.Tensor, *, device: torch.device) -> torch.Tensor:
        depth = depth.to(device=device, dtype=torch.float32).unsqueeze(1)
        if not torch.onnx.is_in_onnx_export():
            depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth = torch.clamp(depth, min=0.0, max=100.0)
        log_depth = torch.log1p(depth)

        c1 = log_depth / 4.61512051684126
        c2 = torch.clamp(log_depth / 2.302585092994046, min=0.0, max=1.0)

        log_flat = log_depth.flatten(1)
        min_log = log_flat.min(dim=1)[0].view(depth.shape[0], 1, 1, 1)
        max_log = log_flat.max(dim=1)[0].view(depth.shape[0], 1, 1, 1)
        denom = max_log - min_log
        denom_safe = torch.where(denom > 0.0, denom, torch.ones_like(denom))
        c3 = torch.where(denom > 0.0, (log_depth - min_log) / denom_safe, torch.zeros_like(log_depth))

        x = torch.cat([c1, c2, c3], dim=1)
        target_h, target_w = self._resolve_target_hw()
        if x.shape[-2:] != (target_h, target_w):
            x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
        mean = self._defm_mean.to(device=x.device, dtype=x.dtype)
        std = self._defm_std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std


class DeFMViTS14Encoder(DeFMEncoder):
    """Frozen-or-trainable DeFM ViT-S/14 depth encoder with metric-aware preprocessing."""

    def __init__(
        self,
        input_height: int,
        input_width: int,
        output_dim: int,
        *,
        pretrained: bool = True,
        pretrained_path: str | None = None,
        freeze_backbone: bool = True,
        target_size: int | tuple[int, int] | None = 224,
        patch_size: int | None = 14,
    ):
        super().__init__(
            input_height=input_height,
            input_width=input_width,
            output_dim=output_dim,
            model_name="defm_vit_s14",
            backbone_dim=384,
            output_key=None,
            pretrained=pretrained,
            pretrained_path=pretrained_path,
            freeze_backbone=freeze_backbone,
            target_size=target_size,
            patch_size=patch_size,
            use_no_bifpn=False,
        )


class DeFMRegNetY800MFEncoder(DeFMEncoder):
    """Frozen-or-trainable DeFM RegNetY-800MF depth encoder with metric-aware preprocessing."""

    def __init__(
        self,
        input_height: int,
        input_width: int,
        output_dim: int,
        *,
        pretrained: bool = True,
        pretrained_path: str | None = None,
        freeze_backbone: bool = True,
        target_size: int | tuple[int, int] | None = 224,
        patch_size: int | None = None,
    ):
        super().__init__(
            input_height=input_height,
            input_width=input_width,
            output_dim=output_dim,
            model_name="defm_regnet_y_800mf",
            backbone_dim=784,
            output_key="global_backbone",
            pretrained=pretrained,
            pretrained_path=pretrained_path,
            freeze_backbone=freeze_backbone,
            target_size=target_size,
            patch_size=patch_size,
            use_no_bifpn=True,
        )


class DeFMEfficientNetB2Encoder(DeFMEncoder):
    """Frozen-or-trainable DeFM EfficientNet-B2 depth encoder with metric-aware preprocessing."""

    def __init__(
        self,
        input_height: int,
        input_width: int,
        output_dim: int,
        *,
        pretrained: bool = True,
        pretrained_path: str | None = None,
        freeze_backbone: bool = True,
        target_size: int | tuple[int, int] | None = 224,
        patch_size: int | None = None,
    ):
        super().__init__(
            input_height=input_height,
            input_width=input_width,
            output_dim=output_dim,
            model_name="defm_efficientnet_b2",
            backbone_dim=208,
            output_key="global_backbone",
            pretrained=pretrained,
            pretrained_path=pretrained_path,
            freeze_backbone=freeze_backbone,
            target_size=target_size,
            patch_size=patch_size,
            use_no_bifpn=True,
        )


class PerceptionTimeGRU(nn.Module):
    """Temporal GRU encoder over per-step perception vectors."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layer_config: LayerConfig,
        mlp_hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"PerceptionTimeGRU input_dim must be positive, got {input_dim}.")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)

        hidden_dims = list(mlp_hidden_dims) if mlp_hidden_dims is not None else []
        activation = getattr(nn, getattr(layer_config, "encoder_activation", layer_config.activation))()
        dropout = float(layer_config.dropout_prob)

        layers: list[nn.Module] = []
        prev_dim = self.input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, self.hidden_dim))
        self.pre_mlp = nn.Sequential(*layers)

        self.gru = nn.GRU(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)
        self.hidden: torch.Tensor | None = None

    def reset(self, dones: torch.Tensor | None) -> None:
        if self.hidden is None or dones is None:
            return
        done_mask = dones.view(-1).bool()
        if done_mask.any():
            keep = (~done_mask).float().view(1, -1, 1)
            self.hidden = self.hidden * keep

    def step(self, x: torch.Tensor) -> torch.Tensor:
        """Process a single time step (x: [B, input_dim])."""
        if x.ndim != 2:
            raise ValueError(f"PerceptionTimeGRU step expects [B, D], got {x.shape}")
        x = self.pre_mlp(x)
        out, self.hidden = self.gru(x.unsqueeze(1), self.hidden)
        return out[:, -1, :]

    def forward_sequence(self, x_seq: torch.Tensor, dones_seq: torch.Tensor | None = None) -> torch.Tensor:
        """Process a sequence (x_seq: [T, B, input_dim]) with optional done resets."""
        if x_seq.ndim != 3:
            raise ValueError(f"PerceptionTimeGRU sequence expects [T, B, D], got {x_seq.shape}")
        t_steps, batch = x_seq.shape[0], x_seq.shape[1]
        x_proj = self.pre_mlp(x_seq.reshape(-1, x_seq.shape[-1])).view(t_steps, batch, -1)
        device = x_seq.device
        h = torch.zeros(1, batch, self.hidden_dim, device=device)
        outputs = []
        for t in range(t_steps):
            if dones_seq is not None:
                done_mask = dones_seq[t].view(-1).bool()
                if done_mask.any():
                    keep = (~done_mask).float().view(1, -1, 1)
                    h = h * keep
            out, h = self.gru(x_proj[t].unsqueeze(1), h)
            outputs.append(out[:, -1, :])
        return torch.stack(outputs, dim=0)


def build_cnn_layer(
    input_channels: int,
    input_height: int,
    input_width: int,
    hidden_channels: tuple[int, ...] | None,
    kernel_size: int | tuple[int, ...],
    stride: int | tuple[int, ...],
    padding: str | int | tuple[str | int, ...],
    layer_config: LayerConfig,
    flatten_output: bool = True,
):
    """Builds a convolutional neural network layer that works with flattened inputs.

    Parameters
    ----------
        input_channels: int
            Number of input channels
        input_height: int
            Height of input feature maps
        input_width: int
            Width of input feature maps
        hidden_channels: tuple[int, ...]
            Tuple of channel dimensions (last value becomes output channels)
        kernel_size: int or tuple[int, ...]
            Kernel size for convolutions (int or tuple for per-layer values)
        stride: int or tuple[int, ...]
            Stride for convolutions (int or tuple for per-layer values)
        padding: str | int | tuple[str | int, ...]
            Padding mode (int, "same", "valid", or tuple for per-layer values)
        layer_config: dict
            Dictionary containing:
            - activation: Activation function name (e.g., "ReLU")
            - dropout_prob: Dropout probability (default: 0)
            - use_layer_norm: Whether to use layer normalization (default: False)

    Returns
    -------
        CNNWrapper
            The constructed CNN layer wrapped to handle flattened inputs/outputs
    """
    if hidden_channels is None:
        return None
    assert len(hidden_channels) > 0, "hidden_channels must be a non-empty tuple"

    layers: list[nn.Module] = []
    activation = getattr(nn, layer_config.encoder_activation)()
    dropout = layer_config.dropout_prob
    use_layer_norm = layer_config.use_layer_norm

    num_layers = len(hidden_channels)
    # Convert single values to tuples if needed
    if isinstance(kernel_size, int):
        kernel_sizes = (kernel_size,) * num_layers
    else:
        kernel_sizes = kernel_size
        if len(kernel_sizes) != num_layers:
            raise ValueError(
                f"kernel_size tuple length ({len(kernel_sizes)}) must match number of layers ({num_layers})"
            )

    if isinstance(stride, int):
        strides = (stride,) * num_layers
    else:
        strides = stride
        if len(strides) != num_layers:
            raise ValueError(f"stride tuple length ({len(strides)}) must match number of layers ({num_layers})")

    if isinstance(padding, (str, int)):
        paddings = (padding,) * num_layers
    else:
        paddings = padding
        if len(paddings) != num_layers:
            raise ValueError(f"padding tuple length ({len(paddings)}) must match number of layers ({num_layers})")

    # Helper function to get padding value
    def get_padding_value(padding_spec, kernel_size_val):
        if padding_spec == "same":
            return kernel_size_val // 2
        if padding_spec == "valid":
            return 0
        return padding_spec

    # Build layers
    current_in_channels = input_channels
    for layer_idx in range(num_layers):
        current_out_channels = hidden_channels[layer_idx]
        current_kernel_size = kernel_sizes[layer_idx]
        current_stride = strides[layer_idx]
        current_padding = get_padding_value(paddings[layer_idx], current_kernel_size)

        # Add convolution layer
        layers.append(
            nn.Conv2d(
                current_in_channels,
                current_out_channels,
                kernel_size=current_kernel_size,
                stride=current_stride,
                padding=current_padding,
            )
        )

        # Add layer norm, activation and dropout for all layers except the last one
        if layer_idx < num_layers - 1:
            if use_layer_norm:
                layers.append(ImgChLayerNorm(current_out_channels))
            layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout2d(p=dropout))

        current_in_channels = current_out_channels

    cnn_sequential = nn.Sequential(*layers)

    # Wrap with CNNWrapper to handle flattened inputs/outputs
    return CNNWrapper(cnn_sequential, input_channels, input_height, input_width, flatten_output)


class BaseModule(nn.Module):
    def __init__(self, obs_dim_dict, module_config_dict, history_length: dict[str, int]):
        super().__init__()
        self.obs_dim_dict = obs_dim_dict
        self.module_config_dict = module_config_dict
        self.history_length = history_length
        self._calculate_input_dim()
        self._calculate_output_dim()
        self._build_network_layer(self.module_config_dict)

    def _setup_perception_encoder(self, layer_config: LayerConfig) -> int:
        self.perception_encoder = None
        self.perception_input_name = layer_config.perception_input_name
        self.perception_output_dim = 0

        if not self.perception_input_name:
            return 0
        if self.perception_input_name in self.input_dim_dict:
            input_dim = self.input_dim_dict[self.perception_input_name]
        elif self.perception_input_name in self.obs_dim_dict:
            input_dim = self.obs_dim_dict[self.perception_input_name]
        else:
            raise ValueError(f"Unknown perception_input_name: {self.perception_input_name}")
        output_dim = layer_config.perception_output_dim or input_dim
        encoder_type = getattr(layer_config, "perception_encoder_type", "gated_linear")
        if encoder_type == "gru":
            encoder_type = "time_gru"
        if encoder_type == "gated_linear":
            self.perception_encoder = GatedLinearEncoder(input_dim, output_dim)
        elif encoder_type == "attention":
            self.perception_encoder = AttentionLinearEncoder(input_dim, output_dim)
        elif encoder_type == "far_tracking_cnn_small":
            input_height = getattr(layer_config, "perception_input_height", None)
            input_width = getattr(layer_config, "perception_input_width", None)
            if input_height is None or input_width is None:
                raise ValueError(
                    "far_tracking_cnn_small requires perception_input_height and perception_input_width to be set."
                )
            self.perception_encoder = FarTrackingDepthSmallEncoder(
                input_height=int(input_height),
                input_width=int(input_width),
                output_dim=output_dim,
            )
        elif encoder_type == "defm_vit_s14":
            input_height = getattr(layer_config, "perception_input_height", None)
            input_width = getattr(layer_config, "perception_input_width", None)
            if input_height is None or input_width is None:
                raise ValueError("defm_vit_s14 requires perception_input_height and perception_input_width to be set.")
            self.perception_encoder = DeFMViTS14Encoder(
                input_height=int(input_height),
                input_width=int(input_width),
                output_dim=output_dim,
                pretrained=bool(getattr(layer_config, "perception_pretrained", True)),
                pretrained_path=getattr(layer_config, "perception_pretrained_path", None),
                freeze_backbone=bool(getattr(layer_config, "perception_freeze_backbone", True)),
                target_size=getattr(layer_config, "perception_target_size", None),
                patch_size=getattr(layer_config, "perception_patch_size", None),
            )
        elif encoder_type == "defm_regnet_y_800mf":
            input_height = getattr(layer_config, "perception_input_height", None)
            input_width = getattr(layer_config, "perception_input_width", None)
            if input_height is None or input_width is None:
                raise ValueError(
                    "defm_regnet_y_800mf requires perception_input_height and perception_input_width to be set."
                )
            self.perception_encoder = DeFMRegNetY800MFEncoder(
                input_height=int(input_height),
                input_width=int(input_width),
                output_dim=output_dim,
                pretrained=bool(getattr(layer_config, "perception_pretrained", True)),
                pretrained_path=getattr(layer_config, "perception_pretrained_path", None),
                freeze_backbone=bool(getattr(layer_config, "perception_freeze_backbone", True)),
                target_size=getattr(layer_config, "perception_target_size", None),
                patch_size=getattr(layer_config, "perception_patch_size", None),
            )
        elif encoder_type == "defm_efficientnet_b2":
            input_height = getattr(layer_config, "perception_input_height", None)
            input_width = getattr(layer_config, "perception_input_width", None)
            if input_height is None or input_width is None:
                raise ValueError(
                    "defm_efficientnet_b2 requires perception_input_height and perception_input_width to be set."
                )
            self.perception_encoder = DeFMEfficientNetB2Encoder(
                input_height=int(input_height),
                input_width=int(input_width),
                output_dim=output_dim,
                pretrained=bool(getattr(layer_config, "perception_pretrained", True)),
                pretrained_path=getattr(layer_config, "perception_pretrained_path", None),
                freeze_backbone=bool(getattr(layer_config, "perception_freeze_backbone", True)),
                target_size=getattr(layer_config, "perception_target_size", None),
                patch_size=getattr(layer_config, "perception_patch_size", None),
            )
        elif encoder_type == "time_gru":
            # Time-GRU is handled at the actor/critic level; no per-step encoder here.
            self.perception_encoder = None
            self.perception_output_dim = output_dim
            return output_dim
        else:
            raise ValueError(f"Unknown perception_encoder_type: {encoder_type}")
        self.perception_output_dim = output_dim
        return output_dim

    def _calculate_input_dim(self):
        # calculate input dimension and input slices
        self.input_dim = 0
        self.input_dim_dict = {}
        self.input_indices_dict = {}

        current_index = 0
        for each_input in self.module_config_dict.input_dim:
            if each_input in self.obs_dim_dict:
                # atomic observation type
                # Note: obs_dim_dict already includes history, so we don't multiply by history_length
                input_dim = self.obs_dim_dict[each_input]
                self.input_dim += input_dim
                self.input_dim_dict[each_input] = input_dim
                self.input_indices_dict[each_input] = slice(current_index, current_index + input_dim)
                current_index += input_dim

            elif isinstance(each_input, (int, float)):
                # direct numeric input
                input_dim = int(each_input)
                self.input_dim += input_dim
                self.input_dim_dict[each_input] = input_dim
                self.input_indices_dict[each_input] = slice(current_index, current_index + input_dim)
                current_index += input_dim

            else:
                current_function_name = inspect.currentframe().f_code.co_name
                raise ValueError(f"{current_function_name} - Unknown input type: {each_input}")

    def _calculate_output_dim(self):
        # calculate output dimension based on the output specifications
        self.output_dim = 0
        for each_output in self.module_config_dict.output_dim:
            if isinstance(each_output, (int, float)):
                self.output_dim += each_output
            else:
                current_function_name = inspect.currentframe().f_code.co_name
                raise ValueError(f"{current_function_name} - Unknown output type: {each_output}")

    def _build_network_layer(self, module_config: ModuleConfig):
        layer_type = module_config.type
        layer_config = module_config.layer_config
        if layer_type == "MLP":
            if layer_config.perception_input_name:
                raise ValueError("perception_input_name is not supported for MLP modules.")
            self.module = build_mlp_layer(
                self.input_dim,
                layer_config.hidden_dims,
                self.output_dim,
                layer_config,
            )
        elif layer_type == "FlowMLP":
            if layer_config.perception_input_name:
                raise ValueError("perception_input_name is not supported for FlowMLP modules.")
            self.encoder = None
            self.module = ConditionalFlowMLP(
                self.input_dim,
                self.output_dim,
                layer_config,
            )
        elif layer_type == "CNNEncoder":
            perception_output_dim = self._setup_perception_encoder(layer_config)
            perception_input_dim = 0 if layer_config.extra_input_to_hidden else perception_output_dim
            self.encoder = build_cnn_layer(
                layer_config.input_channels,
                layer_config.input_height,
                layer_config.input_width,
                layer_config.hidden_channels,
                layer_config.kernel_size,
                layer_config.stride,
                layer_config.padding,
                layer_config,
                flatten_output=True,
            )
            encoder_output_dim = self.encoder.output_size
            mlp_input_dim = sum(self.input_dim_dict[each_input] for each_input in layer_config.module_input_name)
            self.module = build_mlp_layer(
                mlp_input_dim + encoder_output_dim + perception_input_dim,
                layer_config.hidden_dims,
                self.output_dim,
                layer_config,
            )
        elif layer_type == "MLPEncoder":
            perception_output_dim = self._setup_perception_encoder(layer_config)
            perception_input_dim = 0 if layer_config.extra_input_to_hidden else perception_output_dim
            encoder_output_dim = (
                layer_config.encoder_output_dim
                if layer_config.encoder_hidden_dims is not None
                else self.input_dim_dict[layer_config.encoder_input_name]
            )
            self.encoder = build_mlp_layer(
                self.input_dim_dict[layer_config.encoder_input_name],
                layer_config.encoder_hidden_dims,
                encoder_output_dim,
                layer_config,
            )
            mlp_input_dim = sum(self.input_dim_dict[each_input] for each_input in layer_config.module_input_name)
            self.module = build_mlp_layer(
                mlp_input_dim + encoder_output_dim + perception_input_dim,
                layer_config.hidden_dims,
                self.output_dim,
                layer_config,
            )
        elif layer_type == "TransformerEncoder":
            perception_output_dim = self._setup_perception_encoder(layer_config)
            perception_input_dim = 0 if layer_config.extra_input_to_hidden else perception_output_dim
            if layer_config.encoder_num_steps is None:
                raise ValueError("encoder_num_steps must be set for TransformerEncoder modules.")
            encoder_input_dim = self.input_dim_dict[layer_config.encoder_input_name]
            obs_dim = layer_config.encoder_obs_dim
            if obs_dim is None:
                if encoder_input_dim % layer_config.encoder_num_steps != 0:
                    raise ValueError(
                        "encoder_input_dim must be divisible by encoder_num_steps for TransformerEncoder modules."
                    )
                obs_dim = encoder_input_dim // layer_config.encoder_num_steps
            encoder_output_dim = layer_config.encoder_output_dim or layer_config.transformer_latent_dim
            self.encoder = TargetPoseTransformer(
                obs_dim=obs_dim,
                num_steps=layer_config.encoder_num_steps,
                latent_dim=encoder_output_dim,
                num_layers=layer_config.transformer_num_layers,
                num_heads=layer_config.transformer_num_heads,
                ff_dim=layer_config.transformer_ff_dim,
                dropout=layer_config.transformer_dropout,
                activation=layer_config.encoder_activation,
                pooling=layer_config.transformer_pooling,
            )
            mlp_input_dim = sum(self.input_dim_dict[each_input] for each_input in layer_config.module_input_name)
            self.module = build_mlp_layer(
                mlp_input_dim + encoder_output_dim + perception_input_dim,
                layer_config.hidden_dims,
                self.output_dim,
                layer_config,
            )
        elif layer_type == "TransformerObsTokenEncoder":
            perception_output_dim = self._setup_perception_encoder(layer_config)
            perception_input_dim = 0 if layer_config.extra_input_to_hidden else perception_output_dim
            if layer_config.encoder_num_steps is None:
                raise ValueError("encoder_num_steps must be set for TransformerObsTokenEncoder modules.")
            if layer_config.encoder_obs_token_name is None:
                raise ValueError("encoder_obs_token_name must be set for TransformerObsTokenEncoder modules.")
            encoder_input_dim = self.input_dim_dict[layer_config.encoder_input_name]
            obs_token_dim = self.input_dim_dict[layer_config.encoder_obs_token_name]
            target_obs_dim = layer_config.encoder_obs_dim
            if target_obs_dim is None:
                if encoder_input_dim % layer_config.encoder_num_steps != 0:
                    raise ValueError(
                        "encoder_input_dim must be divisible by encoder_num_steps for TransformerObsTokenEncoder modules."
                    )
                target_obs_dim = encoder_input_dim // layer_config.encoder_num_steps
            encoder_output_dim = layer_config.encoder_output_dim or layer_config.transformer_latent_dim
            self.encoder = ObsTargetPoseTransformer(
                obs_dim=obs_token_dim,
                target_obs_dim=target_obs_dim,
                num_steps=layer_config.encoder_num_steps,
                latent_dim=encoder_output_dim,
                num_layers=layer_config.transformer_num_layers,
                num_heads=layer_config.transformer_num_heads,
                ff_dim=layer_config.transformer_ff_dim,
                dropout=layer_config.transformer_dropout,
                activation=layer_config.encoder_activation,
                pooling=layer_config.transformer_pooling,
                layer_config=layer_config,
                encoder_hidden_dims=layer_config.encoder_hidden_dims,
            )
            mlp_input_dim = sum(self.input_dim_dict[each_input] for each_input in layer_config.module_input_name)
            self.module = build_mlp_layer(
                mlp_input_dim + encoder_output_dim + perception_input_dim,
                layer_config.hidden_dims,
                self.output_dim,
                layer_config,
            )
        elif layer_type == "TerrainTransformerObsTokenEncoder":
            if layer_config.encoder_obs_token_name is None:
                raise ValueError("encoder_obs_token_name must be set for TerrainTransformerObsTokenEncoder modules.")
            if not layer_config.perception_input_name:
                raise ValueError("perception_input_name must be set for TerrainTransformerObsTokenEncoder modules.")

            proprio_dim = self.input_dim_dict[layer_config.encoder_obs_token_name]
            perception_output_dim = self._setup_perception_encoder(layer_config)
            if perception_output_dim == 0:
                raise ValueError("perception_input_name must be set for TerrainTransformerObsTokenEncoder modules.")
            depth_dim = perception_output_dim

            target_obs_dim = None
            if layer_config.encoder_input_name:
                if layer_config.encoder_num_steps is None:
                    raise ValueError("encoder_num_steps must be set when encoder_input_name is provided.")
                encoder_input_dim = self.input_dim_dict[layer_config.encoder_input_name]
                target_obs_dim = layer_config.encoder_obs_dim
                if target_obs_dim is None:
                    if encoder_input_dim % layer_config.encoder_num_steps != 0:
                        raise ValueError(
                            "encoder_input_dim must be divisible by encoder_num_steps for "
                            "TerrainTransformerObsTokenEncoder modules."
                        )
                    target_obs_dim = encoder_input_dim // layer_config.encoder_num_steps

            encoder_output_dim = layer_config.encoder_output_dim or layer_config.transformer_latent_dim
            self.encoder = TerrainObsTokenTransformer(
                proprio_dim=proprio_dim,
                depth_dim=depth_dim,
                target_obs_dim=target_obs_dim,
                num_steps=layer_config.encoder_num_steps or 0,
                latent_dim=encoder_output_dim,
                num_layers=layer_config.transformer_num_layers,
                num_heads=layer_config.transformer_num_heads,
                ff_dim=layer_config.transformer_ff_dim,
                dropout=layer_config.transformer_dropout,
                activation=layer_config.encoder_activation,
                pooling=layer_config.transformer_pooling,
                layer_config=layer_config,
                encoder_hidden_dims=layer_config.encoder_hidden_dims,
            )
            mlp_input_dim = sum(self.input_dim_dict[each_input] for each_input in layer_config.module_input_name)
            self.module = build_mlp_layer(
                mlp_input_dim + encoder_output_dim,
                layer_config.hidden_dims,
                self.output_dim,
                layer_config,
            )
        elif layer_type == "MLPPerceptionEncoder":
            perception_output_dim = self._setup_perception_encoder(layer_config)
            if perception_output_dim == 0:
                raise ValueError("perception_input_name must be set for MLPPerceptionEncoder modules.")
            perception_input_dim = 0 if layer_config.extra_input_to_hidden else perception_output_dim
            self.encoder = None
            mlp_input_dim = sum(self.input_dim_dict[each_input] for each_input in layer_config.module_input_name)
            self.module = build_mlp_layer(
                mlp_input_dim + perception_input_dim,
                layer_config.hidden_dims,
                self.output_dim,
                layer_config,
            )
        elif layer_type == "FlowMLPPerceptionEncoder":
            perception_output_dim = self._setup_perception_encoder(layer_config)
            if perception_output_dim == 0:
                raise ValueError("perception_input_name must be set for FlowMLPPerceptionEncoder modules.")
            perception_input_dim = 0 if layer_config.extra_input_to_hidden else perception_output_dim
            self.encoder = None
            mlp_input_dim = sum(self.input_dim_dict[each_input] for each_input in layer_config.module_input_name)
            self.module = ConditionalFlowMLP(
                mlp_input_dim + perception_input_dim,
                self.output_dim,
                layer_config,
            )
        else:
            raise NotImplementedError(f"Unsupported layer type: {layer_type}")

    def forward(self, policy_input: torch.Tensor, extra_input: torch.Tensor | None = None) -> torch.Tensor:
        if extra_input is not None:
            if not getattr(self.module, "supports_extra_input", False):
                raise ValueError("Extra input provided but module is not configured for extra_input_to_hidden.")
            return self.module(policy_input, extra_input=extra_input)
        return self.module(policy_input)

    @property
    def supports_flow_matching(self) -> bool:
        return bool(getattr(self.module, "supports_flow_matching", False))

    def flow_matching_loss(
        self,
        policy_input: torch.Tensor,
        target_action: torch.Tensor,
        *,
        extra_input: torch.Tensor | None = None,
        loss_fn=F.mse_loss,
    ) -> torch.Tensor:
        if not self.supports_flow_matching:
            raise ValueError("flow_matching_loss requested for a non-flow module.")
        return self.module.flow_matching_loss(
            policy_input,
            target_action,
            extra_input=extra_input,
            loss_fn=loss_fn,
        )
