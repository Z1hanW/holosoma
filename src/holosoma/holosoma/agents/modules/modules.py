from __future__ import annotations

import inspect
import math

import torch
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
    """Sinusoidal positional encoding for short token sequences."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}.")
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1], : x.shape[2]]


def _build_transformer_activation(name: str) -> nn.Module:
    lower = name.lower() if isinstance(name, str) else ""
    if lower == "relu":
        return nn.ReLU()
    if lower == "gelu":
        return nn.GELU()
    if lower == "elu":
        return nn.ELU()
    if lower == "tanh":
        return nn.Tanh()
    return nn.GELU()


class TokenSelfAttentionBlock(nn.Module):
    """Small ONNX-opset-13-compatible transformer block for short token sequences."""

    def __init__(self, latent_dim: int, num_heads: int, ff_dim: int, dropout: float, activation: str):
        super().__init__()
        if latent_dim % num_heads != 0:
            raise ValueError(f"latent_dim={latent_dim} must be divisible by num_heads={num_heads}.")
        self.latent_dim = int(latent_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.latent_dim // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.norm1 = nn.LayerNorm(latent_dim)
        self.qkv = nn.Linear(latent_dim, 3 * latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(latent_dim)
        self.ff = nn.Sequential(
            nn.Linear(latent_dim, ff_dim),
            _build_transformer_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, latent_dim),
        )
        self.ff_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        y = self.norm1(x)
        qkv = self.qkv(y).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        y = torch.matmul(attn, v)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.latent_dim)
        x = x + self.out_dropout(self.out_proj(y))
        x = x + self.ff_dropout(self.ff(self.norm2(x)))
        return x


class TokenHSITransformer(nn.Module):
    """Transformer over output, proprio/state, heightmap, and optional target tokens."""

    def __init__(
        self,
        state_dim: int,
        heightmap_dim: int,
        target_obs_dim: int | None,
        num_target_steps: int,
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
        if latent_dim % num_heads != 0:
            raise ValueError(f"transformer_latent_dim={latent_dim} must be divisible by num_heads={num_heads}.")
        if pooling not in ("first", "mean"):
            raise ValueError(f"Unsupported transformer_pooling={pooling!r}; expected 'first' or 'mean'.")

        hidden_dims = encoder_hidden_dims or []
        self.target_obs_dim = target_obs_dim
        self.num_target_steps = int(num_target_steps)
        self.latent_dim = int(latent_dim)
        self.pooling = pooling

        self.output_token = nn.Parameter(torch.zeros(1, 1, latent_dim))
        self.state_encoder = build_mlp_layer(
            state_dim,
            hidden_dims,
            latent_dim,
            layer_config,
            activation_name=activation,
        )
        self.heightmap_encoder = build_mlp_layer(
            heightmap_dim,
            hidden_dims,
            latent_dim,
            layer_config,
            activation_name=activation,
        )
        self.target_encoder = None
        if target_obs_dim is not None and self.num_target_steps > 0:
            self.target_encoder = build_mlp_layer(
                target_obs_dim,
                hidden_dims,
                latent_dim,
                layer_config,
                activation_name=activation,
            )

        max_len = 3 + (self.num_target_steps if self.target_encoder is not None else 0)
        self.positional_encoding = PositionalEncoding(latent_dim, max_len=max_len)
        self.encoder = nn.ModuleList(
            [
                TokenSelfAttentionBlock(
                    latent_dim=latent_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_layers)
            ]
        )
        nn.init.trunc_normal_(self.output_token, std=0.02)

    def forward(
        self,
        state_token: torch.Tensor,
        heightmap_token: torch.Tensor,
        target_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = state_token.shape[0]
        seq_parts = [
            self.output_token.expand(batch_size, -1, -1),
            self.state_encoder(state_token).unsqueeze(1),
            self.heightmap_encoder(heightmap_token).unsqueeze(1),
        ]

        if self.target_encoder is not None:
            if target_tokens is None:
                raise ValueError("target_tokens must be provided when encoder_input_name is configured.")
            expected_dim = self.num_target_steps * self.target_obs_dim
            if target_tokens.shape[-1] != expected_dim:
                raise ValueError(f"Expected target_tokens last dim {expected_dim}, got {target_tokens.shape[-1]}.")
            target_tokens = target_tokens.view(batch_size, self.num_target_steps, self.target_obs_dim)
            target_flat = target_tokens.reshape(-1, self.target_obs_dim)
            target_embed = self.target_encoder(target_flat).view(batch_size, self.num_target_steps, self.latent_dim)
            seq_parts.append(target_embed)

        seq = torch.cat(seq_parts, dim=1)
        seq = self.positional_encoding(seq)
        for encoder_layer in self.encoder:
            seq = encoder_layer(seq)
        if self.pooling == "mean":
            return seq.mean(dim=1)
        return seq[:, 0, :]


class TokenHSIBranch(nn.Module):
    """Slices a concatenated PPO observation and applies the TokenHSI encoder/head."""

    def __init__(
        self,
        input_dim_dict: dict,
        input_indices_dict: dict,
        output_dim: int,
        layer_config: LayerConfig,
    ):
        super().__init__()
        self.input_indices_dict = input_indices_dict
        self.state_input_name = layer_config.encoder_obs_token_name
        self.heightmap_input_name = layer_config.perception_input_name
        self.target_input_name = layer_config.encoder_input_name
        self.module_input_names = tuple(layer_config.module_input_name)

        if not self.state_input_name:
            raise ValueError("TokenHSI requires layer_config.encoder_obs_token_name.")
        if not self.heightmap_input_name:
            raise ValueError("TokenHSI requires layer_config.perception_input_name.")
        if self.state_input_name not in input_dim_dict:
            raise ValueError(f"Unknown TokenHSI state input: {self.state_input_name}")
        if self.heightmap_input_name not in input_dim_dict:
            raise ValueError(f"Unknown TokenHSI heightmap input: {self.heightmap_input_name}")

        target_obs_dim = None
        num_target_steps = 0
        if self.target_input_name:
            if self.target_input_name not in input_dim_dict:
                raise ValueError(f"Unknown TokenHSI target input: {self.target_input_name}")
            if layer_config.encoder_num_steps is None:
                raise ValueError("TokenHSI encoder_num_steps must be set when encoder_input_name is set.")
            target_input_dim = input_dim_dict[self.target_input_name]
            num_target_steps = int(layer_config.encoder_num_steps)
            target_obs_dim = layer_config.encoder_obs_dim
            if target_obs_dim is None:
                if target_input_dim % num_target_steps != 0:
                    raise ValueError(
                        "TokenHSI target input dim must be divisible by encoder_num_steps "
                        "when encoder_obs_dim is not set."
                    )
                target_obs_dim = target_input_dim // num_target_steps

        latent_dim = layer_config.encoder_output_dim or layer_config.transformer_latent_dim
        self.encoder = TokenHSITransformer(
            state_dim=input_dim_dict[self.state_input_name],
            heightmap_dim=input_dim_dict[self.heightmap_input_name],
            target_obs_dim=target_obs_dim,
            num_target_steps=num_target_steps,
            latent_dim=latent_dim,
            num_layers=layer_config.transformer_num_layers,
            num_heads=layer_config.transformer_num_heads,
            ff_dim=layer_config.transformer_ff_dim,
            dropout=layer_config.transformer_dropout,
            activation=layer_config.encoder_activation,
            pooling=layer_config.transformer_pooling,
            layer_config=layer_config,
            encoder_hidden_dims=layer_config.encoder_hidden_dims,
        )

        passthrough_dim = 0
        reserved = {self.state_input_name, self.heightmap_input_name, self.target_input_name}
        for input_name in self.module_input_names:
            if input_name in reserved:
                raise ValueError(f"TokenHSI module_input_name duplicates token input {input_name!r}.")
            if input_name not in input_dim_dict:
                raise ValueError(f"Unknown TokenHSI passthrough input: {input_name}")
            passthrough_dim += input_dim_dict[input_name]
        self.head = build_mlp_layer(latent_dim + passthrough_dim, layer_config.hidden_dims, output_dim, layer_config)

    def _slice(self, policy_input: torch.Tensor, input_name: str) -> torch.Tensor:
        return policy_input[..., self.input_indices_dict[input_name]]

    def forward(self, policy_input: torch.Tensor) -> torch.Tensor:
        state_token = self._slice(policy_input, self.state_input_name)
        heightmap_token = self._slice(policy_input, self.heightmap_input_name)
        target_tokens = self._slice(policy_input, self.target_input_name) if self.target_input_name else None
        encoded = self.encoder(state_token, heightmap_token, target_tokens)

        if self.module_input_names:
            passthrough = [self._slice(policy_input, input_name) for input_name in self.module_input_names]
            encoded = torch.cat([encoded, *passthrough], dim=-1)
        return self.head(encoded)


def build_mlp_layer(
    input_dim,
    hidden_dims,
    output_dim,
    layer_config,
    activation_name: str | None = None,
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
    if hidden_dims is None:
        return None

    layers = []
    activation = getattr(nn, activation_name or layer_config.activation)()
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
            self.module = build_mlp_layer(
                self.input_dim,
                layer_config.hidden_dims,
                self.output_dim,
                layer_config,
            )
        elif layer_type == "TokenHSI":
            self.module = TokenHSIBranch(
                self.input_dim_dict,
                self.input_indices_dict,
                self.output_dim,
                layer_config,
            )
        elif layer_type == "CNNEncoder":
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
                mlp_input_dim + encoder_output_dim,
                layer_config.hidden_dims,
                self.output_dim,
                layer_config,
            )
        elif layer_type == "MLPEncoder":
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
                mlp_input_dim + encoder_output_dim,
                layer_config.hidden_dims,
                self.output_dim,
                layer_config,
            )
        else:
            raise NotImplementedError(f"Unsupported layer type: {layer_type}")

    def forward(self, policy_input):
        # Only forward the MLP layer
        return self.module(policy_input)
