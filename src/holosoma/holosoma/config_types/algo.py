from __future__ import annotations

from dataclasses import field
from typing import Any, List, Union

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class OptimizerConfig:
    """Configuration for optimizer settings."""

    _target_: str
    """Target optimizer class (e.g., torch.optim.AdamW)."""

    weight_decay: float = 0.001
    """Weight decay parameter for the optimizer."""


@dataclass(frozen=True)
class LayerConfig:
    """Configuration for neural network layer settings."""

    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    """List of hidden layer dimensions."""

    activation: str = "ELU"
    """Activation function name."""

    dropout_prob: float = 0.0
    """Dropout probability."""

    use_layer_norm: bool = False
    """Whether to use layer normalization."""

    encoder_activation: str = "ELU"
    """Activation function name for encoder layers."""

    encoder_output_dim: int | None = None
    """Output dimension for encoder. Only used for encoder modules."""

    encoder_hidden_dims: List[int] | None = None
    """Hidden dimensions for encoder. Only used for encoder modules."""

    encoder_input_name: str = ""
    """Input name for encoder. Only used for encoder modules."""

    encoder_obs_token_name: str | None = None
    """Optional input name for a transformer "current-obs" token."""

    perception_input_name: str = ""
    """Optional input name for perception encoder."""

    perception_output_dim: int | None = None
    """Output dimension for perception encoder."""

    perception_encoder_type: str = "gated_linear"
    """Perception encoder type: 'gated_linear', 'attention', 'far_tracking_cnn_small', 'defm_vit_s14', 'defm_regnet_y_800mf', or 'time_gru'."""

    perception_input_height: int | None = None
    """Optional input height for structured perception encoders."""

    perception_input_width: int | None = None
    """Optional input width for structured perception encoders."""

    perception_pretrained: bool = True
    """Whether to load pretrained weights for external perception encoders."""

    perception_pretrained_path: str | None = None
    """Optional local checkpoint path for external perception encoders."""

    perception_freeze_backbone: bool = True
    """Freeze external perception backbones and train only projection layers when supported."""

    perception_target_size: int | tuple[int, int] | None = None
    """Optional target size used by external perception preprocessors."""

    perception_patch_size: int | None = None
    """Optional patch-size alignment used by external perception preprocessors."""

    input_channels: int = 1
    """Number of input channels. Only used for CNN modules."""

    input_height: int = 1
    """Height of input feature maps. Only used for CNN modules."""

    input_width: int = 1
    """Width of input feature maps. Only used for CNN modules."""

    hidden_channels: tuple[int, ...] | None = None
    """Hidden channel dimensions. Only used for CNN modules."""

    kernel_size: int | tuple[int, ...] = 3
    """Kernel size for convolutions. Only used for CNN modules."""

    stride: int | tuple[int, ...] = 1
    """Stride for convolutions. Only used for CNN modules."""

    padding: str | int | tuple[str | int, ...] = "same"
    """Padding mode for convolutions. Only used for CNN modules."""

    module_input_name: tuple[str, ...] = ()
    """Input names for module. Only used for encoder modules."""

    encoder_num_steps: int | None = None
    """Sequence length for transformer-style encoders."""

    encoder_obs_dim: int | None = None
    """Per-step feature dimension for transformer-style encoders."""

    transformer_latent_dim: int = 256
    """Latent dimension for transformer-style encoders."""

    transformer_num_layers: int = 2
    """Number of transformer encoder layers."""

    transformer_num_heads: int = 2
    """Number of attention heads in the transformer encoder."""

    transformer_ff_dim: int = 512
    """Feed-forward dimension inside transformer encoder layers."""

    transformer_dropout: float = 0.0
    """Dropout for transformer encoder layers."""

    transformer_pooling: str = "first"
    """Pooling for transformer encoder output: 'first' or 'mean'."""

    extra_input_to_hidden: bool = False
    """Whether to add an extra input projection to the first hidden layer."""


@dataclass(frozen=True)
class ModuleConfig:
    """Configuration for neural network modules."""

    type: str
    """Module type (e.g., MLP)."""

    input_dim: List[str] = field(default_factory=list)
    """Input dimension specification."""

    output_dim: List[str | int] = field(default_factory=list)
    """Output dimension specification."""

    layer_config: LayerConfig = field(default_factory=LayerConfig)
    """Layer configuration settings."""

    min_noise_std: float | None = None
    """Minimum noise standard deviation."""

    min_mean_noise_std: float | None = None
    """Minimum mean noise standard deviation."""


@dataclass(frozen=True)
class DistillationConfig:
    """Configuration for behavior cloning/distillation from a teacher policy."""

    enabled: bool = False
    """Enable distillation loss against a teacher policy."""

    teacher_checkpoint: str | None = None
    """Path to teacher checkpoint (local path or wandb:// URI)."""

    loss_coef: float = 1.0
    """Weight for the distillation loss added to actor loss."""

    mode: str = "mse"
    """Distillation mode: "mse" (legacy) or "dagger" (VideoMimic-style BC)."""

    policy_to_clone: str | list[str] | None = None
    """Teacher checkpoint path(s) for dagger (aliases teacher_checkpoint)."""

    teacher_obs_keys: list[str] | str | None = None
    """Observation keys to feed the teacher policy (defaults to actor_obs keys)."""

    teacher_use_stochastic_actions: bool = False
    """Use sampled teacher actions for DAgger labels/rollout mixing instead of deterministic teacher means."""

    bc_loss_coef: float | None = None
    """Behavior cloning loss coefficient for dagger. Defaults to loss_coef if unset."""

    clip_teacher_actions: bool = False
    """Whether to clip teacher actions when computing BC loss."""

    clip_actions_threshold: float = 100.0
    """Absolute action clip value for BC loss when clip_teacher_actions is enabled."""

    take_teacher_actions: bool = False
    """Whether to step the environment with teacher actions instead of student actions."""

    teacher_action_mix_ratio: float = 0.0
    """Per-step env mix ratio for teacher actions in DAgger rollout (0.0=student only, 1.0=teacher only)."""

    teacher_action_mix_ratio_start: float | None = None
    """Optional initial teacher-action rollout mix ratio for linear scheduling."""

    teacher_action_mix_ratio_end: float | None = None
    """Optional final teacher-action rollout mix ratio for linear scheduling."""

    teacher_action_mix_ratio_end_iteration: int = -1
    """Iteration where teacher-action rollout mix reaches ``teacher_action_mix_ratio_end``."""

    schedule_name: str | None = None
    """Optional human-readable name for the active distillation schedule."""

    schedule_notes: str | None = None
    """Optional free-form notes describing the distillation curriculum."""

    teacher_compat_profile: str | None = None
    """Optional name for the teacher compatibility profile applied by the launcher."""

    teacher_compat_notes: str | None = None
    """Optional notes describing any teacher compatibility adjustments or remaining mismatches."""

    teacher_perception_preset: str | None = None
    """Optional perception preset name used only for the teacher policy (e.g. heightmap)."""

    teacher_perception_obs_key: str | None = None
    """Optional observation key used to feed teacher-only perception into the teacher policy."""

    critic_perception_preset: str | None = None
    """Optional perception preset name used only for the critic policy (e.g. heightmap)."""

    critic_perception_obs_key: str | None = None
    """Optional observation key used to feed critic-only perception into the critic policy."""

    switch_to_rl_after: int = -1
    """Iteration to switch off BC loss (set to 0 or negative to disable)."""

    use_multi_teacher: bool = False
    """Whether to use multiple teacher checkpoints (requires policy_to_clone list)."""

    multi_teacher_select_obs_var: str = "teacher_checkpoint_index"
    """Observation key used to select teacher policy when use_multi_teacher is True."""

    # Far-tracking DepthDistillationPPO parity controls.
    ppo_start_epoch: int = -1
    """Epoch to start mixing PPO loss in dagger mode (-1 disables schedule)."""

    dagger_end_epoch: int = -1
    """Epoch where PPO contribution saturates in dagger mode (-1 disables schedule)."""

    ppo_target_coeff: float = 0.9
    """Final PPO blend coefficient reached at ``dagger_end_epoch`` in scheduled PPO+DAgger mode."""

    ppo_start_coeff: float = 0.0
    """Initial PPO blend coefficient at ``ppo_start_epoch`` in scheduled PPO+DAgger mode."""

    ppo_start_noise_std: float | None = None
    """Optional max policy noise std enforced while PPO first enters scheduled PPO+DAgger mode."""

    ppo_start_noise_std_until_coeff: float = 0.1
    """Keep ``ppo_start_noise_std`` active until PPO blend coefficient exceeds this value."""

    ppo_schedule_step_epochs: int = 0
    """Optional PPO/DAgger step interval; values > 0 use staircase blending instead of a linear ramp."""

    dagger_loss_coef: float = 10.0
    """Scale on dagger/distillation loss term in scheduled PPO+DAgger mode."""

    distill_loss_type: str = "mse"
    """DAgger loss type: 'mse' or 'huber'."""

    dagger_ignore_zero_teacher_actions: bool = True
    """Ignore samples where teacher action is exactly zero across all dims."""

    dagger_ignore_external_goal_samples: bool = False
    """Ignore BC samples collected under external sparse-goal episodes."""

    dagger_ignore_episode_initial_steps: int = 0
    """Ignore BC samples from the first N episode steps (useful when reset states are outside teacher distribution)."""

    dagger_match_std: bool = False
    """Match policy std against teacher std in BC loss (legacy behavior)."""

    strict_teacher_load: bool = True
    """Fail fast on teacher architecture/obs mismatch instead of fallback loading."""

    fixed_bc_eval_num_samples: int = 4096
    """Number of fixed teacher-labeled samples to cache for deterministic BC evaluation (0 disables)."""

    fixed_bc_eval_log_interval: int = 1
    """Log fixed-set BC evaluation metrics every N learning iterations."""



@dataclass(frozen=True)
class PPOModuleDictConfig:
    """Configuration for PPO module dictionary."""

    actor: ModuleConfig
    """Actor module configuration."""

    critic: ModuleConfig
    """Critic module configuration."""


@dataclass(frozen=True)
class PPOConfig:
    """Configuration for PPO algorithm."""

    module_dict: PPOModuleDictConfig
    """PPO module configurations (actor, critic)."""

    num_learning_epochs: int = 8
    """Number of learning epochs per update."""

    num_mini_batches: int = 4
    """Number of mini-batches per epoch."""

    clip_param: float = 0.2
    """PPO clipping parameter."""

    gamma: float = 0.99
    """Discount factor for future rewards."""

    lam: float = 0.95
    """GAE lambda parameter."""

    value_loss_coef: float = 1.0
    """Value loss coefficient."""

    entropy_coef: float = 0.01
    """Entropy coefficient for exploration."""

    actor_learning_rate: float = 1e-5
    """Learning rate for actor network."""

    actor_optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(_target_="torch.optim.AdamW"))
    """Actor optimizer configuration."""

    critic_learning_rate: float = 1e-5
    """Learning rate for critic network."""

    critic_optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(_target_="torch.optim.AdamW"))
    """Critic optimizer configuration."""

    max_grad_norm: float = 1.0
    """Maximum gradient norm for clipping."""

    schedule: str = "adaptive"
    """Learning rate schedule type."""

    desired_kl: float = 0.01
    """Desired KL divergence for adaptive learning rate."""

    use_symmetry: bool = False
    """Whether to use symmetry in training."""

    symmetry_actor_coef: float = 1.0
    """Symmetry coefficient for actor."""

    symmetry_critic_coef: float = 0.0
    """Symmetry coefficient for critic."""

    num_steps_per_env: int = 24
    """Number of steps per environment."""

    save_interval: int = 1000
    """Interval for saving model checkpoints."""

    load_optimizer: bool = True
    """Whether to load optimizer state."""

    init_noise_std: float = 0.8
    """Initial noise standard deviation."""

    num_learning_iterations: int = 1000000
    """Total number of learning iterations."""

    init_at_random_ep_len: bool = True
    """Whether to initialize at random episode length."""

    eval_callbacks: Any = None
    """Evaluation callbacks configuration."""

    normalize_actor_obs: bool = False
    """Whether to apply empirical normalization to actor observations."""

    normalize_critic_obs: bool = False
    """Whether to apply empirical normalization to critic observations."""

    obs_normalizer_eps: float = 1e-2
    """Epsilon for observation normalization."""

    obs_normalizer_until: int | None = None
    """Optional cap on number of samples used to update the obs normalizer."""

    max_actor_learning_rate: float | None = None
    min_actor_learning_rate: float | None = None
    max_critic_learning_rate: float | None = None
    min_critic_learning_rate: float | None = None

    distill: DistillationConfig = field(default_factory=DistillationConfig)
    """Optional teacher distillation configuration."""


@dataclass(frozen=True)
class FastSACConfig:
    num_learning_iterations: int = 25000
    """total timesteps of the experiments"""

    critic_learning_rate: float = 3e-4
    """the learning rate of the critic"""

    actor_learning_rate: float = 3e-4
    """the learning rate for the actor"""

    alpha_learning_rate: float = 3e-4
    """the learning rate for the alpha"""

    buffer_size: int = 1024
    """the replay memory buffer size per environment"""

    num_steps: int = 1
    """the number of steps to use for the multi-step return"""

    gamma: float = 0.97
    """the discount factor gamma"""

    tau: float = 0.125
    """target smoothing coefficient (default: 0.005)"""

    batch_size: int = 8192
    """the batch size of sample from the replay memory"""

    learning_starts: int = 10
    """timestep to start learning"""

    policy_frequency: int = 4
    """the frequency of training policy (delayed)"""

    num_updates: int = 8
    """the number of updates to perform per step"""

    target_entropy_ratio: float = 0.0
    """the ratio of the target entropy to the number of actions"""

    num_atoms: int = 101
    """the number of atoms"""

    v_min: float = -20.0
    """the minimum value of the support"""

    v_max: float = 20.0
    """the maximum value of the support"""

    critic_hidden_dim: int = 768
    """the hidden dimension of the critic network"""

    actor_hidden_dim: int = 512
    """the hidden dimension of the actor network"""

    use_symmetry: bool = False
    """whether to use symmetry"""

    alpha_init: float = 0.001
    """the initial value of the alpha"""

    use_autotune: bool = True
    """whether to use autotune for the alpha"""

    use_tanh: bool = True
    """whether to use tanh for the action"""

    log_std_max: float = 0.0
    """the maximum value of the log std"""

    log_std_min: float = -5.0
    """the minimum value of the log std"""

    compile: bool = True
    """whether to use torch.compile."""

    obs_normalization: bool = True
    """whether to enable observation normalization"""

    use_layer_norm: bool = True
    """whether to use layer normalization"""

    num_q_networks: int = 2
    """number of Q-networks to ensemble"""

    max_grad_norm: float = 0.0
    """the maximum gradient norm"""

    amp: bool = True
    """whether to use amp"""

    amp_dtype: str = "bf16"
    """the dtype of the amp"""

    weight_decay: float = 0.001
    """the weight decay of the optimizer"""

    save_interval: int = 1000
    """the interval to save the model"""

    logging_interval: int = 100
    """the interval to log the metrics"""

    encoder_obs_key: str = "perception_obs"
    """the key of the encoder observation. only valid if use_cnn_encoder is True"""

    encoder_obs_shape: tuple[int, int, int] = (1, 13, 9)
    """the shape of the encoder observation. only valid if use_cnn_encoder is True"""

    use_cnn_encoder: bool = False
    """whether to use CNN for the encoder"""

    actor_obs_keys: List[str] = field(default_factory=lambda: ["actor_obs"])
    critic_obs_keys: List[str] = field(default_factory=lambda: ["critic_obs"])


@dataclass(frozen=True)
class PPOAlgoConfig:
    """Configuration for algorithm wrapper."""

    _target_: str
    """Target algorithm class."""

    _recursive_: bool
    """Whether to recursively instantiate."""

    config: PPOConfig
    """Algorithm-specific configuration."""


@dataclass(frozen=True)
class FastSACAlgoConfig:
    """Configuration for algorithm wrapper."""

    _target_: str
    """Target algorithm class."""

    _recursive_: bool
    """Whether to recursively instantiate."""

    config: FastSACConfig
    """Algorithm-specific configuration."""


AlgoInitConfig = Union[PPOConfig, FastSACConfig]

AlgoConfig = Union[PPOAlgoConfig, FastSACAlgoConfig]
