import pytest
from pydantic import ValidationError

from holosoma.config_types.algo import LayerConfig


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("flow_integration_steps", True),
        ("flow_integration_steps", 3.7),
        ("flow_integration_steps", "4"),
        ("flow_integration_steps", 0),
        ("flow_integration_steps", 4097),
        ("flow_train_noise_std", True),
        ("flow_train_noise_std", "1.0"),
        ("flow_train_noise_std", -0.1),
        ("flow_train_noise_std", float("nan")),
        ("flow_train_noise_std", float("inf")),
        ("flow_train_noise_std", 10**400),
        ("flow_train_noise_std", 1.0e19),
        ("flow_time_epsilon", False),
        ("flow_time_epsilon", "0.1"),
        ("flow_time_epsilon", -0.1),
        ("flow_time_epsilon", 0.5),
        ("flow_time_epsilon", float("nan")),
        ("flow_time_epsilon", float("inf")),
        ("flow_inference_noise_std", True),
        ("flow_inference_noise_std", "0.0"),
        ("flow_inference_noise_std", -0.1),
        ("flow_inference_noise_std", float("nan")),
        ("flow_inference_noise_std", float("inf")),
        ("flow_inference_noise_std", 1.0e19),
    ],
)
def test_layer_config_rejects_invalid_flow_values(field, value):
    with pytest.raises(ValidationError, match=field):
        LayerConfig(**{field: value})


def test_layer_config_accepts_valid_flow_boundaries_and_integer_float_inputs():
    config = LayerConfig(
        flow_integration_steps=4096,
        flow_train_noise_std=1.0e18,
        flow_time_epsilon=0.49,
        flow_inference_noise_std=1.0e18,
    )

    assert config.flow_integration_steps == 4096
    assert config.flow_train_noise_std == 1.0e18
    assert config.flow_time_epsilon == 0.49
    assert config.flow_inference_noise_std == 1.0e18
