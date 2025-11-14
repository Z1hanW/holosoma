import numpy as np
import pytest
import torch

from holosoma.utils.torch_utils import (
    get_axis_params,
    normalize,
    split_and_pad_trajectories,
    to_torch,
    torch_rand_float,
    unpad_trajectories,
)


@pytest.fixture
def tensors_and_dones():
    # Shape: [time=6, envs=2, dim=1]
    tensors = torch.tensor(
        [
            [[1], [10]],
            [[5], [14]],
            [[9], [18]],
            [[13], [22]],
            [[17], [26]],
            [[21], [30]],
        ],
        dtype=torch.float32,
    )

    # Done flags for each env
    # Shape: [time=6, envs=2]
    dones = torch.tensor(
        [
            [0, 0],
            [0, 0],
            [0, 1],
            [1, 0],
            [0, 1],
            [0, 0],
        ],
        dtype=torch.bool,
    )

    return tensors, dones


def test_split_and_pad_trajectories(tensors_and_dones):
    tensors, dones = tensors_and_dones

    padded, masks = split_and_pad_trajectories(tensors, dones)

    assert torch.allclose(
        padded.squeeze(-1),
        torch.tensor(
            [
                [1.0, 17.0, 10.0, 22.0, 30.0],
                [5.0, 21.0, 14.0, 26.0, 0.0],
                [9.0, 0.0, 18.0, 0.0, 0.0],
                [13.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )
    assert torch.allclose(
        masks,
        torch.tensor(
            [
                [True, True, True, True, True],
                [True, True, True, True, False],
                [True, False, True, False, False],
                [True, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
            ]
        ),
    )

    # Check dimensions
    assert padded.dim() == 3  # [T_padded, N_trajectories, D]
    assert masks.shape == padded[..., 0].shape


def test_unpad_matches_original(tensors_and_dones):
    tensors, dones = tensors_and_dones
    padded, masks = split_and_pad_trajectories(tensors, dones)
    unpadded = unpad_trajectories(padded, masks)
    assert torch.allclose(unpadded, tensors)


def test_normalize():
    # Test normalization of vector
    v = torch.tensor([3.0, 4.0, 0.0], dtype=torch.float32)
    norm_v = normalize(v)
    assert torch.allclose(torch.norm(norm_v), torch.tensor(1.0), atol=1e-6)

    # Test with zero vector - should return zero vector
    zero = torch.zeros(3, dtype=torch.float32)
    norm_zero = normalize(zero)
    expected = torch.zeros(3, dtype=torch.float32)
    assert torch.allclose(norm_zero, expected, atol=1e-6)


def test_to_torch():
    # Test numpy array conversion with explicit dtype
    np_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    torch_tensor = to_torch(np_array)
    assert isinstance(torch_tensor, torch.Tensor)
    expected = torch.tensor(np_array, dtype=torch.float32, device=torch_tensor.device)
    assert torch.allclose(torch_tensor, expected)

    # Test list conversion
    py_list = [1.0, 2.0, 3.0]
    torch_tensor = to_torch(py_list)
    assert isinstance(torch_tensor, torch.Tensor)
    expected = torch.tensor(py_list, dtype=torch.float32, device=torch_tensor.device)
    assert torch.allclose(torch_tensor, expected)

    # Test device placement
    torch_tensor = to_torch(np_array, device="cpu")
    assert torch_tensor.device.type == "cpu"


def test_torch_rand_float():
    # Test range
    lower, upper = -1.0, 1.0
    shape = (1000, 1)
    device = "cpu"
    rand_tensor = torch_rand_float(lower, upper, shape, device)
    assert torch.all(rand_tensor >= lower)
    assert torch.all(rand_tensor <= upper)
    assert rand_tensor.shape == shape
    assert rand_tensor.device.type == device


def test_get_axis_params():
    """Test get_axis_params function for different axis indices and values."""
    # Test default parameters (3D vector)
    params = get_axis_params(value=1.0, axis_idx=0)
    assert params == [1.0, 0.0, 0.0]

    # Test with different axis
    params = get_axis_params(value=2.0, axis_idx=1)
    assert params == [0.0, 2.0, 0.0]

    # Test with different dimensions
    params = get_axis_params(value=4.0, axis_idx=1, n_dims=4)
    assert params == [0.0, 4.0, 0.0, 0.0]

    # Test with different dtype
    params = get_axis_params(value=5.0, axis_idx=0, dtype=np.float32)
    assert all(isinstance(x, np.float32) for x in params)

    # Test error case
    with pytest.raises(AssertionError):
        get_axis_params(value=1.0, axis_idx=3, n_dims=3)  # axis_idx out of bounds
