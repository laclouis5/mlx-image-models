import mlx.core as mx
import numpy as np
import pytest
import timm.layers.create_norm as timm_norms
import torch

from mlx_im.layers.create_norm import create_norm_layer

from . import utils as U


@pytest.mark.parametrize(
    "norm_name",
    [
        "batchnorm",
        "batchnorm2d",
        "groupnorm",
        "groupnorm1",
        "layernorm2d",
        "rmsnorm2d",
        "frozenbatchnorm2d",
    ],
)
def test_norm_2d(norm_name):
    if "rms" in norm_name:
        pytest.xfail("Timm RMSNorm buggy")

    # NOTE: Channel size of 64 to be a multiple of num_groups of group norm.
    x_mlx = U.sample_mlx_array_2d(shape=(1, 512, 768, 64))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mlx_norm = create_norm_layer(layer_name=norm_name, num_features=64)
    timm_norm = timm_norms.create_norm_layer(layer_name=norm_name, num_features=64)

    out_mlx = mlx_norm(x_mlx)
    out_timm = timm_norm(x_torch)

    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-7
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


@pytest.mark.parametrize("norm_name", ["batchnorm1d"])
def test_norm_1d(norm_name):
    x_mlx = mx.random.normal(
        shape=(1, 16, 128), key=mx.array([42, 42], dtype=mx.uint32)
    )
    x_torch = torch.from_numpy(np.array(x_mlx)).permute(0, 2, 1)

    mlx_norm = create_norm_layer(layer_name=norm_name, num_features=128)
    timm_norm = timm_norms.create_norm_layer(layer_name=norm_name, num_features=128)

    out_mlx = mlx_norm(x_mlx).transpose(0, 2, 1)
    out_timm = timm_norm(x_torch)

    mx.eval(out_mlx)

    out_mlx = np.array(out_mlx)
    out_timm = out_timm.detach().numpy()

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-7
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


@pytest.mark.parametrize("norm_name", ["layernorm", "rmsnorm"])
def test_norm(norm_name):
    if "rms" in norm_name:
        pytest.xfail("Timm RMSNorm buggy")

    x_mlx = mx.random.normal(
        shape=(1, 16, 128), key=mx.array([42, 42], dtype=mx.uint32)
    )
    x_torch = torch.from_numpy(np.array(x_mlx))

    mlx_norm = create_norm_layer(layer_name=norm_name, num_features=128)
    timm_norm = timm_norms.create_norm_layer(layer_name=norm_name, num_features=128)

    out_mlx = mlx_norm(x_mlx)
    out_timm = timm_norm(x_torch)

    mx.eval(out_mlx)

    out_mlx = np.array(out_mlx)
    out_timm = out_timm.detach().numpy()

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-5
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"
