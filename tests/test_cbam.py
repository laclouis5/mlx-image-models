import mlx.core as mx
import numpy as np
import pytest
import timm.layers.cbam as timm_m
import torch

import mlx_im.layers.cbam as mlx_m

from . import utils as U
from . import weights as W


@pytest.mark.parametrize("bias", [True, False])
def test_channel_attn(bias):
    torch.manual_seed(42)

    x_mlx = U.sample_mlx_array_2d(shape=(1, 32, 48, 128))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_m.ChannelAttn(channels=128, mlp_bias=bias)
    mod_timm = timm_m.ChannelAttn(channels=128, mlp_bias=bias)

    W.transfer_weights(mod_timm, mod_mlx)

    out_timm = mod_timm(x_torch)
    out_mlx = mod_mlx(x_mlx)
    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-5
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


@pytest.mark.parametrize("bias", [True, False])
def test_light_channel_attn(bias):
    torch.manual_seed(42)

    x_mlx = U.sample_mlx_array_2d(shape=(1, 32, 48, 128))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_m.LightChannelAttn(channels=128, mlp_bias=bias)
    mod_timm = timm_m.LightChannelAttn(channels=128, mlp_bias=bias)

    W.transfer_weights(mod_timm, mod_mlx)

    out_timm = mod_timm(x_torch)
    out_mlx = mod_mlx(x_mlx)
    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-5
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


@pytest.mark.parametrize("kernel_size", [1, 3, 5, 7])
def test_spatial_attn(kernel_size):
    torch.manual_seed(42)

    x_mlx = U.sample_mlx_array_2d(shape=(1, 32, 48, 128))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_m.SpatialAttn(kernel_size=kernel_size)
    mod_timm = timm_m.SpatialAttn(kernel_size=kernel_size)

    W.transfer_weights(mod_timm, mod_mlx)

    out_timm = mod_timm(x_torch)
    out_mlx = mod_mlx(x_mlx)
    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-5
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


@pytest.mark.parametrize("kernel_size", [1, 3, 5, 7])
def test_lights_patial_attn(kernel_size):
    torch.manual_seed(42)

    x_mlx = U.sample_mlx_array_2d(shape=(1, 32, 48, 128))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_m.LightSpatialAttn(kernel_size=kernel_size)
    mod_timm = timm_m.LightSpatialAttn(kernel_size=kernel_size)

    W.transfer_weights(mod_timm, mod_mlx)

    out_timm = mod_timm(x_torch)
    out_mlx = mod_mlx(x_mlx)
    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-5
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


@pytest.mark.parametrize("kernel_size", [1, 3, 5, 7])
@pytest.mark.parametrize("bias", [True, False])
def test_cbam(kernel_size, bias):
    torch.manual_seed(42)

    x_mlx = U.sample_mlx_array_2d(shape=(1, 32, 48, 128))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_m.CbamModule(
        channels=128, spatial_kernel_size=kernel_size, mlp_bias=bias
    )
    mod_timm = timm_m.CbamModule(
        channels=128, spatial_kernel_size=kernel_size, mlp_bias=bias
    )

    W.transfer_weights(mod_timm, mod_mlx)

    out_timm = mod_timm(x_torch)
    out_mlx = mod_mlx(x_mlx)
    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-5
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


@pytest.mark.parametrize("kernel_size", [1, 3, 5, 7])
@pytest.mark.parametrize("bias", [True, False])
def test_light_cbam(kernel_size, bias):
    torch.manual_seed(42)

    x_mlx = U.sample_mlx_array_2d(shape=(1, 32, 48, 128))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_m.LightCbamModule(
        channels=128, spatial_kernel_size=kernel_size, mlp_bias=bias
    )
    mod_timm = timm_m.LightCbamModule(
        channels=128, spatial_kernel_size=kernel_size, mlp_bias=bias
    )

    W.transfer_weights(mod_timm, mod_mlx)

    out_timm = mod_timm(x_torch)
    out_mlx = mod_mlx(x_mlx)
    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-5
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"
