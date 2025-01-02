import mlx.core as mx
import numpy as np
import pytest
import timm.layers.mlp as timm_m
import torch

import mlx_im.layers.mlp as mlx_m

from . import utils as U
from . import weights as W


def test_grn():
    torch.manual_seed(42)

    x_mlx = U.sample_mlx_array_2d(shape=(1, 32, 48, 128))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_m.GlobalResponseNorm(128)
    mod_timm = timm_m.GlobalResponseNorm(128, channels_last=False)

    W.transfer_weights(mod_timm, mod_mlx)

    out_timm = mod_timm(x_torch)
    out_mlx = mod_mlx(x_mlx)
    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-5
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


@pytest.mark.parametrize("use_conv", [True, False])
@pytest.mark.parametrize("bias", [True, False])
def test_mlp(use_conv, bias):
    torch.manual_seed(42)

    if use_conv:
        x_mlx = U.sample_mlx_array_2d(shape=(1, 32, 48, 128))
        x_torch = U.mlx_to_torch_2d(x_mlx)
    else:
        x_mlx = mx.random.normal(shape=(1, 32, 128))
        x_torch = torch.from_numpy(np.array(x_mlx))

    mod_mlx = mlx_m.Mlp(128, use_conv=use_conv, bias=bias)
    mod_timm = timm_m.Mlp(128, use_conv=use_conv, bias=bias)

    W.transfer_weights(mod_timm, mod_mlx)

    out_timm = mod_timm(x_torch)
    out_mlx = mod_mlx(x_mlx)
    mx.eval(out_mlx)

    if use_conv:
        out_mlx = U.mlx_to_numpy_2d(out_mlx)
        out_timm = U.torch_to_numpy_2d(out_timm)
    else:
        out_mlx = np.array(out_mlx)
        out_timm = out_timm.detach().numpy()

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-5
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("use_conv", [True, False])
@pytest.mark.parametrize("gate_last", [True, False])
def test_glu_mlp(bias, use_conv, gate_last):
    torch.manual_seed(42)

    if use_conv:
        x_mlx = U.sample_mlx_array_2d(shape=(1, 32, 48, 128))
        x_torch = U.mlx_to_torch_2d(x_mlx)
    else:
        x_mlx = mx.random.normal(shape=(1, 32, 128))
        x_torch = torch.from_numpy(np.array(x_mlx))

    mod_mlx = mlx_m.GluMlp(128, bias=bias, use_conv=use_conv, gate_last=gate_last)
    mod_timm = timm_m.GluMlp(128, bias=bias, use_conv=use_conv, gate_last=gate_last)

    W.transfer_weights(mod_timm, mod_mlx)

    out_timm = mod_timm(x_torch)
    out_mlx = mod_mlx(x_mlx)
    mx.eval(out_mlx)

    if use_conv:
        out_mlx = U.mlx_to_numpy_2d(out_mlx)
        out_timm = U.torch_to_numpy_2d(out_timm)
    else:
        out_mlx = np.array(out_mlx)
        out_timm = out_timm.detach().numpy()

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-5
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


@pytest.mark.parametrize("bias", [True, False])
def test_swi_glu(bias):
    torch.manual_seed(42)

    x_mlx = mx.random.normal(shape=(1, 32, 128))
    x_torch = torch.from_numpy(np.array(x_mlx))

    mod_mlx = mlx_m.SwiGLU(128, bias=bias)
    mod_timm = timm_m.SwiGLU(128, bias=bias)

    W.transfer_weights(mod_timm, mod_mlx)

    out_timm = mod_timm(x_torch)
    out_mlx = mod_mlx(x_mlx)
    mx.eval(out_mlx)

    out_mlx = np.array(out_mlx)
    out_timm = out_timm.detach().numpy()

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-5
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


@pytest.mark.parametrize("bias", [True, False])
def test_gated_mlp(bias):
    torch.manual_seed(42)

    x_mlx = mx.random.normal(shape=(1, 32, 128))
    x_torch = torch.from_numpy(np.array(x_mlx))

    mod_mlx = mlx_m.GatedMlp(128, bias=bias)
    mod_timm = timm_m.GatedMlp(128, bias=bias)

    W.transfer_weights(mod_timm, mod_mlx)

    out_timm = mod_timm(x_torch)
    out_mlx = mod_mlx(x_mlx)
    mx.eval(out_mlx)

    out_mlx = np.array(out_mlx)
    out_timm = out_timm.detach().numpy()

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-5
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


@pytest.mark.parametrize("bias", [True, False])
def test_conv_mlp(bias):
    torch.manual_seed(42)

    x_mlx = U.sample_mlx_array_2d(shape=(1, 32, 48, 128))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_m.ConvMlp(128, bias=bias)
    mod_timm = timm_m.ConvMlp(128, bias=bias)

    W.transfer_weights(mod_timm, mod_mlx)

    out_timm = mod_timm(x_torch)
    out_mlx = mod_mlx(x_mlx)
    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-5
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


@pytest.mark.parametrize("use_conv", [True, False])
@pytest.mark.parametrize("bias", [True, False])
def test_grn_mlp(use_conv, bias):
    if not use_conv:
        pytest.xfail("GRN only makes sense for 2D inputs")
        
    torch.manual_seed(42)

    if use_conv:
        x_mlx = U.sample_mlx_array_2d(shape=(1, 32, 48, 128))
        x_torch = U.mlx_to_torch_2d(x_mlx)
    else:
        x_mlx = mx.random.normal(shape=(1, 32, 128))
        x_torch = torch.from_numpy(np.array(x_mlx))

    mod_mlx = mlx_m.GlobalResponseNormMlp(128, use_conv=use_conv, bias=bias)
    mod_timm = timm_m.GlobalResponseNormMlp(128, use_conv=use_conv, bias=bias)

    W.transfer_weights(mod_timm, mod_mlx)

    out_timm = mod_timm(x_torch)
    out_mlx = mod_mlx(x_mlx)
    mx.eval(out_mlx)

    if use_conv:
        out_mlx = U.mlx_to_numpy_2d(out_mlx)
        out_timm = U.torch_to_numpy_2d(out_timm)
    else:
        out_mlx = np.array(out_mlx)
        out_timm = out_timm.detach().numpy()

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-5
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"
