import mlx.core as mx
import numpy as np
import pytest
import timm.layers.squeeze_excite as timm_m
import torch

import mlx_im.layers.squeeze_excite as mlx_m

from . import utils as U
from . import weights as W


@pytest.mark.parametrize("add_maxpool", [True, False])
@pytest.mark.parametrize("bias", [True, False])
def test_se_module(add_maxpool, bias):
    torch.manual_seed(42)

    x_mlx = U.sample_mlx_array_2d(shape=(2, 32, 48, 128))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_m.SEModule(128, add_maxpool=add_maxpool, bias=bias)
    mod_timm = timm_m.SEModule(128, add_maxpool=add_maxpool, bias=bias)

    W.transfer_weights(mod_timm, mod_mlx)

    out_timm = mod_timm(x_torch)
    out_mlx = mod_mlx(x_mlx)
    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-4
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


@pytest.mark.parametrize("add_maxpool", [True, False])
def test_eff_se_module(add_maxpool):
    torch.manual_seed(42)

    x_mlx = U.sample_mlx_array_2d(shape=(2, 32, 48, 128))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_m.EffectiveSEModule(128, add_maxpool=add_maxpool)
    mod_timm = timm_m.EffectiveSEModule(128, add_maxpool=add_maxpool)

    W.transfer_weights(mod_timm, mod_mlx)

    out_timm = mod_timm(x_torch)
    out_mlx = mod_mlx(x_mlx)
    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-4
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


@pytest.mark.parametrize("bias", [True, False])
def test_sq_cl(bias):
    pytest.xfail("Module not currently used.")
    torch.manual_seed(42)

    x_mlx = U.sample_mlx_array_2d(shape=(2, 32, 48, 128))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_m.SqueezeExciteCl(128, bias=bias)
    mod_timm = timm_m.SqueezeExciteCl(128, bias=bias)

    W.transfer_weights(mod_timm, mod_mlx)

    out_timm = mod_timm(x_torch)
    out_mlx = mod_mlx(x_mlx)
    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-4
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"
