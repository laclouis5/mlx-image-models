import mlx.core as mx
import numpy as np
import pytest
import timm.layers.filter_response_norm as timm_m

import mlx_im.layers.filter_response_norm as mlx_m

from . import utils as U


def test_inv_instance_rms():
    x_mlx = U.sample_mlx_array_2d(shape=(2, 32, 48, 32))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    out_mlx = mlx_m.inv_instance_rms(x_mlx)
    out_timm = timm_m.inv_instance_rms(x_torch)

    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-5
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


@pytest.mark.parametrize("apply_act", [True, False])
@pytest.mark.parametrize("rms", [True, False])
def test_filter_resp_norm_tlu_2d(apply_act, rms):
    x_mlx = U.sample_mlx_array_2d(shape=(2, 32, 48, 64))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_m.FilterResponseNormTlu2d(
        num_features=64, apply_act=apply_act, rms=rms
    )
    mod_timm = timm_m.FilterResponseNormTlu2d(
        num_features=64, apply_act=apply_act, rms=rms
    )

    out_mlx = mod_mlx(x_mlx)
    out_timm = mod_timm(x_torch)

    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-5
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


@pytest.mark.parametrize("apply_act", [True, False])
@pytest.mark.parametrize("rms", [True, False])
def test_filter_resp_norm_act_2d(apply_act, rms):
    x_mlx = U.sample_mlx_array_2d(shape=(2, 32, 48, 64))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_m.FilterResponseNormAct2d(
        num_features=64, apply_act=apply_act, rms=rms
    )
    mod_timm = timm_m.FilterResponseNormAct2d(
        num_features=64, apply_act=apply_act, rms=rms
    )

    out_timm = mod_timm(x_torch)
    out_mlx = mod_mlx(x_mlx)
    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-5
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"
