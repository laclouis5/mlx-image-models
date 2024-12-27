import mlx.core as mx
import numpy as np
import pytest
import timm.layers.create_act as timm_acts

from mlx_im.layers.create_act import (
    create_act_layer,
    get_act_fn,
)

from . import utils as U


@pytest.mark.parametrize("act_name", list(timm_acts._ACT_FN_DEFAULT.keys()))
def test_act_fn(act_name):
    x_mlx = U.sample_mlx_array_2d(shape=(1, 512, 768, 3))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mlx_act = get_act_fn(name=act_name)
    timm_act = timm_acts.get_act_fn(name=act_name)

    out_mlx = mlx_act(x_mlx)
    out_timm = timm_act(x_torch)

    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = out_timm.numpy()

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-5
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


@pytest.mark.parametrize("act_name", list(timm_acts._ACT_LAYER_DEFAULT.keys()))
def test_act_layer(act_name):
    x_mlx = U.sample_mlx_array_2d(shape=(1, 512, 768, 3))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mlx_act = create_act_layer(name=act_name)
    timm_act = timm_acts.create_act_layer(name=act_name)

    out_mlx = mlx_act(x_mlx)
    out_timm = timm_act(x_torch)

    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-5
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"
