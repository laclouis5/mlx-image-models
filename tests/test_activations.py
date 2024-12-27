import mlx.core as mx
import numpy as np
import pytest
import timm.layers.create_act as timm_acts

from mlx_im.layers.create_act import (
    create_act_layer,
    get_act_fn,
)

from . import utils as U


@pytest.fixture
def sample_mlx_array_2d():
    return U.sample_mlx_array_2d(shape=(1, 512, 768, 3))


@pytest.mark.parametrize("act_name", list(timm_acts._ACT_FN_DEFAULT.keys()))
def test_act_fn(sample_mlx_array_2d, act_name):
    x_mlx = sample_mlx_array_2d
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
