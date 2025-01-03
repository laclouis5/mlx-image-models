import mlx.core as mx
import numpy as np
import pytest
import timm.layers.pool2d_same as timm_m

import mlx_im.layers.pool2d_same as mlx_m

from . import utils as U


@pytest.mark.parametrize("pool_type", ["avg", "max"])
@pytest.mark.parametrize("kernel_size", [(2, 2), (3, 3), (4, 4)])
def test_pool2d_same(pool_type, kernel_size):
    x_mlx = U.sample_mlx_array_2d(shape=(2, 32, 48, 32))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_m.create_pool2d(pool_type, kernel_size, padding="same")
    mod_timm = timm_m.create_pool2d(pool_type, kernel_size, padding="same")

    out_timm = mod_timm(x_torch)
    out_mlx = mod_mlx(x_mlx)
    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-5
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"
