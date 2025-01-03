import mlx.core as mx
import numpy as np
import pytest
from timm.layers import adaptive_avgmax_pool as timm_avgmax_pool

from mlx_im.layers.adaptive_avgmax_pool import SelectAdaptivePool2d

from . import utils as U


@pytest.mark.parametrize("size", [1, 2, 4, 8])
@pytest.mark.parametrize("pool_type", ["avg", "max", "avgmax", "catavgmax"])
def test_adaptative_pooling(size, pool_type):
    x_mlx = U.sample_mlx_array_2d(shape=(2, 32, 48, 3))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    pool_mlx = SelectAdaptivePool2d(output_size=size, pool_type=pool_type)
    pool_timm = timm_avgmax_pool.SelectAdaptivePool2d(
        output_size=size, pool_type=pool_type
    )

    out_timm = pool_timm(x_torch)
    out_mlx = pool_mlx(x_mlx)
    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-5
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


@pytest.mark.parametrize(
    "pool_type", ["fast_avg", "fast_max", "fast_avgmax", "fast_catavgmax"]
)
def test_fast_adaptative_pooling(pool_type):
    x_mlx = U.sample_mlx_array_2d(shape=(2, 32, 48, 3))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    pool_mlx = SelectAdaptivePool2d(output_size=1, pool_type=pool_type)
    pool_timm = timm_avgmax_pool.SelectAdaptivePool2d(
        output_size=1, pool_type=pool_type
    )

    out_timm = pool_timm(x_torch)
    out_mlx = pool_mlx(x_mlx)
    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-5
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"
