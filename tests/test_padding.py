import mlx.core as mx
import numpy as np
import pytest
from timm.layers import padding as timm_padding

from mlx_im.layers.padding import pad_same

from . import utils as U


@pytest.mark.parametrize("kernel_size", [(1, 1), (3, 3), (5, 5), (7, 7)])
@pytest.mark.parametrize("stride", [(1, 1), (2, 2)])
@pytest.mark.parametrize("dilation", [(1, 1), (2, 2)])
def test_pad_same(kernel_size, stride, dilation):
    x_mlx = U.sample_mlx_array_2d(shape=(1, 512, 768, 3))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    out_mlx = pad_same(x_mlx, kernel_size, stride, dilation)
    out_timm = timm_padding.pad_same(x_torch, kernel_size, stride, dilation)

    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-9
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"
