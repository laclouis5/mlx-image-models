import mlx.core as mx
import numpy as np
import pytest
import torch
import torch.nn as timm_m

import mlx_im.layers.mlx_layers as mlx_m

from . import utils as U
from . import weights as W


@pytest.mark.parametrize(
    "kernel_size,stride,padding",
    [
        (2, 2, 0),
        (3, 1, 1),
    ],
)
@pytest.mark.parametrize("count_include_pad", [True, False])
def test_avg_pool_2d_patched(kernel_size, stride, padding, count_include_pad):
    pytest.fail("Bug in MLX AvgPool2d without count_include_pad")

    torch.manual_seed(42)

    x_mlx = mx.ones((1, 4, 6, 1))
    x_torch = torch.from_numpy(np.array(x_mlx)).permute(0, 3, 1, 2)

    mod_mlx = mlx_m.AvgPool2d(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        count_include_pad=count_include_pad,
    )
    mod_timm = timm_m.AvgPool2d(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        count_include_pad=count_include_pad,
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
