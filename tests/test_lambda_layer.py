import mlx.core as mx
import numpy as np
import pytest
import timm.layers.lambda_layer as timm_m
import torch

import mlx_im.layers.lambda_layer as mlx_m

from . import utils as U
from . import weights as W


def test_rel_pos_indices():
    torch.manual_seed(42)

    mod_mlx = mlx_m.rel_pos_indices
    mod_timm = timm_m.rel_pos_indices

    out_timm = mod_timm((32, 48))
    out_mlx = mod_mlx((32, 48))
    mx.eval(out_mlx)

    out_mlx = np.array(out_mlx)
    out_timm = out_timm.detach().numpy()

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-5
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


@pytest.mark.parametrize("dim_out", [128, 256])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("r", [9, None])
def test_lambda_layer(dim_out, stride, r):
    torch.manual_seed(42)

    if dim_out != 128:
        pytest.xfail("Timm does not supports dim_out != dim.")

    x_mlx = U.sample_mlx_array_2d(shape=(2, 32, 48, 128))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_m.LambdaLayer(
        128, dim_out=dim_out, feat_size=(32, 48), stride=stride, r=r
    )
    mod_timm = timm_m.LambdaLayer(
        128, dim_out=dim_out, feat_size=(32, 48), stride=stride, r=r
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
