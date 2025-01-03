import mlx.core as mx
import numpy as np
import pytest
import timm.layers.gather_excite as timm_m
import torch

import mlx_im.layers.gather_excite as mlx_m

from . import utils as U
from . import weights as W


@pytest.mark.parametrize("extra_params", [True, False])
@pytest.mark.parametrize("extent,feat_size", [(0, 3), (2, None)])
@pytest.mark.parametrize("use_mlp", [True, False])
@pytest.mark.parametrize("add_maxpool", [True, False])
def test_ge(extra_params, extent, feat_size, use_mlp, add_maxpool):
    pytest.fail("Bug in MLX AvgPool2d without count_include_pad")

    torch.manual_seed(42)

    x_mlx = U.sample_mlx_array_2d(shape=(2, 32, 48, 128))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_m.GatherExcite(
        channels=128,
        feat_size=feat_size,
        extra_params=extra_params,
        extent=extent,
        use_mlp=use_mlp,
        add_maxpool=add_maxpool,
    )
    mod_timm = timm_m.GatherExcite(
        channels=128,
        feat_size=feat_size,
        extra_params=extra_params,
        extent=extent,
        use_mlp=use_mlp,
        add_maxpool=add_maxpool,
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
