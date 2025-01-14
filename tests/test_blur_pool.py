import mlx.core as mx
import numpy as np
import pytest
import timm.layers.blur_pool as timm_m

import mlx_im.layers.blur_pool as mlx_m

from . import utils as U


@pytest.mark.parametrize("aa_layer", ["avg", "blur", "blurpc"])
def test_blur_pool(aa_layer):
    if aa_layer == "blur":
        pytest.xfail("Not supported (TODO: pad mode 'reflect').")

    x_mlx = U.sample_mlx_array_2d(shape=(2, 32, 48, 32))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_m.create_aa(aa_layer)
    mod_timm = timm_m.create_aa(aa_layer)

    out_mlx = mod_mlx(x_mlx)
    out_timm = mod_timm(x_torch)

    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-5
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"
