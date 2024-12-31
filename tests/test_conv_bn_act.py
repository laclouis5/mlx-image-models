import mlx.core as mx
import numpy as np
import timm.layers.conv_bn_act as timm_m

import mlx_im.layers.conv_bn_act as mlx_m

from . import utils as U


def test_conv_bn_act():
    x_mlx = U.sample_mlx_array_2d(shape=(1, 512, 768, 32))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_m.ConvNormAct(
        in_channels=32,
        out_channels=16,
        kernel_size=3,
        padding="same",
        bias=True,
        aa_layer="blurc",
    )
    mod_timm = timm_m.ConvNormAct(
        in_channels=32,
        out_channels=16,
        kernel_size=3,
        padding="same",
        bias=True,
        aa_layer="blurc",
    )

    mod_mlx.conv.weight = mx.array(mod_timm.conv.weight.detach().numpy()).transpose(
        0, 2, 3, 1
    )
    mod_mlx.conv.bias = mx.array(mod_timm.conv.bias.detach().numpy())

    out_mlx = mod_mlx(x_mlx)
    out_timm = mod_timm(x_torch)

    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-5
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"
