import mlx.core as mx
import numpy as np
import pytest
import timm.layers.create_norm_act as timm_m
import torch.nn as nn_torch
from mlx import nn as nn_mlx

import mlx_im.layers.create_norm_act as mlx_m

from . import utils as U


@pytest.mark.parametrize("layer_name", list(mlx_m._NORM_ACT_MAP.keys()))
def test_norm_act(layer_name: str):
    if "abn" in layer_name:
        pytest.xfail("ABN not supported for now")
    elif layer_name == "layernorm":
        pytest.xfail("Not sure about the implementation")

    x_mlx = U.sample_mlx_array_2d(shape=(1, 512, 768, 32))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_m.create_norm_act_layer(
        layer_name=layer_name, num_features=32, act_layer=nn_mlx.ReLU
    )
    mod_timm = timm_m.create_norm_act_layer(
        layer_name=layer_name, num_features=32, act_layer=nn_torch.ReLU
    )

    out_mlx = mod_mlx(x_mlx)
    out_timm = mod_timm(x_torch)

    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-7
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"
