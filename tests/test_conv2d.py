import mlx.core as mx
import numpy as np
import pytest
import torch
from timm.layers.create_conv2d import create_conv2d as create_conv2d_torch

from mlx_im.layers.create_conv2d import create_conv2d

from . import utils as U


@pytest.mark.parametrize(
    "kernel_size,padding,stride,dilation",
    [
        (1, 0, 1, 1),  # 1x1 conv
        (3, 1, 1, 1),  # 3x3 conv
        (3, 0, 1, 1),  # 3x3 conv without padding
        (3, 1, 2, 1),  # 3x3s2 conv
        (3, 1, 1, 2),  # 3x3 dilated conv
        (2, "same", 1, 1),  # Weird SAME padding case
    ],
)
@pytest.mark.parametrize("groups", [1, 2, 4])
@pytest.mark.parametrize("bias", [True, False])
def test_conv2d_same(kernel_size, padding, stride, dilation, groups, bias):
    x_mlx = U.sample_mlx_array_2d(shape=(2, 512, 768, 4))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mlx_conv = create_conv2d(
        4,
        16,
        kernel_size,
        padding=padding,
        stride=stride,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )
    timm_conv = create_conv2d_torch(
        4,
        16,
        kernel_size,
        padding=padding,
        stride=stride,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )

    mlx_conv.weight = U.torch_to_mlx_2d(timm_conv.weight)

    if timm_conv.bias is not None:
        mlx_conv.bias = mx.array(timm_conv.bias.detach().numpy())

    mx.eval(mlx_conv.parameters())

    out_mlx = mlx_conv(x_mlx)
    out_timm = timm_conv(x_torch)

    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-5
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


@pytest.mark.parametrize(
    "kernel_size,padding,stride,dilation",
    [
        ([1, 3], "", 1, 1),
    ],
)
@pytest.mark.parametrize("bias", [True, False])
def test_conv2d_mixed(kernel_size, padding, stride, dilation, bias):
    x_mlx = U.sample_mlx_array_2d(shape=(2, 512, 768, 4))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mlx_conv = create_conv2d(
        4,
        16,
        kernel_size,
        padding=padding,
        stride=stride,
        dilation=dilation,
        bias=bias,
    )
    timm_conv = create_conv2d_torch(
        4,
        16,
        kernel_size,
        padding=padding,
        stride=stride,
        dilation=dilation,
        bias=bias,
    )

    for i, timm_layer in enumerate(timm_conv.values()):
        mlx_conv.layers[i].weight = U.torch_to_mlx_2d(timm_layer.weight)

        if timm_layer.bias is not None:
            mlx_conv.layers[i].bias = mx.array(timm_layer.bias.detach().numpy())

    mx.eval(mlx_conv.parameters())

    out_mlx = mlx_conv(x_mlx)
    out_timm = timm_conv(x_torch)

    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-5
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


@pytest.mark.parametrize(
    "kernel_size,padding,stride,dilation",
    [
        # (1, 0, 1, 1),
        (3, 1, 1, 1),
        # (3, 1, 2, 1),
    ],
)
@pytest.mark.parametrize("bias", [True, False])
def test_conv2d_cond(kernel_size, padding, stride, dilation, bias):
    x_mlx = U.sample_mlx_array_2d(shape=(2, 512, 768, 4))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    r_mlx = mx.sigmoid(mx.random.normal((2, 4)))
    r_torch = torch.from_numpy(np.array(r_mlx))

    mlx_conv = create_conv2d(
        4,
        16,
        kernel_size,
        padding=padding,
        stride=stride,
        dilation=dilation,
        bias=bias,
        num_experts=4,
    )
    timm_conv = create_conv2d_torch(
        4,
        16,
        kernel_size,
        padding=padding,
        stride=stride,
        dilation=dilation,
        bias=bias,
        num_experts=4,
    )

    # NOTE: CondConv2d needs special care to convert the weights from timm to MLX.
    mlx_conv.weight = mx.array(
        timm_conv.weight.reshape(4, 16, 4, kernel_size, kernel_size)
        .permute(0, 1, 3, 4, 2)
        .reshape(4, -1)
        .detach()
        .numpy()
    )

    if timm_conv.bias is not None:
        mlx_conv.bias = mx.array(timm_conv.bias.detach().numpy())

    out_mlx = mlx_conv(x_mlx, r_mlx)
    out_timm = timm_conv(x_torch, r_torch)

    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-5
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"
