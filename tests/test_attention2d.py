import mlx.core as mx
import numpy as np
import pytest
import timm.layers.attention2d as timm_m
import torch

import mlx_im.layers.attention2d as mlx_m

from . import utils as U


@pytest.mark.parametrize("expand_first", [True, False])
@pytest.mark.parametrize("head_first", [True, False])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("attn_mask", [True, False])
def test_attn2d(expand_first, head_first, bias, attn_mask):
    torch.manual_seed(42)

    x_mlx = U.sample_mlx_array_2d(shape=(1, 32, 48, 128))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_m.Attention2d(
        128,
        64,
        num_heads=4,
        expand_first=expand_first,
        head_first=head_first,
        bias=bias,
    )
    mod_timm = timm_m.Attention2d(
        128,
        64,
        num_heads=4,
        expand_first=expand_first,
        head_first=head_first,
        bias=bias,
    )

    if not mod_timm.fused_attn:
        pytest.xfail("Timm unfused implementation is wrong and may fail")

    mod_mlx.qkv.weight = mx.array(mod_timm.qkv.weight.detach().numpy()).transpose(
        0, 2, 3, 1
    )
    mod_mlx.proj.weight = mx.array(mod_timm.proj.weight.detach().numpy()).transpose(
        0, 2, 3, 1
    )

    if bias:
        mod_mlx.qkv.bias = mx.array(mod_timm.qkv.bias.detach().numpy())
        mod_mlx.proj.bias = mx.array(mod_timm.proj.bias.detach().numpy())

    if attn_mask:
        mask_mlx = (
            -mx.random.randint(
                low=0,
                high=1,
                shape=(32 * 48, 32 * 48),
                key=mx.array([42, 42], dtype=mx.uint32),
            )
            * 1e7
        )
        mask_torch = torch.from_numpy(np.array(mask_mlx))
    else:
        mask_mlx = mask_torch = None

    out_mlx = mod_mlx(x_mlx, attn_mask=mask_mlx)
    out_timm = mod_timm(x_torch, attn_mask=mask_torch)

    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-6
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


@pytest.mark.parametrize("dim_out", [128, 256])
@pytest.mark.parametrize("attn_mask", [True, False])
def test_mqa_v2(dim_out, attn_mask):
    dim = 128

    if dim_out != dim:
        pytest.xfail("Timm implementation does not support dim != dim_out")

    torch.manual_seed(42)

    x_mlx = U.sample_mlx_array_2d(shape=(1, 32, 48, dim))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_m.MultiQueryAttentionV2(dim=dim, dim_out=dim_out)
    mod_timm = timm_m.MultiQueryAttentionV2(dim=dim, dim_out=dim_out)

    mod_mlx.query_proj = mx.array(mod_timm.query_proj.detach().numpy())
    mod_mlx.key_proj = mx.array(mod_timm.key_proj.detach().numpy())
    mod_mlx.value_proj = mx.array(mod_timm.value_proj.detach().numpy())
    mod_mlx.out_proj = mx.array(mod_timm.out_proj.detach().numpy())

    if attn_mask:
        mask_mlx = -mx.random.normal(
            shape=(1, 16, 24, dim), key=mx.array([42, 42], dtype=mx.uint32)
        )
        mask_torch = torch.from_numpy(np.array(mask_mlx)).permute(0, 3, 1, 2)
    else:
        mask_mlx = mask_torch = None

    out_mlx = mod_mlx(x_mlx, m=mask_mlx)
    out_timm = mod_timm(x_torch, m=mask_torch)
    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-4
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"
