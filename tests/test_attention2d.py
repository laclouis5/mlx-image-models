import mlx.core as mx
import numpy as np
import pytest
import timm.layers.attention2d as timm_m
import torch

import mlx_im.layers.attention2d as mlx_m

from . import utils as U
from . import weights as W


@pytest.mark.parametrize("expand_first", [True, False])
@pytest.mark.parametrize("head_first", [True, False])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("attn_mask", [True, False])
def test_attn2d(expand_first, head_first, bias, attn_mask):
    torch.manual_seed(42)

    x_mlx = U.sample_mlx_array_2d(shape=(2, 32, 48, 128))
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

    W.transfer_weights(mod_timm, mod_mlx)

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

    x_mlx = U.sample_mlx_array_2d(shape=(2, 32, 48, dim))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_m.MultiQueryAttentionV2(dim=dim, dim_out=dim_out)
    mod_timm = timm_m.MultiQueryAttentionV2(dim=dim, dim_out=dim_out)

    W.transfer_weights(mod_timm, mod_mlx)

    if attn_mask:
        mask_mlx = -mx.random.normal(
            shape=(2, 16, 24, dim), key=mx.array([42, 42], dtype=mx.uint32)
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


@pytest.mark.parametrize("dim", [128])
@pytest.mark.parametrize("dim_out", [128, 256])
@pytest.mark.parametrize("query_strides", [1, 2])
@pytest.mark.parametrize("kv_stride", [1, 2])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("attn_mask", [True, False])
def test_mqa(dim, dim_out, query_strides, kv_stride, use_bias, attn_mask):
    torch.manual_seed(42)

    x_mlx = U.sample_mlx_array_2d(shape=(2, 32, 48, dim))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_m.MultiQueryAttention2d(
        dim=dim,
        dim_out=dim_out,
        key_dim=64,
        value_dim=64,
        query_strides=query_strides,
        kv_stride=kv_stride,
        padding="same",
        use_bias=use_bias,
    )
    mod_timm = timm_m.MultiQueryAttention2d(
        dim=dim,
        dim_out=dim_out,
        key_dim=64,
        value_dim=64,
        query_strides=query_strides,
        kv_stride=kv_stride,
        padding="same",
        use_bias=use_bias,
    )

    W.transfer_weights(mod_timm, mod_mlx)

    if attn_mask:
        mask_mlx = (
            -mx.random.randint(
                low=0,
                high=1,
                shape=(32 * 48 // query_strides**2, 32 * 48 // kv_stride**2),
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
