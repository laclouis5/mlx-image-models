import mlx.core as mx
import numpy as np
import pytest
import timm.layers.bottleneck_attn as timm_m
import torch

import mlx_im.layers.bottleneck_attn as mlx_m

from . import utils as U
from . import weights as W


def test_rel_logits_1d():
    x_mlx = mx.random.normal(shape=(2, 32, 48, 128))
    x_torch = torch.from_numpy(np.array(x_mlx))

    rel_k_mlx = mx.random.normal(shape=(2 * 48 - 1, 128))
    rel_k_torch = torch.from_numpy(np.array(rel_k_mlx))

    mod_mlx = mlx_m.rel_logits_1d
    mod_timm = timm_m.rel_logits_1d

    out_timm = mod_timm(q=x_torch, rel_k=rel_k_torch, permute_mask=(0, 1, 3, 2, 4))
    out_mlx = mod_mlx(q=x_mlx, rel_k=rel_k_mlx, permute_mask=(0, 1, 3, 2, 4))
    mx.eval(out_mlx)

    out_mlx = np.array(out_mlx)
    out_timm = out_timm.detach().numpy()

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-5
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


@pytest.mark.parametrize("scale", [1.0, 0.5, 1.5])
def test_pos_emb_rel(scale):
    torch.manual_seed(42)

    x_mlx = mx.random.normal(shape=(2, 32 * 48, 128))
    x_torch = torch.from_numpy(np.array(x_mlx))

    mod_mlx = mlx_m.PosEmbedRel(feat_size=(32, 48), dim_head=128, scale=scale)
    mod_timm = timm_m.PosEmbedRel(feat_size=(32, 48), dim_head=128, scale=scale)

    mod_mlx.height_rel = mx.array(mod_timm.height_rel.detach().numpy())
    mod_mlx.width_rel = mx.array(mod_timm.width_rel.detach().numpy())

    out_mlx = mod_mlx(q=x_mlx)
    out_timm = mod_timm(q=x_torch)

    mx.eval(out_mlx)

    out_mlx = np.array(out_mlx)
    out_timm = out_timm.detach().numpy()

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-9
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


@pytest.mark.parametrize("dim_out", [128, 256])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("scale_pos_emb", [True, False])
def test_bottleneck_attn(dim_out, bias, scale_pos_emb):
    torch.manual_seed(42)

    x_mlx = U.sample_mlx_array_2d(shape=(2, 32, 48, 128))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_m.BottleneckAttn(
        dim=128,
        dim_out=dim_out,
        feat_size=(32, 48),
        dim_head=64,
        qkv_bias=bias,
        scale_pos_embed=scale_pos_emb,
    )
    mod_timm = timm_m.BottleneckAttn(
        dim=128,
        dim_out=dim_out,
        feat_size=(32, 48),
        dim_head=64,
        qkv_bias=bias,
        scale_pos_embed=scale_pos_emb,
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
