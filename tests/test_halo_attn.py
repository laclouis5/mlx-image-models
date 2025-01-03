import mlx.core as mx
import numpy as np
import pytest
import timm.layers.halo_attn as timm_m
import torch

import mlx_im.layers.halo_attn as mlx_m

from . import utils as U
from . import weights


def test_rel_logits_1d():
    torch.manual_seed(42)

    B, N = 2, 8
    H, W = 32, 48
    Bs, Hs = 8, 3
    Ws = Bs * 2 * Hs
    Nbh, Nbw = H // Bs, W // Bs
    Nb = Nbh * Nbw
    Dqk = 128

    x_mlx = mx.random.normal(shape=(B * N * Nb, Bs, Bs, Dqk))
    x_torch = torch.from_numpy(np.array(x_mlx))

    rel_k_mlx = mx.random.normal(shape=(2 * Ws - 1, Dqk))
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

    B, N = 2, 8
    H, W = 32, 48
    Bs, Hs = 8, 3
    Ws = Bs * 2 * Hs
    Nbh, Nbw = H // Bs, W // Bs
    Nb = Nbh * Nbw
    Dqk = 128

    x_mlx = mx.random.normal(shape=(B * N, Nb, Bs * Bs, Dqk))
    x_torch = torch.from_numpy(np.array(x_mlx))

    mod_mlx = mlx_m.PosEmbedRel(block_size=Bs, win_size=Ws, dim_head=Dqk, scale=scale)
    mod_timm = timm_m.PosEmbedRel(block_size=Bs, win_size=Ws, dim_head=Dqk, scale=scale)

    weights.transfer_weights(mod_timm, mod_mlx)

    out_timm = mod_timm(q=x_torch)
    out_mlx = mod_mlx(q=x_mlx)
    mx.eval(out_mlx)

    out_mlx = np.array(out_mlx)
    out_timm = out_timm.detach().numpy()

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-5
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("stride", [1, 2])
def test_halo_attn(bias, stride):
    torch.manual_seed(42)

    B, N = 2, 8
    H, W = 32, 48
    Bs, Hs = 8, 3
    D = 128

    x_mlx = mx.random.normal(shape=(B, H, W, D))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_m.HaloAttn(
        D, num_heads=N, block_size=Bs, halo_size=Hs, qkv_bias=bias, stride=stride
    )
    mod_timm = timm_m.HaloAttn(
        D, num_heads=N, block_size=Bs, halo_size=Hs, qkv_bias=bias, stride=stride
    )

    weights.transfer_weights(mod_timm, mod_mlx)

    out_timm = mod_timm(x_torch)
    out_mlx = mod_mlx(x_mlx)
    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-5
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"
