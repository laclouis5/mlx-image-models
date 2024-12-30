import mlx.core as mx
import numpy as np
import pytest
import timm.layers.norm_act as timm_m
from mlx import nn as nn_mlx
from torch import nn as nn_torch

import mlx_im.layers.norm_act as mlx_m

from . import utils as U


@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("track_running_stats", [True, False])
@pytest.mark.parametrize("apply_act", [True, False])
def test_batchnorm_act(affine, track_running_stats, apply_act):
    x_mlx = U.sample_mlx_array_2d(shape=(1, 512, 768, 32))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_m.BatchNormAct2d(
        num_features=32,
        affine=affine,
        track_running_stats=track_running_stats,
        apply_act=apply_act,
    )
    mod_timm = timm_m.BatchNormAct2d(
        num_features=32,
        affine=affine,
        track_running_stats=track_running_stats,
        apply_act=apply_act,
    )

    out_mlx = mod_mlx(x_mlx)
    out_timm = mod_timm(x_torch)

    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-7
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


@pytest.mark.parametrize("apply_act", [True, False])
def test_frozen_batchnorm_act(apply_act):
    x_mlx = U.sample_mlx_array_2d(shape=(1, 512, 768, 32))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_m.FrozenBatchNormAct2d(
        num_features=32,
        apply_act=apply_act,
    )
    mod_timm = timm_m.FrozenBatchNormAct2d(
        num_features=32,
        apply_act=apply_act,
    )

    out_mlx = mod_mlx(x_mlx)
    out_timm = mod_timm(x_torch)

    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-7
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("track_running_stats", [True, False])
@pytest.mark.parametrize("apply_act", [True, False])
def test_freeze_bn(affine, track_running_stats, apply_act):
    if not track_running_stats:
        pytest.xfail("FrozenBatchNorm seems to always track running stats")

    x_mlx = U.sample_mlx_array_2d(shape=(1, 512, 768, 32))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = nn_mlx.Sequential(
        mlx_m.BatchNormAct2d(
            num_features=32,
            affine=affine,
            track_running_stats=track_running_stats,
            apply_act=apply_act,
        ),
        nn_mlx.BatchNorm(num_features=32),
        nn_mlx.Identity(),
    )

    mod_timm = nn_torch.Sequential(
        timm_m.BatchNormAct2d(
            num_features=32,
            affine=affine,
            track_running_stats=track_running_stats,
            apply_act=apply_act,
        ),
        nn_torch.BatchNorm2d(32),
        nn_torch.Identity(),
    )

    mod_mlx = mlx_m.freeze_batch_norm_2d(mod_mlx)
    mod_timm = timm_m.freeze_batch_norm_2d(mod_timm)

    assert isinstance(mod_mlx, nn_mlx.Sequential)
    assert isinstance(mod_timm, nn_torch.Sequential)

    assert isinstance(mod_mlx["layers"][0], mlx_m.FrozenBatchNormAct2d)
    assert isinstance(mod_timm[0], timm_m.FrozenBatchNormAct2d)

    assert isinstance(mod_mlx["layers"][1], mlx_m.FrozenBatchNorm2d)
    assert isinstance(mod_timm[1], timm_m.FrozenBatchNorm2d)

    assert isinstance(mod_mlx["layers"][2], nn_mlx.Identity)
    assert isinstance(mod_timm[2], nn_torch.Identity)

    out_mlx = mod_mlx(x_mlx)
    out_timm = mod_timm(x_torch)

    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-7
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


@pytest.mark.parametrize("apply_act", [True, False])
def test_unfreeze_bn(apply_act):
    pytest.xfail("Timm implementation bug")

    x_mlx = U.sample_mlx_array_2d(shape=(1, 512, 768, 32))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = nn_mlx.Sequential(
        mlx_m.FrozenBatchNormAct2d(
            num_features=32,
            apply_act=apply_act,
        ),
        mlx_m.FrozenBatchNorm2d(num_features=32),
        nn_mlx.Identity(),
    )

    mod_timm = nn_torch.Sequential(
        timm_m.FrozenBatchNormAct2d(
            num_features=32,
            apply_act=apply_act,
        ),
        timm_m.FrozenBatchNorm2d(32),
        nn_torch.Identity(),
    )

    mod_mlx = mlx_m.unfreeze_batch_norm_2d(mod_mlx)
    mod_timm = timm_m.unfreeze_batch_norm_2d(mod_timm)

    assert isinstance(mod_mlx, nn_mlx.Sequential)
    assert isinstance(mod_timm, nn_torch.Sequential)

    assert isinstance(mod_mlx["layers"][0], mlx_m.BatchNormAct2d)
    assert isinstance(mod_timm[0], timm_m.BatchNormAct2d)

    assert isinstance(mod_mlx["layers"][1], nn_mlx.BatchNorm)
    assert isinstance(mod_timm[1], nn_torch.BatchNorm2d)

    assert isinstance(mod_mlx["layers"][2], nn_mlx.Identity)
    assert isinstance(mod_timm[2], nn_torch.Identity)

    out_mlx = mod_mlx(x_mlx)
    out_timm = mod_timm(x_torch)

    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-7
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"
