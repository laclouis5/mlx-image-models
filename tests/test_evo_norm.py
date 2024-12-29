import mlx.core as mx
import numpy as np
import pytest
import timm.layers.evo_norm as timm_evo_norm

import mlx_im.layers.evo_norm as mlx_evo_norm

from . import utils as U


def test_instance_std():
    x_mlx = U.sample_mlx_array_2d(shape=(1, 512, 768, 64))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    out_mlx = mlx_evo_norm.instance_std(x_mlx)
    out_timm = timm_evo_norm.instance_std(x_torch)

    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-7
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


def test_instance_rms():
    x_mlx = U.sample_mlx_array_2d(shape=(1, 512, 768, 64))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    out_mlx = mlx_evo_norm.instance_rms(x_mlx)
    out_timm = timm_evo_norm.instance_rms(x_torch)

    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-7
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


@pytest.mark.parametrize("groups", [1, 2, 4, 8, 16, 32, 64])
@pytest.mark.parametrize("flatten", [False, True])
def test_group_std(groups, flatten):
    x_mlx = U.sample_mlx_array_2d(shape=(1, 512, 768, 64))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    out_mlx = mlx_evo_norm.group_std(x_mlx, groups=groups, flatten=flatten)
    out_timm = timm_evo_norm.group_std(x_torch, groups=groups, flatten=flatten)

    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-7
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


@pytest.mark.parametrize("groups", [1, 2, 4, 8, 16, 32, 64])
def test_group_rms(groups):
    x_mlx = U.sample_mlx_array_2d(shape=(1, 512, 768, 64))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    out_mlx = mlx_evo_norm.group_rms(x_mlx, groups=groups)
    out_timm = timm_evo_norm.group_rms(x_torch, groups=groups)

    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-7
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


@pytest.mark.parametrize("apply_act", [True, False])
def test_evonorm_2db0(apply_act):
    x_mlx = U.sample_mlx_array_2d(shape=(1, 512, 768, 64))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_evo_norm.EvoNorm2dB0(num_features=64, apply_act=apply_act)
    mod_timm = timm_evo_norm.EvoNorm2dB0(num_features=64, apply_act=apply_act)

    out_mlx = mod_mlx(x_mlx)
    out_timm = mod_timm(x_torch)

    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-7
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


@pytest.mark.parametrize("apply_act", [True, False])
def test_evonorm_2db1(apply_act):
    x_mlx = U.sample_mlx_array_2d(shape=(1, 512, 768, 64))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_evo_norm.EvoNorm2dB1(num_features=64, apply_act=apply_act)
    mod_timm = timm_evo_norm.EvoNorm2dB1(num_features=64, apply_act=apply_act)

    out_mlx = mod_mlx(x_mlx)
    out_timm = mod_timm(x_torch)

    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-7
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


@pytest.mark.parametrize("apply_act", [True, False])
def test_evonorm_2db2(apply_act):
    x_mlx = U.sample_mlx_array_2d(shape=(1, 512, 768, 64))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_evo_norm.EvoNorm2dB2(num_features=64, apply_act=apply_act)
    mod_timm = timm_evo_norm.EvoNorm2dB2(num_features=64, apply_act=apply_act)

    out_mlx = mod_mlx(x_mlx)
    out_timm = mod_timm(x_torch)

    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-7
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


@pytest.mark.parametrize("groups", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("apply_act", [True, False])
def test_evonorm_2ds0(groups, apply_act):
    x_mlx = U.sample_mlx_array_2d(shape=(1, 512, 768, 32))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_evo_norm.EvoNorm2dS0(
        num_features=32, groups=groups, apply_act=apply_act
    )
    mod_timm = timm_evo_norm.EvoNorm2dS0(
        num_features=32, groups=groups, apply_act=apply_act
    )

    out_mlx = mod_mlx(x_mlx)
    out_timm = mod_timm(x_torch)

    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-7
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


@pytest.mark.parametrize("groups", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("apply_act", [True, False])
def test_evonorm_2ds0a(groups, apply_act):
    x_mlx = U.sample_mlx_array_2d(shape=(1, 512, 768, 32))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_evo_norm.EvoNorm2dS0a(
        num_features=32, groups=groups, apply_act=apply_act
    )
    mod_timm = timm_evo_norm.EvoNorm2dS0a(
        num_features=32, groups=groups, apply_act=apply_act
    )

    out_mlx = mod_mlx(x_mlx)
    out_timm = mod_timm(x_torch)

    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-7
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


@pytest.mark.parametrize("groups", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("apply_act", [True, False])
def test_evonorm_2ds1(groups, apply_act):
    x_mlx = U.sample_mlx_array_2d(shape=(1, 512, 768, 32))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_evo_norm.EvoNorm2dS1(
        num_features=32, groups=groups, apply_act=apply_act
    )
    mod_timm = timm_evo_norm.EvoNorm2dS1(
        num_features=32, groups=groups, apply_act=apply_act
    )

    out_mlx = mod_mlx(x_mlx)
    out_timm = mod_timm(x_torch)

    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-7
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


@pytest.mark.parametrize("groups", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("apply_act", [True, False])
def test_evonorm_2ds1a(groups, apply_act):
    x_mlx = U.sample_mlx_array_2d(shape=(1, 512, 768, 32))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_evo_norm.EvoNorm2dS1a(
        num_features=32, groups=groups, apply_act=apply_act
    )
    mod_timm = timm_evo_norm.EvoNorm2dS1a(
        num_features=32, groups=groups, apply_act=apply_act
    )

    out_mlx = mod_mlx(x_mlx)
    out_timm = mod_timm(x_torch)

    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-7
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


@pytest.mark.parametrize("groups", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("apply_act", [True, False])
def test_evonorm_2ds2(groups, apply_act):
    x_mlx = U.sample_mlx_array_2d(shape=(1, 512, 768, 32))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_evo_norm.EvoNorm2dS2(
        num_features=32, groups=groups, apply_act=apply_act
    )
    mod_timm = timm_evo_norm.EvoNorm2dS2(
        num_features=32, groups=groups, apply_act=apply_act
    )

    out_mlx = mod_mlx(x_mlx)
    out_timm = mod_timm(x_torch)

    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-7
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"


@pytest.mark.parametrize("groups", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("apply_act", [True, False])
def test_evonorm_2ds2a(groups, apply_act):
    x_mlx = U.sample_mlx_array_2d(shape=(1, 512, 768, 32))
    x_torch = U.mlx_to_torch_2d(x_mlx)

    mod_mlx = mlx_evo_norm.EvoNorm2dS2a(
        num_features=32, groups=groups, apply_act=apply_act
    )
    mod_timm = timm_evo_norm.EvoNorm2dS2a(
        num_features=32, groups=groups, apply_act=apply_act
    )

    out_mlx = mod_mlx(x_mlx)
    out_timm = mod_timm(x_torch)

    mx.eval(out_mlx)

    out_mlx = U.mlx_to_numpy_2d(out_mlx)
    out_timm = U.torch_to_numpy_2d(out_timm)

    assert np.allclose(
        out_mlx, out_timm, atol=1.0e-7
    ), f"{np.max(np.abs(out_mlx - out_timm)).item()}"
