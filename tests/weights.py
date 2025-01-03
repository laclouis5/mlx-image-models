import mlx.core as mx
from mlx import nn as nn_mlx
from timm import layers as L_timm
from torch import nn as nn_torch

from mlx_im import layers as L_mlx

from . import utils as U


def transfer_weights(torch_module: nn_torch.Module, mlx_module: nn_mlx.Module):
    """Recursively transfer weights from a Torch module to a MLX module."""

    if isinstance(torch_module, L_timm.mlp.GlobalResponseNorm):
        assert isinstance(
            mlx_module, L_mlx.mlp.GlobalResponseNorm
        ), f"{type(mlx_module)}"
        mlx_module.weight = mx.array(torch_module.weight.detach().numpy())
        mlx_module.bias = mx.array(torch_module.bias.detach().numpy())

    elif isinstance(torch_module, L_timm.CondConv2d):
        assert isinstance(mlx_module, L_mlx.CondConv2d), f"{type(mlx_module)}"
        weight = torch_module.weight.reshape(
            torch_module.num_experts,
            torch_module.out_channels,
            torch_module.in_channels // torch_module.groups,
            torch_module.kernel_size[0],
            torch_module.kernel_size[1],
        )
        weight = weight.permute(0, 1, 3, 4, 2).reshape(torch_module.num_experts, -1)
        mlx_module.weight = mx.array(weight.detach().numpy())
        if torch_module.bias is not None:
            mlx_module.bias = mx.array(torch_module.bias.detach().numpy())

    elif isinstance(torch_module, L_timm.MixedConv2d):
        assert isinstance(mlx_module, L_mlx.MixedConv2d), f"{type(mlx_module)}"
        for mod_t, mod_m in zip(torch_module.values(), mlx_module.layers):
            transfer_weights(mod_t, mod_m)

    elif isinstance(torch_module, nn_torch.Conv3d):
        assert isinstance(mlx_module, nn_mlx.Conv3d), f"{type(mlx_module)}"
        weight = mx.array(torch_module.weight.detach().numpy())
        mlx_module.weight = weight.transpose(0, 2, 3, 4, 1)
        if torch_module.bias is not None:
            mlx_module.bias = mx.array(torch_module.bias.detach().numpy())

    elif isinstance(torch_module, nn_torch.Conv2d):
        assert isinstance(mlx_module, nn_mlx.Conv2d), f"{type(mlx_module)}"
        weight = mx.array(torch_module.weight.detach().numpy())
        mlx_module.weight = weight.transpose(0, 2, 3, 1)
        if torch_module.bias is not None:
            mlx_module.bias = mx.array(torch_module.bias.detach().numpy())

    elif isinstance(torch_module, nn_torch.Conv1d):
        assert isinstance(mlx_module, nn_mlx.Conv1d), f"{type(mlx_module)}"
        weight = mx.array(torch_module.weight.detach().numpy())
        mlx_module.weight = weight.transpose(0, 2, 1)
        if torch_module.bias is not None:
            mlx_module.bias = mx.array(torch_module.bias.detach().numpy())

    elif isinstance(torch_module, nn_torch.Linear):
        assert isinstance(mlx_module, nn_mlx.Linear), f"{type(mlx_module)}"
        mlx_module.weight = mx.array(torch_module.weight.detach().numpy())
        if torch_module.bias is not None:
            mlx_module.bias = mx.array(torch_module.bias.detach().numpy())

    elif isinstance(torch_module, nn_torch.Sequential):
        assert isinstance(mlx_module, nn_mlx.Sequential), f"{type(mlx_module)}"
        for mod_t, mod_m in zip(torch_module, mlx_module.layers):
            transfer_weights(mod_t, mod_m)

    # Leaf module or high-level containers.
    elif isinstance(torch_module, nn_torch.Module):
        assert isinstance(mlx_module, nn_mlx.Module), f"{type(mlx_module)}"
        sub_t = dict(torch_module.named_children())

        # Containers and composition of modules.
        for name, mod_t in sub_t.items():
            assert name in mlx_module, f"{name}"
            transfer_weights(mod_t, mlx_module[name])

        # Fallback to transfer params with heuristics for weight shapes.
        transfer_params(torch_module, mlx_module)
    else:
        raise ValueError(f"Module '{type(torch_module)}' not supported.")


def transfer_params(torch_module: nn_torch.Module, mlx_module: nn_mlx.Module):
    subp_t = dict(torch_module.named_parameters(recurse=False))

    for name, p_t in subp_t.items():
        assert name in mlx_module, f"{name}"
        assert p_t.ndim == mlx_module[name].ndim
        mlx_module[name] = mx.array(p_t.detach().numpy())
