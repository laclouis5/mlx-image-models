import mlx.core as mx
from mlx import nn as nn_mlx
from timm import layers as L_timm
from torch import nn as nn_torch

from mlx_im import layers as L_mlx

from . import utils as U


def transfer_weights(torch_module: nn_torch.Module, mlx_module: nn_mlx.Module):
    """Recursively transfer weights from a Torch module to a MLX module."""

    if isinstance(torch_module, L_timm.mlp.GlobalResponseNorm):
        assert isinstance(mlx_module, L_mlx.mlp.GlobalResponseNorm)
        mlx_module.weight = mx.array(torch_module.weight.detach().numpy())
        mlx_module.bias = mx.array(torch_module.bias.detach().numpy())

    elif isinstance(torch_module, L_timm.CondConv2d):
        assert isinstance(mlx_module, L_mlx.CondConv2d)
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
        assert isinstance(mlx_module, L_mlx.MixedConv2d)
        for mod_t, mod_m in zip(torch_module.values(), mlx_module.layers):
            transfer_weights(mod_t, mod_m)

    elif isinstance(torch_module, nn_torch.Conv2d):
        assert isinstance(mlx_module, nn_mlx.Conv2d)
        mlx_module.weight = U.torch_to_mlx_2d(torch_module.weight)
        if torch_module.bias is not None:
            mlx_module.bias = mx.array(torch_module.bias.detach().numpy())

    elif isinstance(torch_module, nn_torch.Conv1d):
        assert isinstance(mlx_module, nn_mlx.Conv1d)
        weight = mx.array(torch_module.weight.detach().numpy())
        mlx_module.weight = weight.transpose(0, 2, 1)
        if torch_module.bias is not None:
            mlx_module.bias = mx.array(torch_module.bias.detach().numpy())

    elif isinstance(torch_module, nn_torch.Linear):
        assert isinstance(mlx_module, nn_mlx.Linear)
        mlx_module.weight = mx.array(torch_module.weight.detach().numpy())
        if torch_module.bias is not None:
            mlx_module.bias = mx.array(torch_module.bias.detach().numpy())

    elif isinstance(torch_module, nn_torch.Sequential):
        assert isinstance(mlx_module, nn_mlx.Sequential)
        for mod_t, mod_m in zip(torch_module, mlx_module.layers):
            transfer_weights(mod_t, mod_m)

    # Leaf module or high-level containers.
    elif isinstance(torch_module, nn_torch.Module):
        assert isinstance(mlx_module, nn_mlx.Module)
        sub_t = dict(torch_module.named_children())

        # Leaf module (norm, activation, etc.).
        if len(sub_t) == 0:
            return

        # Containers and composition of modules.
        for name, mod_t in sub_t.items():
            assert name in mlx_module
            transfer_weights(mod_t, mlx_module[name])

    else:
        raise ValueError(f"Module '{type(torch_module)}' not supported.")


def transfer_params(torch_module: nn_torch.Module, mlx_module: nn_mlx.Module):
    subp_t = dict(torch_module.named_parameters())

    if len(subp_t) == 0:
        return

    for name, p_t in subp_t.items():
        assert name in mlx_module
        assert p_t.ndim == mlx_module[name]

        if p_t.ndim == 1 or p_t.ndim == 2:
            mlx_module[name] = mx.array(p_t.detach().numpy())
        elif p_t.ndim == 3:
            weight = mx.array(p_t.detach().numpy())
            mlx_module[name] = weight.transpose(0, 2, 1)
        elif p_t.ndim == 4:
            weight = mx.array(p_t.detach().numpy())
            mlx_module[name] = weight.transpose(0, 2, 3, 1)
        else:
            raise ValueError(f"Paramter with {p_t.ndim} dimensions not supported.")