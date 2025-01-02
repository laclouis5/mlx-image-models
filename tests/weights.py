import mlx.core as mx
from mlx import nn as nn_mlx
from timm import layers as L_timm
from torch import nn as nn_torch

from mlx_im import layers as L_mlx

from . import utils as U


def transfer_weights(torch_module: nn_torch.Module, mlx_module: nn_mlx.Module):
    """Recursively transfer weights from a Torch module to a MLX module."""

    if isinstance(torch_module, L_timm.CondConv2d):
        assert isinstance(mlx_module, L_mlx.CondConv2d)
        mlx_module.weight = mx.array(
            torch_module.weight.reshape(
                torch_module.num_experts,
                torch_module.out_channels,
                torch_module.in_channels // torch_module.groups,
                torch_module.kernel_size[0],
                torch_module.kernel_size[1],
            )
            .permute(0, 1, 3, 4, 2)
            .reshape(torch_module.num_experts, -1)
            .detach()
            .numpy()
        )

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

    elif isinstance(torch_module, nn_torch.Linear):
        assert isinstance(mlx_module, nn_mlx.Conv2d)
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
