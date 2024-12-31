import mlx.core as mx
import numpy as np
import pytest
import timm.models._manipulate as timm_m
import torch
import torch.nn as nn_torch
from mlx import nn as nn_mlx
from mlx.utils import tree_flatten, tree_map

import mlx_im.models._manipulate as mlx_m
from mlx_im.layers.mlx_layers import AdaptiveAvgPool2d


@pytest.mark.parametrize("exclude_head", [True, False])
def test_model_parameters(exclude_head: bool):
    class MLXModel(nn_mlx.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn_mlx.Conv2d(32, 32, 3, 1)
            self.classifier = nn_mlx.Sequential(
                AdaptiveAvgPool2d(1),
                nn_mlx.Linear(32, 4),
            )

        def __call__(self, x):
            return self.classifier(self.backbone(x))

    class TorchModel(nn_torch.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.backbone = nn_torch.Conv2d(32, 32, 3, 1)
            self.classifier = nn_torch.Sequential(
                nn_torch.AdaptiveAvgPool2d(1),
                nn_torch.Linear(32, 4),
            )

        def forward(self, x):
            return self.classifier(self.backbone(x))

    mod_mlx = MLXModel()
    mod_timm = TorchModel()

    param_mlx = mlx_m.model_parameters(mod_mlx, exclude_head=exclude_head)
    param_timm = timm_m.model_parameters(mod_timm, exclude_head=exclude_head)

    for p_mlx, p_timm in zip(param_mlx, param_timm):
        if p_mlx.ndim == 4:
            p_mlx = p_mlx.transpose(0, 3, 1, 2)

        assert tuple(p_timm.shape) == p_mlx.shape


def test_named_apply():
    class MLXModel(nn_mlx.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn_mlx.Conv2d(32, 32, 3, 1)
            self.classifier = nn_mlx.Sequential(
                AdaptiveAvgPool2d(1),
                nn_mlx.Linear(32, 4),
            )

        def __call__(self, x):
            return self.classifier(self.backbone(x))

    class TorchModel(nn_torch.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.backbone = nn_torch.Conv2d(32, 32, 3, 1)
            self.classifier = nn_torch.Sequential(
                nn_torch.AdaptiveAvgPool2d(1),
                nn_torch.Linear(32, 4),
            )

        def forward(self, x):
            return self.classifier(self.backbone(x))

    mod_mlx = MLXModel()
    mod_timm = TorchModel()

    def mlx_apply(module: nn_mlx.Module, name: str):
        if "classifier" in name:
            module.eval()

    def timm_apply(module: nn_torch.Module, name: str):
        if "classifier" in name:
            module.eval()

    out_mlx = mlx_m.named_apply(mlx_apply, mod_mlx)
    out_timm = timm_m.named_apply(timm_apply, mod_timm)

    assert not out_mlx.classifier.training and not out_timm.classifier.training


def test_named_modules():
    class MLXModel(nn_mlx.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn_mlx.Conv2d(32, 32, 3, 1)
            self.classifier = nn_mlx.Sequential(
                AdaptiveAvgPool2d(1),
                nn_mlx.Linear(32, 4),
            )
            self.more_weights = [mx.zeros((2, 2)), mx.zeros((2, 2))]
            self.no_weights = ["a", "b"]
            self.dict_weights = {
                "one": mx.zeros((2, 2)),
                "three": mx.zeros((2, 2)),
            }
            self.mods = [nn_mlx.Linear(5, 5)]

        def __call__(self, x: mx.array) -> mx.array:
            return self.classifier(self.backbone(x))

    class TorchModel(nn_torch.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.backbone = nn_torch.Conv2d(32, 32, 3, 1)
            self.classifier = nn_torch.Sequential(
                nn_torch.AdaptiveAvgPool2d(1),
                nn_torch.Linear(32, 4),
            )
            self.more_weights = nn_torch.ParameterList(
                [
                    nn_torch.Parameter(torch.ones(2, 2)),
                    nn_torch.Parameter(torch.ones(2, 2)),
                ]
            )
            self.no_weights = ["a", "b"]
            self.dict_weights = nn_torch.ParameterDict(
                {
                    "one": nn_torch.Parameter(torch.ones(3, 3)),
                    "three": nn_torch.Parameter(torch.ones(3, 3)),
                }
            )
            self.mods = nn_torch.ModuleList([nn_torch.Linear(5, 5)])

        def forward(self, x):
            return self.classifier(self.backbone(x))

    mod_mlx = MLXModel()
    mod_timm = TorchModel()

    out_mlx = list(n for n, _ in mlx_m.named_modules(mod_mlx))
    out_timm = list(n for n, _ in timm_m.named_modules(mod_timm))

    assert set(n.replace(".layers", "") for n in out_mlx) == set(out_timm)


def test_named_modules_with_params():
    class MLXModel(nn_mlx.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn_mlx.Conv2d(32, 32, 3, 1)
            self.classifier = nn_mlx.Sequential(
                AdaptiveAvgPool2d(1),
                nn_mlx.Linear(32, 4),
            )

        def __call__(self, x):
            return self.classifier(self.backbone(x))

    class TorchModel(nn_torch.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.backbone = nn_torch.Conv2d(32, 32, 3, 1)
            self.classifier = nn_torch.Sequential(
                nn_torch.AdaptiveAvgPool2d(1),
                nn_torch.Linear(32, 4),
            )

        def forward(self, x):
            return self.classifier(self.backbone(x))

    mod_mlx = MLXModel()
    mod_timm = TorchModel()

    out_mlx = list(mlx_m.named_modules_with_params(mod_mlx))
    out_timm = list(timm_m.named_modules_with_params(mod_timm))

    raise AssertionError
