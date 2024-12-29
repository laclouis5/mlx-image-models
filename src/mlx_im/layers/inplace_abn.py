import mlx.core as mx
from mlx import nn

has_iabn = False


def inplace_abn(
    x: mx.array,
    weight: mx.array | None,
    bias: mx.array | None,
    running_mean: mx.array,
    running_var: mx.array,
    training: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-05,
    activation: str = "leaky_relu",
    activation_param: float = 0.01,
) -> mx.array:
    raise NotImplementedError("InPlaceABN is not supported in MLX.")


def inplace_abn_sync(**kwargs):
    inplace_abn(**kwargs)


class InplaceAbn(nn.Module):
    """Activated Batch Normalization

    This gathers a BatchNorm and an activation function in a single module

    Parameters
    ----------
    num_features : int
        Number of feature channels in the input and output.
    eps : float
        Small constant to prevent numerical issues.
    momentum : float
        Momentum factor applied to compute running statistics.
    affine : bool
        If `True` apply learned scale and shift transformation after normalization.
    act_layer : str or nn.Module type
        Name or type of the activation functions, one of: `leaky_relu`, `elu`
    act_param : float
        Negative slope for the `leaky_relu` activation.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        apply_act: bool = True,
        act_layer: str = "leaky_relu",
        act_param: float = 0.01,
        drop_layer: float = None,
    ):
        super().__init__()
        raise NotImplementedError("InPlaceABN is not supported in MLX.")

    def __call__(self, x: mx.array) -> mx.array:
        raise NotImplementedError("InPlaceABN is not supported in MLX.")
