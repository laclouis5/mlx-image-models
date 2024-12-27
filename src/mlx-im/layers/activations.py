import mlx.core as mx
from mlx import nn


def swish(x: mx.array, inplace: bool = False) -> mx.array:
    return nn.silu(x)


class Swish(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def __call__(self, x: mx.array) -> mx.array:
        return swish(x, inplace=self.inplace)


def mish(x: mx.array, inplace: bool = False) -> mx.array:
    return nn.mish(x)


class Mish(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def __call__(self, x: mx.array) -> mx.array:
        return mish(x, inplace=self.inplace)


def sigmoid(x: mx.array, inplace: bool = False) -> mx.array:
    return nn.sigmoid(x)


class Sigmoid(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def __call__(self, x: mx.array) -> mx.array:
        return sigmoid(x, inplace=self.inplace)


def tanh(x: mx.array, inplace: bool = False) -> mx.array:
    return nn.tanh(x)


class Tanh(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def __call__(self, x: mx.array) -> mx.array:
        return tanh(x, inplace=self.inplace)


def hard_swish(x: mx.array, inplace: bool = False) -> mx.array:
    return nn.hardswish(x)


class HardSwish(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def __call__(self, x: mx.array) -> mx.array:
        return hard_swish(x, inplace=self.inplace)


def hard_sigmoid(x, inplace: bool = False):
    return nn.relu6(x + 3.0) / 6.0


class HardSigmoid(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def __call__(self, x: mx.array) -> mx.array:
        return hard_sigmoid(x, inplace=self.inplace)


def hard_mish(x: mx.array, inplace: bool = False) -> mx.array:
    return mx.clip(0.5 * x * (x + 2), a_min=0, a_max=2)


class HardMish(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def __call__(self, x: mx.array) -> mx.array:
        return hard_mish(x, inplace=self.inplace)


class PReLU(nn.PReLU):
    def __init__(
        self, num_parameters: int = 1, init: float = 0.25, inplace: bool = False
    ):
        super().__init__(num_parameters=num_parameters, init=init)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.prelu(x, self.weight)


def gelu(x: mx.array, inplace: bool = False) -> mx.array:
    return nn.gelu(x)


class GELU(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def __call__(self, x: mx.array) -> mx.array:
        return gelu(x, inplace=self.inplace)


def gelu_tanh(x: mx.array, inplace: bool = False) -> mx.array:
    return nn.gelu_approx(x)


class GELUTanh(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def __call__(self, x: mx.array) -> mx.array:
        return gelu_tanh(x, inplace=self.inplace)


def quick_gelu(x: mx.array, inplace: bool = False) -> mx.array:
    return nn.gelu_fast_approx(x)


class QuickGELU(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def __call__(self, x: mx.array) -> mx.array:
        return quick_gelu(x)
