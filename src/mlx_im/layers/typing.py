from typing import Callable, Tuple, Type, Union

from mlx import nn

LayerType = Union[str, Callable, Type[nn.Module]]
PadType = Union[str, int, Tuple[int, int]]
