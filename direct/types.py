# coding=utf-8
# Copyright (c) DIRECT Contributors

import pathlib
from typing import Any, Callable, Dict, Iterable, NewType, Optional, Tuple, Union

import torch
from torch import nn as nn
from torch.cuda.amp import GradScaler

Number = Union[float, int]
PathOrString = Union[pathlib.Path, str]
FileOrUrl = NewType("FileOrUrl", PathOrString)
HasStateDict = Union[nn.Module, torch.optim.Optimizer, GradScaler]
LossClosure = Callable[[], float]
OptLossClosure = Optional[LossClosure]
OptFloat = Optional[float]
Params = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]
