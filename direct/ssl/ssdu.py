# coding=utf-8
# Copyright (c) DIRECT Contributors

import contextlib
from enum import Enum
from math import ceil
from typing import List, Tuple, Union

import numpy as np
import torch

from direct.ssl._fill_gaussian_ssdu import fill_gaussian_ssdu

__all__ = [
    "GaussianMaskSplitter",
    "UniformMaskSplitter",
]


@contextlib.contextmanager
def temp_seed(rng, seed):
    state = rng.get_state()
    rng.seed(seed)
    try:
        yield
    finally:
        rng.set_state(state)


class MaskSplitterSplitType(str, Enum):

    uniform = "uniform"
    gaussian = "gaussian"


class MaskSplitter:
    r"""Splits input mask :math:`A` into two disjoint masks :math:`B`, :math:`C` such that

    .. math::
        A = B \cup C, B = A \backslash C,

    to be used as training (input) and loss (target) masks.

    Inspired and adapted from code implementation of _[1], _[2].

    References
    ----------

    .. [1] Yaman, Burhaneddin, et al. “Self‐supervised Learning of Physics‐guided Reconstruction Neural Networks
        without Fully Sampled Reference Data.” Magnetic Resonance in Medicine, vol. 84, no. 6, Dec. 2020,
        pp. 3172–91. DOI.org (Crossref), https://doi.org/10.1002/mrm.28378.
    .. [2] Yaman, Burhaneddin, et al. “Self-Supervised Physics-Based Deep Learning MRI Reconstruction Without
        Fully-Sampled Data.” 2020 IEEE 17th International Symposium on Biomedical Imaging (ISBI), 2020,
        pp. 921–25. IEEE Xplore, https://doi.org/10.1109/ISBI45749.2020.9098514.
    """

    def __init__(
        self,
        split_type: MaskSplitterSplitType,
        ratio: float = 0.5,
        acs_region: Union[List[int], Tuple[int, int]] = (4, 4),
    ):
        r"""Inits :class:`MaskSplitter`.

        Parameters
        ----------
        split_type: MaskSplitterSplitType
            Type of mask splitting. Can be `gaussian` or `uniform`.
        ratio: float
            Split ratio such that :math:`ratio \approx \frac{|A|}{|B|}. Default: 0.5.
        acs_region: list or tuple of ints
            Size of ACS region to include in training (input) mask. Default: (4, 4).

        """
        assert split_type in ["gaussian", "uniform"]
        self.split_type = split_type
        self.ratio = ratio
        self.acs_region = acs_region

        self.rng = np.random.RandomState()

    def gaussian_split(
        self, mask: torch.Tensor, std_scale: float = 3.0, seed: Union[Tuple[int, ...], List[int], int] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Splits `mask` into an input and target disjoint masks using a bivariate Gaussian sampling.

        Parameters
        ----------
        mask: torch.Tensor
            Masking tensor to split.
        std_scale: float = 3.0
            This is used to calculate the standard deviation of the Gaussian distribution. Default: 3.0.
        seed: int, list of tuple of ints

        Returns
        -------
        input_mask: torch.Tensor
            Mask to be used by model (input/training mask).
        target_mask: torch.Tensor
            Mask to be used as target (target/loss mask).
        """
        nrow, ncol = mask.shape

        center_x = nrow // 2
        center_y = ncol // 2

        std_scale = std_scale

        temp_mask = mask.clone()
        temp_mask[
            center_x - self.acs_region[0] // 2 : center_x + self.acs_region[0] // 2,
            center_y - self.acs_region[1] // 2 : center_y + self.acs_region[1] // 2,
        ] = False

        target_mask = torch.zeros_like(mask, dtype=mask.dtype)
        nonzero_mask_count = int(ceil(mask.sum() * self.ratio))

        with temp_seed(self.rng, seed):
            if isinstance(seed, (tuple, list)):
                seed = int(np.mean(seed))
            elif isinstance(seed, int):
                seed = seed

        target_mask = fill_gaussian_ssdu(
            nonzero_mask_count,
            nrow,
            ncol,
            center_x,
            center_y,
            std_scale,
            temp_mask.numpy().astype(int),
            target_mask.numpy().astype(int),
            seed,
        )

        target_mask = torch.tensor(target_mask, dtype=mask.dtype)
        input_mask = mask & (~target_mask)

        return input_mask, target_mask

    def uniform_split(
        self, mask: torch.Tensor, seed: Union[Tuple[int, ...], List[int], int] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Splits `mask` into an input and target disjoint masks using a uniform sampling.

        Parameters
        ----------
        mask: torch.Tensor
            Masking tensor to split.
        seed: int, list of tuple of ints

        Returns
        -------
        input_mask: torch.Tensor
            Mask to be used by model (input/training mask).
        target_mask: torch.Tensor
            Mask to be used as target (target/loss mask).
        """
        nrow, ncol = mask.shape

        center_x = nrow // 2
        center_y = ncol // 2

        temp_mask = mask.clone()

        temp_mask[
            center_x - self.acs_region[0] // 2 : center_x + self.acs_region[0] // 2,
            center_y - self.acs_region[1] // 2 : center_y + self.acs_region[1] // 2,
        ] = False

        prob = temp_mask.flatten().numpy()
        with temp_seed(self.rng, seed):
            ind_flattened = np.random.choice(
                torch.arange(nrow * ncol),
                size=int(np.count_nonzero(prob) * self.ratio),
                replace=False,
                p=prob / prob.sum(),
            )

        (ind_x, ind_y) = np.unravel_index(ind_flattened, (nrow, ncol))

        target_mask = torch.zeros_like(mask, dtype=mask.dtype)
        target_mask[ind_x, ind_y] = True

        input_mask = mask & (~target_mask)

        return input_mask, target_mask

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(f"Must be implemented by inheriting class.")


class UniformMaskSplitter(MaskSplitter):
    def __init__(self, ratio: float = 0.5, acs_region: Union[List[int], Tuple[int, int]] = (4, 4)):
        super().__init__(split_type=MaskSplitterSplitType.uniform, ratio=ratio, acs_region=acs_region)

    def __call__(
        self, sampling_mask: torch.Tensor, seed: Union[Tuple[int, ...], List[int], int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.uniform_split(mask=sampling_mask, seed=seed)


class GaussianMaskSplitter(MaskSplitter):
    def __init__(self, ratio: float = 0.5, acs_region: Union[List[int], Tuple[int, int]] = (4, 4)):
        super().__init__(split_type=MaskSplitterSplitType.gaussian, ratio=ratio, acs_region=acs_region)

    def __call__(
        self, sampling_mask: torch.Tensor, std_scale: float, seed: Union[Tuple[int, ...], List[int], int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.gaussian_split(mask=sampling_mask, std_scale=std_scale, seed=seed)
