# coding=utf-8
# Copyright (c) DIRECT Contributors

import contextlib
from enum import Enum
from math import ceil
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

import numpy as np
import torch

from direct.data.transforms import apply_mask
from direct.ssl._fill_gaussian_ssdu import fill_gaussian_ssdu
from direct.utils import DirectModule

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


class MaskSplitter(DirectModule):
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
        backward_operator: Callable,
        split_type: MaskSplitterSplitType,
        ratio: float = 0.5,
        acs_region: Union[List[int], Tuple[int, int]] = (4, 4),
        use_seed: bool = True,
        kspace_key: str = "masked_kspace",
        target_key: str = "target",
    ):
        r"""Inits :class:`MaskSplitter`.

        Parameters
        ----------
        backward_operator: Callable
            The backward operator, e.g. some form of inverse FFT (centered or uncentered).
        split_type: MaskSplitterSplitType
            Type of mask splitting. Can be `gaussian` or `uniform`.
        ratio: float
            Split ratio such that :math:`ratio \approx \frac{|A|}{|B|}. Default: 0.5.
        acs_region: list or tuple of ints
            Size of ACS region to include in training (input) mask. Default: (4, 4).
        use_seed: bool
            If true, a pseudo-random number based on the filename is computed so that every slice of the volume get
            the same mask every time. Default: True.
        kspace_key: str
            K-space key. Default `masked_kspace`.
        target_key: str
            Target key. Default `target`.

        """
        super().__init__()
        assert split_type in ["gaussian", "uniform"]
        self.split_type = split_type
        self.ratio = ratio
        self.acs_region = acs_region

        self.backward_operator = backward_operator
        self.target_key = target_key
        self.kspace_key = kspace_key

        self.use_seed = use_seed
        self.rng = np.random.RandomState()

    def gaussian_split(
        self, mask: torch.Tensor, std_scale: float = 3.0, seed: Union[Tuple[int, ...], List[int], int, None] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Splits `mask` into an input and target disjoint masks using a bivariate Gaussian sampling.

        Parameters
        ----------
        mask: torch.Tensor
            Masking tensor to split.
        std_scale: float = 3.0
            This is used to calculate the standard deviation of the Gaussian distribution. Default: 3.0.
        seed: int, list or tuple of ints or None
            Default: None.

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
            if seed is None:
                seed = np.random.randint(0, 1e5)
            elif isinstance(seed, (tuple, list)):
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
        self, mask: torch.Tensor, seed: Union[Tuple[int, ...], List[int], int, None] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Splits `mask` into an input and target disjoint masks using a uniform sampling.

        Parameters
        ----------
        mask: torch.Tensor
            Masking tensor to split.
        seed: int, list or tuple of ints or None
            Default: None.

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

    @staticmethod
    def _unsqueeze_mask(masks: Iterable[torch.Tensor]) -> List[torch.Tensor]:
        return [mask[None, ..., None] for mask in masks]


class UniformMaskSplitter(MaskSplitter):
    def __init__(
        self,
        backward_operator: Callable,
        ratio: float = 0.5,
        acs_region: Union[List[int], Tuple[int, int]] = (4, 4),
        use_seed: bool = True,
        kspace_key: str = "masked_kspace",
        target_key: str = "target",
    ):
        super().__init__(
            backward_operator=backward_operator,
            split_type=MaskSplitterSplitType.uniform,
            ratio=ratio,
            acs_region=acs_region,
            use_seed=use_seed,
            kspace_key=kspace_key,
            target_key=target_key,
        )

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sampling_mask = sample["sampling_mask"].clone().squeeze()
        kspace = sample[self.kspace_key].clone()

        seed = None if not self.use_seed else tuple(map(ord, str(sample["filename"])))
        input_mask, target_mask = self._unsqueeze_mask(self.uniform_split(mask=sampling_mask, seed=seed))
        del sampling_mask

        # TODO: See which keys we actually need, or maybe can be deleted by DeleteKey transform
        sample[self.kspace_key], _ = apply_mask(kspace, input_mask)
        sample[self.kspace_key + "_sampling_mask"] = input_mask
        target_kspace, _ = apply_mask(kspace, target_mask)
        del kspace
        sample[self.target_key] = self.backward_operator(target_kspace, dim=(1, 2))
        sample[self.target_key + "_sampling_mask"] = target_mask
        return sample


class GaussianMaskSplitter(MaskSplitter):
    def __init__(
        self,
        backward_operator: Callable,
        ratio: float = 0.5,
        acs_region: Union[List[int], Tuple[int, int]] = (4, 4),
        use_seed: bool = True,
        kspace_key: str = "masked_kspace",
        target_key: str = "target",
        std_scale: float = 3.0,
    ):
        super().__init__(
            backward_operator=backward_operator,
            split_type=MaskSplitterSplitType.gaussian,
            ratio=ratio,
            acs_region=acs_region,
            use_seed=use_seed,
            kspace_key=kspace_key,
            target_key=target_key,
        )
        self.std_scale = std_scale

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sampling_mask = sample["sampling_mask"].clone().squeeze()
        kspace = sample[self.kspace_key].clone()

        seed = None if not self.use_seed else tuple(map(ord, str(sample["filename"])))
        input_mask, target_mask = self._unsqueeze_mask(
            self.gaussian_split(mask=sampling_mask, seed=seed, std_scale=self.std_scale)
        )
        del sampling_mask

        # TODO: See which keys we actually need, or maybe can be deleted by DeleteKey transform
        sample[self.kspace_key], _ = apply_mask(kspace, input_mask)
        sample[self.kspace_key + "_sampling_mask"] = input_mask
        target_kspace, _ = apply_mask(kspace, target_mask)
        del kspace
        sample[self.target_key] = self.backward_operator(target_kspace, dim=(1, 2))
        sample[self.target_key + "_sampling_mask"] = target_mask
        return sample
