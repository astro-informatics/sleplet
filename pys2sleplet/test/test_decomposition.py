from typing import List

import numpy as np
import pytest

from pys2sleplet.slepian.slepian_decomposition import SlepianDecomposition
from pys2sleplet.utils.config import config


@pytest.fixture
def function() -> np.ndarray:
    pass


@pytest.fixture
def ranks() -> List[int]:
    return [0, (config.L * config.L // 2), config.L * config.L - 1]


def test_slepian_decomposition_integrate_region_and_forward_transform() -> None:
    """
    test to ensure that the alternaitve methods of Slepian decomposition
    are as expected i.e. return coefficients close in value to each other

    LHS: integral over the region divided by the eigenvalues
    RHS: Slepian forward transform
    """
    sd = SlepianDecomposition(config.L, function)

    for rank in ranks:
        integrate_region = sd.decompose(rank, method="integrate_region")
        forward_transform = sd.decompose(rank, method="forward_transform")

        assert integrate_region == pytest.approx(forward_transform)


def test_slepian_decomposition_forward_transform_and_harmonic_sum() -> None:
    """
    test to ensure that the alternaitve methods of Slepian decomposition
    are as expected i.e. return coefficients close in value to each other

    LHS: Slepian forward transform
    RHS: sum over flm and the (Sp)lm* quantity
    """
    sd = SlepianDecomposition(config.L, function)

    for rank in ranks:
        forward_transform = sd.decompose(rank, method="forward_transform")
        harmonic_sum = sd.decompose(rank, method="harmonic_sum")

        assert forward_transform == pytest.approx(harmonic_sum)


def test_slepian_decomposition_harmonic_sum_and_integrate_region() -> None:
    """
    test to ensure that the alternaitve methods of Slepian decomposition
    are as expected i.e. return coefficients close in value to each other

    LHS: sum over flm and the (Sp)lm* quantity
    RHS: integral over the region divided by the eigenvalues
    """
    sd = SlepianDecomposition(config.L, function)

    for rank in ranks:
        harmonic_sum = sd.decompose(rank, method="harmonic_sum")
        integrate_region = sd.decompose(rank, method="integrate_region")

        assert harmonic_sum == pytest.approx(integrate_region)
