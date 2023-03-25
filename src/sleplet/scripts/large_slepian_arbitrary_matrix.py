from pathlib import Path

import numpy as np
from numpy import linalg as LA  # noqa: N812
from numpy.testing import assert_equal

from sleplet.data.setup_pooch import find_on_pooch_then_local
from sleplet.utils.slepian_arbitrary_methods import (
    calculate_high_L_matrix,
    clean_evals_and_evecs,
)

_data_path = Path(__file__).resolve().parents[1] / "data"


def compute_large_D_matrix(  # noqa: N802
    mask_name: str, L: int, L_ranges: list[int], shannon: int
) -> None:
    """
    checks that the split up D matrix has the same eigenvalues
    & eigenvectors as the computation of the whole D matrix in one step
    """
    D = calculate_high_L_matrix(L, L_ranges)
    eigenvalues_split, eigenvectors_split = clean_evals_and_evecs(LA.eigh(D))

    slepian_loc = f"slepian_eigensolutions_D_{mask_name}_L{L}_N{shannon}"
    eval_loc = f"{slepian_loc}_eigenvalues.npy"
    evec_loc = f"{slepian_loc}_eigenvectors.npy"
    try:
        eigenvalues = np.load(find_on_pooch_then_local(eval_loc))
        eigenvectors = np.load(find_on_pooch_then_local(evec_loc))
        assert_equal(eigenvalues_split, eigenvalues)
        assert_equal(eigenvectors_split, eigenvectors)
    except TypeError:
        np.save(_data_path / eval_loc, eigenvalues_split)
        np.save(_data_path / evec_loc, eigenvectors_split[:shannon])


if __name__ == "__main__":
    compute_large_D_matrix(
        mask_name="africa", L=128, L_ranges=[0, 83, 128], shannon=1208
    )
