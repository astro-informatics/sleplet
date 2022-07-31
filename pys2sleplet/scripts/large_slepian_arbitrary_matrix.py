from pathlib import Path

import numpy as np
from numpy import linalg as LA
from numpy.testing import assert_equal

from pys2sleplet.utils.slepian_arbitrary_methods import (
    calculate_high_L_matrix,
    clean_evals_and_evecs,
)

_file_location = Path(__file__).resolve()
_eigen_path = _file_location.parents[1] / "data" / "slepian" / "eigensolutions"


def compute_large_D_matrix(
    mask_name: str, L: int, L_ranges: list[int], shannon: int
) -> None:
    """
    checks that the split up D matrix has the same eigenvalues
    & eigenvectors as the computation of the whole D matrix in one step
    """
    slepian_loc = _eigen_path / f"D_{mask_name}_L{L}_N{shannon}"
    D = calculate_high_L_matrix(slepian_loc, L, L_ranges)
    eigenvalues_split, eigenvectors_split = clean_evals_and_evecs(LA.eigh(D))

    eval_loc = slepian_loc / "eigenvalues.npy"
    evec_loc = slepian_loc / "eigenvectors.npy"
    if eval_loc.exists() and evec_loc.exists():
        eigenvalues = np.load(eval_loc)
        eigenvectors = np.load(evec_loc)
        assert_equal(eigenvalues_split, eigenvalues)
        assert_equal(eigenvectors_split, eigenvectors)
    else:
        np.save(eval_loc, eigenvalues_split)
        np.save(evec_loc, eigenvectors_split[:shannon])


if __name__ == "__main__":
    compute_large_D_matrix(
        mask_name="south_america", L=128, L_ranges=[0, 64, 128], shannon=678
    )
