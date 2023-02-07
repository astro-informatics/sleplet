from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyssht as ssht
from numba import njit, prange
from numpy import linalg as LA

from sleplet.slepian.slepian_functions import SlepianFunctions
from sleplet.utils.array_methods import fill_upper_triangle_of_hermitian_matrix
from sleplet.utils.config import settings
from sleplet.utils.logger import logger
from sleplet.utils.mask_methods import create_mask_region
from sleplet.utils.region import Region
from sleplet.utils.vars import (
    PHI_MAX_DEFAULT,
    PHI_MIN_DEFAULT,
    THETA_MAX_DEFAULT,
    THETA_MIN_DEFAULT,
)

_file_location = Path(__file__).resolve()
_eigen_path = _file_location.parents[2] / "data" / "slepian" / "eigensolutions"


@dataclass
class SlepianLimitLatLon(SlepianFunctions):
    theta_min: float
    theta_max: float
    phi_min: float
    phi_max: float
    _N: int = field(init=False, repr=False)
    _phi_max: float = field(default=PHI_MAX_DEFAULT, init=False, repr=False)
    _phi_min: float = field(default=PHI_MIN_DEFAULT, init=False, repr=False)
    _region: Region = field(init=False, repr=False)
    _theta_max: float = field(default=THETA_MAX_DEFAULT, init=False, repr=False)
    _theta_min: float = field(default=THETA_MIN_DEFAULT, init=False, repr=False)

    def __post_init__(self) -> None:
        self.region = Region(
            theta_min=self.theta_min,
            theta_max=self.theta_max,
            phi_min=self.phi_min,
            phi_max=self.phi_max,
        )
        super().__post_init__()

    def _create_fn_name(self) -> None:
        self.name = f"slepian_{self.region.name_ending}"

    def _create_mask(self) -> None:
        self.mask = create_mask_region(self.L, self.region)

    def _calculate_area(self) -> None:
        self.area = (self.phi_max - self.phi_min) * (
            np.cos(self.theta_min) - np.cos(self.theta_max)
        )

    def _create_matrix_location(self) -> None:
        self.matrix_location = (
            _eigen_path / f"D_{self.region.name_ending}_L{self.L}_N{self.N}"
        )

    def _solve_eigenproblem(self) -> None:
        eval_loc = self.matrix_location / "eigenvalues.npy"
        evec_loc = self.matrix_location / "eigenvectors.npy"
        if eval_loc.exists() and evec_loc.exists():
            logger.info("binaries found - loading...")
            self.eigenvalues = np.load(eval_loc)
            self.eigenvectors = np.load(evec_loc)
        else:
            K = self._create_K_matrix()
            self.eigenvalues, self.eigenvectors = self._clean_evals_and_evecs(
                LA.eigh(K)
            )
            if settings.SAVE_MATRICES:
                np.save(eval_loc, self.eigenvalues)
                np.save(evec_loc, self.eigenvectors[: self.N])

    def _create_K_matrix(self) -> np.ndarray:
        """
        computes the K matrix
        """
        # Compute sub-integral matrix
        G = self._slepian_integral()

        # Compute Slepian matrix
        dl_array = ssht.generate_dl(np.pi / 2, self.L)
        K = self._slepian_matrix(dl_array, self.L, self.L - 1, G)
        fill_upper_triangle_of_hermitian_matrix(K)

        return K

    def _slepian_integral(self) -> np.ndarray:
        """
        Syntax:
        G = _slepian_integral()

        Output:
        G  =  Sub-integral matrix (obtained after the use of Wigner-D and
        Wigner-d functions in computing the Slepian integral) for all orders

        Description:
        This piece of code computes the sub-integral matrix (obtained after the
        use of Wigner-D and Wigner-d functions in computing the Slepian integral)
        for all orders using the formulation given in "Slepian spatialspectral
        concentration problem on the sphere: Analytical formulation for limited
        colatitude-longitude spatial region" by A. P. Bates, Z. Khalid and R. A.
        Kennedy.
        """
        G = np.zeros((4 * self.L - 3, 4 * self.L - 3), dtype=np.complex_)

        def helper(row: int, col: int, S: float) -> None:
            """
            Using conjugate symmetry property to reduce the number of iterations
            """
            try:
                Q = (1 / (col**2 - 1)) * (
                    np.exp(1j * col * self.theta_min)
                    * (1j * col * np.sin(self.theta_min) - np.cos(self.theta_min))
                    + np.exp(1j * col * self.theta_max)
                    * (np.cos(self.theta_max) - 1j * col * np.sin(self.theta_max))
                )
            except ZeroDivisionError:
                Q = 0.25 * (
                    2 * 1j * col * (self.theta_max - self.theta_min)
                    + np.exp(2 * 1j * col * self.theta_min)
                    - np.exp(2 * 1j * col * self.theta_max)
                )

            G[2 * (self.L - 1) + row, 2 * (self.L - 1) + col] = Q * S
            G[2 * (self.L - 1) - row, 2 * (self.L - 1) - col] = G[
                2 * (self.L - 1) + row, 2 * (self.L - 1) + col
            ].conj()

        # row = 0
        S = self.phi_max - self.phi_min
        for col in range(-2 * (self.L - 1), 1):
            helper(0, col, S)

        # row != 0
        for row in range(-2 * (self.L - 1), 0):
            S = (1j / row) * (
                np.exp(1j * row * self.phi_min) - np.exp(1j * row * self.phi_max)
            )
            for col in range(-2 * (self.L - 1), 2 * (self.L - 1) + 1):
                helper(row, col, S)

        return G

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _slepian_matrix(dl: np.ndarray, L: int, N: int, G: np.ndarray) -> np.ndarray:
        """
        Syntax:
        K = _slepian_matrix(dl, L, N, G)

        Input:
        G  =  Sub-integral matrix (obtained after the use of Wigner-D and
        Wigner-d functions in computing the Slepian integral) for all orders

        Output:
        K  =  Slepian matrix

        Description:
        This piece of code computes the Slepian matrix using the formulation
        given in "Slepian spatialspectral concentration problem on the sphere:
        Analytical formulation for limited colatitude-longitude spatial region"
        by A. P. Bates, Z. Khalid and R. A. Kennedy.
        """
        K = np.zeros((L**2, L**2), dtype=np.complex_)

        for l in prange(L):
            for p in range(l + 1):
                C1 = np.sqrt((2 * l + 1) * (2 * p + 1)) / (4 * np.pi)

                for m in range(-l, l + 1):
                    ind_lm = l * (l + 1) + m

                    for q in range(-p, p + 1):
                        ind_pq = p * (p + 1) + q
                        row = m - q
                        C2 = (-1j) ** row
                        ind_r = 2 * N + row

                        for mp in range(-l, l + 1):
                            C3 = dl[l, N + mp, N + m] * dl[l, N + mp, N]
                            S1 = 0

                            for qp in range(-p, p + 1):
                                col = mp - qp
                                C4 = dl[p, N + qp, N + q] * dl[p, N + qp, N]
                                ind_c = 2 * N + col
                                S1 += C4 * G[ind_r, ind_c]

                            K[ind_lm, ind_pq] += C3 * S1

                        K[ind_lm, ind_pq] *= C1 * C2
        return K

    @staticmethod
    def _clean_evals_and_evecs(
        eigendecomposition: tuple[Any, ...]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        need eigenvalues and eigenvectors to be in a certain format
        """
        # access values
        eigenvalues, eigenvectors = eigendecomposition

        # eigenvalues should be real
        eigenvalues = eigenvalues.real

        # Sort eigenvalues and eigenvectors in descending order of eigenvalues
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx].conj().T

        # ensure first element of each eigenvector is positive
        eigenvectors *= np.where(eigenvectors[:, 0] < 0, -1, 1)[:, np.newaxis]

        return eigenvalues, eigenvectors
