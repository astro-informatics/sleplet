from pathlib import Path

import numpy as np
import pyssht as ssht
from numba import njit, prange
from numpy import linalg as LA  # noqa: N812
from numpy import typing as npt
from pydantic.dataclasses import dataclass

from sleplet._array_methods import fill_upper_triangle_of_hermitian_matrix
from sleplet._data.setup_pooch import find_on_pooch_then_local
from sleplet._mask_methods import create_mask_region
from sleplet._validation import Validation
from sleplet._vars import (
    PHI_MAX_DEFAULT,
    PHI_MIN_DEFAULT,
    THETA_MAX_DEFAULT,
    THETA_MIN_DEFAULT,
)
from sleplet.region import Region
from sleplet.slepian._slepian_functions import SlepianFunctions

_data_path = Path(__file__).resolve().parents[2] / "_data"


@dataclass(config=Validation, kw_only=True)
class SlepianLimitLatLon(SlepianFunctions):
    phi_max: float = PHI_MAX_DEFAULT
    phi_min: float = PHI_MIN_DEFAULT
    theta_max: float = THETA_MAX_DEFAULT
    theta_min: float = THETA_MIN_DEFAULT

    def __post_init_post_parse__(self) -> None:
        super().__post_init_post_parse__()

    def _create_fn_name(self) -> str:
        return f"slepian_{self.region.name_ending}"

    def _create_region(self) -> Region:
        return Region(
            theta_min=self.theta_min,
            theta_max=self.theta_max,
            phi_min=self.phi_min,
            phi_max=self.phi_max,
        )

    def _create_mask(self) -> npt.NDArray[np.float_]:
        return create_mask_region(self.L, self.region)

    def _calculate_area(self) -> float:
        return (self.phi_max - self.phi_min) * (
            np.cos(self.theta_min) - np.cos(self.theta_max)
        )

    def _create_matrix_location(self) -> str:
        return f"slepian_eigensolutions_D_{self.region.name_ending}_L{self.L}_N{self.N}"

    def _solve_eigenproblem(
        self,
    ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.complex_]]:
        eval_loc = f"{self.matrix_location}_eigenvalues.npy"
        evec_loc = f"{self.matrix_location}_eigenvectors.npy"
        try:
            return np.load(find_on_pooch_then_local(eval_loc)), np.load(
                find_on_pooch_then_local(evec_loc)
            )
        except TypeError:
            K = self._create_K_matrix()
            eigenvalues, eigenvectors = self._clean_evals_and_evecs(LA.eigh(K))
            np.save(_data_path / eval_loc, eigenvalues)
            np.save(_data_path / evec_loc, eigenvectors[: self.N])
            return eigenvalues, eigenvectors

    def _create_K_matrix(self) -> npt.NDArray[np.complex_]:  # noqa: N802
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

    def _slepian_integral(self) -> npt.NDArray[np.complex_]:
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
    def _slepian_matrix(
        dl: npt.NDArray[np.float_], L: int, N: int, G: npt.NDArray[np.complex_]
    ) -> npt.NDArray[np.complex_]:
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

        for ell in prange(L):
            for p in range(ell + 1):
                C1 = np.sqrt((2 * ell + 1) * (2 * p + 1)) / (4 * np.pi)

                for m in range(-ell, ell + 1):
                    ind_lm = ell * (ell + 1) + m

                    for q in range(-p, p + 1):
                        ind_pq = p * (p + 1) + q
                        row = m - q
                        C2 = (-1j) ** row
                        ind_r = 2 * N + row

                        for mp in range(-ell, ell + 1):
                            C3 = dl[ell, N + mp, N + m] * dl[ell, N + mp, N]
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
        eigendecomposition: tuple,
    ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.complex_]]:
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
