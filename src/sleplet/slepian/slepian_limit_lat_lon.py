"""Contains the `SlepianLimitLatLon` class."""

import numba
import numpy as np
import numpy.linalg as LA  # noqa: N812
import numpy.typing as npt
import platformdirs
import pydantic
import typing_extensions

import pyssht as ssht

import sleplet._array_methods
import sleplet._data.setup_pooch
import sleplet._mask_methods
import sleplet._validation
import sleplet._vars
import sleplet.slepian.region
from sleplet.slepian.slepian_functions import SlepianFunctions


@pydantic.dataclasses.dataclass(config=sleplet._validation.validation, kw_only=True)
class SlepianLimitLatLon(SlepianFunctions):
    """Class to create a limited latitude longitude Slepian region on the sphere."""

    phi_max: float = sleplet._vars.PHI_MAX_DEFAULT
    r"""Maximum \(\phi\) value."""
    phi_min: float = sleplet._vars.PHI_MIN_DEFAULT
    r"""Minimum \(\phi\) value."""
    theta_max: float = sleplet._vars.THETA_MAX_DEFAULT
    r"""Maximum \(\theta\) value."""
    theta_min: float = sleplet._vars.THETA_MIN_DEFAULT
    r"""Minimum \(\theta\) value."""

    def __post_init__(self: typing_extensions.Self) -> None:
        super().__post_init__()

    def _create_fn_name(self: typing_extensions.Self) -> str:
        return f"slepian_{self.region._name_ending}"

    def _create_region(self: typing_extensions.Self) -> "sleplet.slepian.region.Region":
        return sleplet.slepian.region.Region(
            theta_min=self.theta_min,
            theta_max=self.theta_max,
            phi_min=self.phi_min,
            phi_max=self.phi_max,
        )

    def _create_mask(self: typing_extensions.Self) -> npt.NDArray[np.float_]:
        return sleplet._mask_methods.create_mask_region(self.L, self.region)

    def _calculate_area(self: typing_extensions.Self) -> float:
        return (self.phi_max - self.phi_min) * (
            np.cos(self.theta_min) - np.cos(self.theta_max)
        )

    def _create_matrix_location(self: typing_extensions.Self) -> str:
        return (
            f"slepian_eigensolutions_D_{self.region._name_ending}_L{self.L}_N{self.N}"
        )

    def _solve_eigenproblem(
        self: typing_extensions.Self,
    ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.complex_]]:
        eval_loc = f"{self.matrix_location}_eigenvalues.npy"
        evec_loc = f"{self.matrix_location}_eigenvectors.npy"
        try:
            eigenvalues = np.load(
                sleplet._data.setup_pooch.find_on_pooch_then_local(eval_loc),
            )
            eigenvectors = np.load(
                sleplet._data.setup_pooch.find_on_pooch_then_local(evec_loc),
            )
        except TypeError:
            K = self._create_K_matrix()
            eigenvalues, eigenvectors = self._clean_evals_and_evecs(LA.eigh(K))
            np.save(platformdirs.user_data_path() / eval_loc, eigenvalues)
            np.save(platformdirs.user_data_path() / evec_loc, eigenvectors[: self.N])
        return eigenvalues, eigenvectors

    def _create_K_matrix(  # noqa: N802
        self: typing_extensions.Self,
    ) -> npt.NDArray[np.complex_]:
        """Compute the K matrix."""
        # Compute sub-integral matrix
        G = self._slepian_integral()

        # Compute Slepian matrix
        dl_array = ssht.generate_dl(np.pi / 2, self.L)
        K = self._slepian_matrix(dl_array, self.L, self.L - 1, G)
        sleplet._array_methods.fill_upper_triangle_of_hermitian_matrix(K)

        return K

    def _slepian_integral(self: typing_extensions.Self) -> npt.NDArray[np.complex_]:
        """
        Syntax:
        G = _slepian_integral().

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
            """Use conjugate symmetry property to reduce the number of iterations."""
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
                2 * (self.L - 1) + row,
                2 * (self.L - 1) + col,
            ].conj()

        # for row equal to zero
        S = self.phi_max - self.phi_min
        for col in range(-2 * (self.L - 1), 1):
            helper(0, col, S)

        # for row not equal to zero
        for row in range(-2 * (self.L - 1), 0):
            S = (1j / row) * (
                np.exp(1j * row * self.phi_min) - np.exp(1j * row * self.phi_max)
            )
            for col in range(-2 * (self.L - 1), 2 * (self.L - 1) + 1):
                helper(row, col, S)

        return G

    @staticmethod
    @numba.njit(parallel=True, fastmath=True)
    def _slepian_matrix(
        dl: npt.NDArray[np.float_],
        L: int,
        N: int,
        G: npt.NDArray[np.complex_],
    ) -> npt.NDArray[np.complex_]:
        """
        Syntax:
        K = _slepian_matrix(dl, L, N, G).

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

        for ell in numba.prange(L):
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
        eigendecomposition: tuple[npt.NDArray[np.complex_], npt.NDArray[np.complex_]],
    ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.complex_]]:
        """Need eigenvalues and eigenvectors to be in a certain format."""
        # access values
        eigenvalues_complex, eigenvectors = eigendecomposition

        # eigenvalues should be real
        eigenvalues = eigenvalues_complex.real

        # Sort eigenvalues and eigenvectors in descending order of eigenvalues
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx].conj().T

        # ensure first element of each eigenvector is positive
        eigenvectors *= np.where(eigenvectors[:, 0] < 0, -1, 1)[:, np.newaxis]

        return eigenvalues, eigenvectors
