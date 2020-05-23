import multiprocessing.sharedctypes as sct
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pyssht as ssht

from pys2sleplet.slepian.slepian_region.slepian_specific import SlepianSpecific
from pys2sleplet.utils.config import config
from pys2sleplet.utils.parallel_methods import split_L_into_chunks
from pys2sleplet.utils.vars import ARROW_STYLE, SAMPLING_SCHEME

_file_location = Path(__file__).resolve()


@dataclass
class SlepianLimitLatLong(SlepianSpecific):
    theta_min: float
    theta_max: float
    phi_min: float
    phi_max: float
    _name_ending: str = field(
        default=(
            f"_theta{config.THETA_MIN}-{config.THETA_MAX}"
            f"_phi{config.PHI_MIN}-{config.PHI_MAX}"
        ),
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_annotations(self) -> None:
        p1, p2, t1, t2 = (
            np.array(self.phi_min),
            np.array(self.phi_max),
            np.array(self.theta_min),
            np.array(self.theta_max),
        )
        p3, p4, t3, t4 = (
            (p1 + 2 * p2) / 3,
            (2 * p1 + p2) / 3,
            (t1 + 2 * t2) / 3,
            (2 * t1 + t2) / 3,
        )
        for t in [t1, t2, t3, t4]:
            t_condition = t in {t3, t4}
            for p in [p1, p2, p3, p4]:
                p_condition = p in {p3, p4}
                if not (t_condition and p_condition):
                    x, y, z = ssht.s2_to_cart(t, p)
                    self.annotation.append(
                        {**dict(x=x, y=y, z=z, arrowcolor="black"), **ARROW_STYLE}
                    )

    def _create_fn_name(self) -> str:
        name = f"slepian{self._name_ending}"
        return name

    def _create_mask(self, L: int) -> np.ndarray:
        theta_grid, phi_grid = ssht.sample_positions(
            L, Grid=True, Method=SAMPLING_SCHEME
        )
        mask = (
            (theta_grid >= self.theta_min)
            & (theta_grid <= self.theta_max)
            & (phi_grid >= self.phi_min)
            & (phi_grid <= self.phi_max)
        )
        return mask

    def _create_matrix_location(self, L: int) -> Path:
        location = (
            _file_location.parents[3]
            / "data"
            / "slepian"
            / "lat_lon"
            / f"D_L{L}{self._name_ending}.npy"
        )
        return location

    def _solve_eigenproblem(self, L: int) -> Tuple[np.ndarray, np.ndarray]:
        K = self._load_K_matrix(L)

        eigenvalues, eigenvectors = np.linalg.eigh(K)

        eigenvalues, eigenvectors = self._clean_evals_and_evecs(
            eigenvalues, eigenvectors
        )
        return eigenvalues, eigenvectors

    def _load_K_matrix(self, L: int) -> np.ndarray:
        """
        if the K matrix already exists load it
        otherwise create it and save the result
        """
        # check if matrix already exists
        if Path(self.matrix_location).exists():
            K = np.load(self.matrix_location)
        else:
            # Compute sub-integral matrix
            G = self._slepian_integral(L)

            # Compute Slepian matrix
            if config.NCPU == 1:
                K = self._slepian_matrix_serial(L, G)
            else:
                K = self._slepian_matrix_parallel(L, G, config.NCPU)

            # save to speed up for future
            if config.SAVE_MATRICES:
                np.save(self.matrix_location, K)

        return K

    @staticmethod
    def _clean_evals_and_evecs(
        eigenvalues: np.ndarray, eigenvectors: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        need eigenvalues and eigenvectors to be in a certain format
        """
        # eigenvalues should be real
        eigenvalues = eigenvalues.real

        # Sort eigenvalues and eigenvectors in descending order of eigenvalues
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx].conj().T

        # ensure first element of each eigenvector is positive
        eigenvectors *= np.where(eigenvectors[:, 0] < 0, -1, 1)[:, np.newaxis]

        return eigenvalues, eigenvectors

    def _slepian_integral(self, L: int) -> np.ndarray:
        """
        Syntax:
        G = _slepian_integral(L)

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
        G = np.zeros((4 * L - 3, 4 * L - 3), dtype=complex)

        def helper(row: int, col: int, S: float) -> None:
            """
            Using conjugate symmetry property to reduce the number of iterations
            """
            try:
                Q = (1 / (col * col - 1)) * (
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

            G[2 * (L - 1) + row, 2 * (L - 1) + col] = Q * S
            G[2 * (L - 1) - row, 2 * (L - 1) - col] = G[
                2 * (L - 1) + row, 2 * (L - 1) + col
            ].conj()

        # row = 0
        S = self.phi_max - self.phi_min
        for col in range(-2 * (L - 1), 1):
            helper(0, col, S)

        # row != 0
        for row in range(-2 * (L - 1), 0):
            S = (1j / row) * (
                np.exp(1j * row * self.phi_min) - np.exp(1j * row * self.phi_max)
            )
            for col in range(-2 * (L - 1), 2 * (L - 1) + 1):
                helper(row, col, S)

        return G

    def _slepian_matrix_serial(self, L: int, G: np.ndarray) -> np.ndarray:
        """
        Syntax:
        K = _slepian_matrix_serial(L, G)

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
        dl_array = ssht.generate_dl(np.pi / 2, L)

        # initialise real and imaginary matrices
        real = np.zeros((L * L, L * L), dtype=complex)
        imag = np.zeros((L * L, L * L), dtype=complex)

        for l in range(L):
            self._slepian_matrix_helper(real, imag, L, l, dl_array, G)

        # retrieve real and imag components
        K = real + 1j * imag

        # fill in remaining triangle section
        i_upper = np.triu_indices(K.shape[0])
        K[i_upper] = K.T[i_upper].conj()

        return K

    def _slepian_matrix_parallel(self, L: int, G: np.ndarray, ncpu: int) -> np.ndarray:
        """
        Syntax:
        K = _slepian_matrix_parallel(L, G, ncpu)

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
        dl_array = ssht.generate_dl(np.pi / 2, L)

        # initialise real and imaginary matrices
        real = np.zeros((L * L, L * L))
        imag = np.zeros((L * L, L * L))

        # create arrays to store final and intermediate steps
        result_r = np.ctypeslib.as_ctypes(real)
        result_i = np.ctypeslib.as_ctypes(imag)
        shared_array_r = sct.RawArray(result_r._type_, result_r)
        shared_array_i = sct.RawArray(result_i._type_, result_i)

        def func(chunk: List[int]) -> None:
            """
            calculate K matrix components for each chunk
            """
            # store real and imag parts separately
            tmp_r = np.ctypeslib.as_array(shared_array_r)
            tmp_i = np.ctypeslib.as_array(shared_array_i)

            # deal with chunk
            for l in chunk:
                self._slepian_matrix_helper(tmp_r, tmp_i, L, l, dl_array, G)

        # split up L range to maximise effiency
        chunks = split_L_into_chunks(L, ncpu)

        # initialise pool and apply function
        with Pool(processes=ncpu) as p:
            p.map(func, chunks)

        # retrieve real and imag components
        result_r = np.ctypeslib.as_array(shared_array_r)
        result_i = np.ctypeslib.as_array(shared_array_i)
        K = result_r + 1j * result_i

        # fill in remaining triangle section
        i_upper = np.triu_indices(K.shape[0])
        K[i_upper] = K.T[i_upper].conj()

        return K

    @staticmethod
    def _slepian_matrix_helper(
        K_r: np.ndarray,
        K_i: np.ndarray,
        L: int,
        l: int,
        dl_array: np.ndarray,
        G: np.ndarray,
    ) -> None:
        """
        used in both serial and parallel calculations

        the hack with splitting into real and imaginary parts
        is not required for the serial case but here for ease
        """
        dl = dl_array[l]

        for p in range(l + 1):
            dp = dl_array[p]
            C1 = np.sqrt((2 * l + 1) * (2 * p + 1)) / (4 * np.pi)

            for m in range(-l, l + 1):
                for q in range(-p, p + 1):

                    row = m - q
                    C2 = (-1j) ** row
                    ind_r = 2 * (L - 1) + row

                    for mp in range(-l, l + 1):
                        C3 = dl[L - 1 + mp, L - 1 + m] * dl[L - 1 + mp, L - 1]
                        S1 = 0

                        for qp in range(-p, p + 1):
                            col = mp - qp
                            C4 = dp[L - 1 + qp, L - 1 + q] * dp[L - 1 + qp, L - 1]
                            ind_c = 2 * (L - 1) + col
                            S1 += C4 * G[ind_r, ind_c]

                        idx = (l * (l + 1) + m, p * (p + 1) + q)
                        K_r[idx] += (C3 * S1).real
                        K_i[idx] += (C3 * S1).imag

                    idx = (l * (l + 1) + m, p * (p + 1) + q)
                    real, imag = K_r[idx], K_i[idx]
                    K_r[idx] = real * (C1 * C2).real - imag * (C1 * C2).imag
                    K_i[idx] = real * (C1 * C2).imag + imag * (C1 * C2).real
