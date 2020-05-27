import multiprocessing.sharedctypes as sct
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pyssht as ssht

from pys2sleplet.slepian.slepian_functions import SlepianFunctions
from pys2sleplet.utils.config import config
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.parallel_methods import split_L_into_chunks
from pys2sleplet.utils.string_methods import angle_as_degree, multiples_of_pi
from pys2sleplet.utils.vars import (
    ARROW_STYLE,
    PHI_MAX_DEFAULT,
    PHI_MIN_DEFAULT,
    SAMPLING_SCHEME,
    THETA_MAX_DEFAULT,
    THETA_MIN_DEFAULT,
)

_file_location = Path(__file__).resolve()


@dataclass
class SlepianLimitLatLong(SlepianFunctions):
    theta_min: float
    theta_max: float
    phi_min: float
    phi_max: float
    _phi_max: float = field(default=PHI_MAX_DEFAULT, init=False, repr=False)
    _phi_min: float = field(default=PHI_MIN_DEFAULT, init=False, repr=False)
    _theta_max: float = field(default=THETA_MAX_DEFAULT, init=False, repr=False)
    _theta_min: float = field(default=THETA_MIN_DEFAULT, init=False, repr=False)
    _name_ending: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._name_ending = (
            f"_theta{angle_as_degree(self.theta_min)}"
            f"-{angle_as_degree(self.theta_max)}"
            f"_phi{angle_as_degree(self.phi_min)}"
            f"-{angle_as_degree(self.phi_max)}"
        )
        super().__post_init__()

    def _create_annotations(self) -> None:
        p1, p2, t1, t2 = (
            np.array([self.phi_min]),
            np.array([self.phi_max]),
            np.array([self.theta_min]),
            np.array([self.theta_max]),
        )
        p3, p4, t3, t4 = (
            (p1 + 2 * p2) / 3,
            (2 * p1 + p2) / 3,
            (t1 + 2 * t2) / 3,
            (2 * t1 + t2) / 3,
        )
        for t in [t1, t2, t3, t4]:
            t_condition = (t == [t3, t4]).any()
            for p in [p1, p2, p3, p4]:
                p_condition = (p == [p3, p4]).any()
                if not (t_condition and p_condition):
                    x, y, z = ssht.s2_to_cart(t, p)
                    self.annotations.append(
                        {
                            **dict(x=x[0], y=y[0], z=z[0], arrowcolor="black"),
                            **ARROW_STYLE,
                        }
                    )

    def _create_fn_name(self) -> str:
        name = f"slepian{self._name_ending}"
        return name

    def _create_mask(self) -> np.ndarray:
        theta_grid, phi_grid = ssht.sample_positions(
            self.L, Grid=True, Method=SAMPLING_SCHEME
        )
        mask = (
            (theta_grid >= self.theta_min)
            & (theta_grid <= self.theta_max)
            & (phi_grid >= self.phi_min)
            & (phi_grid <= self.phi_max)
        )
        return mask

    def _create_matrix_location(self) -> Path:
        location = (
            _file_location.parents[2]
            / "data"
            / "slepian"
            / "lat_lon"
            / f"D_L{self.L}{self._name_ending}.npy"
        )
        return location

    def _solve_eigenproblem(self) -> Tuple[np.ndarray, np.ndarray]:
        K = self._load_K_matrix()

        eigenvalues, eigenvectors = np.linalg.eigh(K)

        eigenvalues, eigenvectors = self._clean_evals_and_evecs(
            eigenvalues, eigenvectors
        )
        return eigenvalues, eigenvectors

    def _load_K_matrix(self) -> np.ndarray:
        """
        if the K matrix already exists load it
        otherwise create it and save the result
        """
        # check if matrix already exists
        if Path(self.matrix_location).exists():
            K = np.load(self.matrix_location)
        else:
            # Compute sub-integral matrix
            G = self._slepian_integral()

            # Compute Slepian matrix
            if config.NCPU == 1:
                K = self._slepian_matrix_serial(G)
            else:
                K = self._slepian_matrix_parallel(G)

            # save to speed up for future
            if config.SAVE_MATRICES:
                np.save(self.matrix_location, K)

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
        G = np.zeros((4 * self.L - 3, 4 * self.L - 3), dtype=complex)

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

    def _slepian_matrix_serial(self, G: np.ndarray) -> np.ndarray:
        """
        Syntax:
        K = _slepian_matrix_serial(G)

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
        dl_array = ssht.generate_dl(np.pi / 2, self.L)

        # initialise real and imaginary matrices
        real = np.zeros((self.L * self.L, self.L * self.L))
        imag = np.zeros((self.L * self.L, self.L * self.L))

        for l in range(self.L):
            self._slepian_matrix_helper(real, imag, l, dl_array, G)

        # retrieve real and imag components
        K = real + 1j * imag

        # fill in remaining triangle section
        i_upper = np.triu_indices(K.shape[0])
        K[i_upper] = K.T[i_upper].conj()

        return K

    def _slepian_matrix_parallel(self, G: np.ndarray) -> np.ndarray:
        """
        Syntax:
        K = _slepian_matrix_parallel(G)

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
        dl_array = ssht.generate_dl(np.pi / 2, self.L)

        # initialise real and imaginary matrices
        real = np.zeros((self.L * self.L, self.L * self.L))
        imag = np.zeros((self.L * self.L, self.L * self.L))

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
                self._slepian_matrix_helper(tmp_r, tmp_i, l, dl_array, G)

        # split up L range to maximise effiency
        chunks = split_L_into_chunks(self.L, config.NCPU)

        # initialise pool and apply function
        with Pool(processes=config.NCPU) as p:
            p.map(func, chunks)

        # retrieve real and imag components
        result_r = np.ctypeslib.as_array(shared_array_r)
        result_i = np.ctypeslib.as_array(shared_array_i)
        K = result_r + 1j * result_i

        # fill in remaining triangle section
        i_upper = np.triu_indices(K.shape[0])
        K[i_upper] = K.T[i_upper].conj()

        return K

    def _slepian_matrix_helper(
        self,
        K_r: np.ndarray,
        K_i: np.ndarray,
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
                    ind_r = 2 * (self.L - 1) + row

                    for mp in range(-l, l + 1):
                        C3 = (
                            dl[self.L - 1 + mp, self.L - 1 + m]
                            * dl[self.L - 1 + mp, self.L - 1]
                        )
                        S1 = 0

                        for qp in range(-p, p + 1):
                            col = mp - qp
                            C4 = (
                                dp[self.L - 1 + qp, self.L - 1 + q]
                                * dp[self.L - 1 + qp, self.L - 1]
                            )
                            ind_c = 2 * (self.L - 1) + col
                            S1 += C4 * G[ind_r, ind_c]

                        idx = (l * (l + 1) + m, p * (p + 1) + q)
                        K_r[idx] += (C3 * S1).real
                        K_i[idx] += (C3 * S1).imag

                    idx = (l * (l + 1) + m, p * (p + 1) + q)
                    real, imag = K_r[idx], K_i[idx]
                    K_r[idx] = real * (C1 * C2).real - imag * (C1 * C2).imag
                    K_i[idx] = real * (C1 * C2).imag + imag * (C1 * C2).real

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

    @property  # type:ignore
    def phi_max(self) -> float:
        return self._phi_max

    @phi_max.setter
    def phi_max(self, phi_max: float) -> None:
        if isinstance(phi_max, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            phi_max = SlepianLimitLatLong._phi_max
        if phi_max < PHI_MIN_DEFAULT:
            raise ValueError("phi_max cannot be negative")
        if phi_max >= PHI_MAX_DEFAULT:
            raise ValueError(
                f"phi_max cannot be greater than or equal to {multiples_of_pi(PHI_MAX_DEFAULT)}"
            )
        self._phi_max = phi_max
        logger.info(f"phi_max={phi_max}")

    @property  # type:ignore
    def phi_min(self) -> float:
        return self._phi_min

    @phi_min.setter
    def phi_min(self, phi_min: float) -> None:
        if isinstance(phi_min, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            phi_min = SlepianLimitLatLong._phi_min
        if phi_min < PHI_MIN_DEFAULT:
            raise ValueError("phi_min cannot be negative")
        if phi_min >= PHI_MAX_DEFAULT:
            raise ValueError(
                f"phi_min cannot be greater than or equal to {multiples_of_pi(PHI_MAX_DEFAULT)}"
            )
        self._phi_min = phi_min
        logger.info(f"phi_min={phi_min}")

    @property  # type:ignore
    def theta_max(self) -> float:
        return self._theta_max

    @theta_max.setter
    def theta_max(self, theta_max: float) -> None:
        if isinstance(theta_max, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            theta_max = SlepianLimitLatLong._theta_max
        if theta_max < THETA_MIN_DEFAULT:
            raise ValueError("theta_max cannot be negative")
        if theta_max > THETA_MAX_DEFAULT:
            raise ValueError(
                f"theta_max cannot be greater than {multiples_of_pi(THETA_MAX_DEFAULT)}"
            )
        self._theta_max = theta_max
        logger.info(f"theta_max={theta_max}")

    @property  # type: ignore
    def theta_min(self) -> float:
        return self._theta_min

    @theta_min.setter
    def theta_min(self, theta_min: float) -> None:
        if isinstance(theta_min, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            theta_min = SlepianLimitLatLong._theta_min
        if theta_min < THETA_MIN_DEFAULT:
            raise ValueError("theta_min cannot be negative")
        if theta_min > THETA_MAX_DEFAULT:
            raise ValueError(
                f"theta_min cannot be greater than {multiples_of_pi(THETA_MAX_DEFAULT)}"
            )
        self._theta_min = theta_min
        logger.info(f"theta_min={theta_min}")
