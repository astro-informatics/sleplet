import multiprocessing as mp
import multiprocessing.sharedctypes as sct
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pyssht as ssht

from pys2sleplet.slepian.slepian_functions import SlepianFunctions
from pys2sleplet.utils.config import config
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.parallel_methods import split_L_into_chunks
from pys2sleplet.utils.vars import SAMPLING_SCHEME

_file_location = Path(__file__).resolve()
_arbitrary_path = _file_location.parents[2] / "data" / "slepian" / "arbitrary"


@dataclass
class SlepianArbitrary(SlepianFunctions):
    mask_name: str
    _mask_name: str = field(init=False, repr=False)
    _N: int = field(init=False, repr=False)
    _name_ending: str = field(init=False, repr=False)
    _weight: np.ndarray = field(init=False, repr=False)
    _ylm: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._N = self.L * self.L
        self._name_ending = f"_{self.mask_name}"
        theta_grid, phi_grid = ssht.sample_positions(
            self.L, Grid=True, Method=SAMPLING_SCHEME
        )
        self._ylm = ssht.create_ylm(theta_grid, phi_grid, self.L)[:, self.mask]
        delta_theta = np.ediff1d(theta_grid[:, 0]).mean()
        delta_phi = np.ediff1d(phi_grid[0]).mean()
        self._weight = np.sin(theta_grid) * delta_theta * delta_phi
        super().__post_init__()

    def _create_annotations(self) -> None:
        pass

    def _create_fn_name(self) -> str:
        name = f"slepian{self._name_ending}"
        return name

    def _create_mask(self) -> np.ndarray:
        mask = self._load_mask()
        return mask

    def _create_matrix_location(self) -> Path:
        location = _arbitrary_path / "matrices" / f"D_L{self.L}{self._name_ending}.npy"
        return location

    def _solve_eigenproblem(self) -> Tuple[np.ndarray, np.ndarray]:
        logger.info("start solving eigenproblem")
        if config.NCPU == 1:
            D = self.matrix_serial()
        else:
            D = self.matrix_parallel()

        eigenvalues, eigenvectors = np.linalg.eigh(D)

        eigenvalues, eigenvectors = self._clean_evals_and_evecs(
            eigenvalues, eigenvectors
        )
        logger.info("finished solving eigenproblem")
        return eigenvalues, eigenvectors

    def _load_mask(self) -> np.ndarray:
        """
        attempts to read the mask from the config file
        """
        location = _arbitrary_path / "masks" / self.mask_name
        try:
            mask = np.load(location)
        except FileNotFoundError:
            logger.error(f"can not find the file: {self.mask_name}")
            raise
        return mask

    def _matrix_serial(self) -> np.ndarray:
        # initialise real and imaginary matrices
        real = np.zeros((self._N, self._N))
        imag = np.zeros((self._N, self._N))

        for i in range(self._N):
            self._matrix_helper(real, imag, i)

        # retrieve real and imag components
        D = real + 1j * imag

        return D

    def _matrix_parallel(self) -> np.ndarray:
        # initialise real and imaginary matrices
        real = np.zeros((self._N, self._N))
        imag = np.zeros((self._N, self._N))

        # create arrays to store final and intermediate steps
        result_r = np.ctypeslib.as_ctypes(real)
        result_i = np.ctypeslib.as_ctypes(imag)
        shared_array_r = sct.RawArray(result_r._type_, result_r)
        shared_array_i = sct.RawArray(result_i._type_, result_i)

        def func(chunk: List[int]) -> None:
            """
            calculate D matrix components for each chunk
            """
            # store real and imag parts separately
            tmp_r = np.ctypeslib.as_array(shared_array_r)
            tmp_i = np.ctypeslib.as_array(shared_array_i)

            # deal with chunk
            for i in chunk:
                self._matrix_helper(tmp_r, tmp_i, i)

        # split up L range to maximise effiency
        chunks = split_L_into_chunks(self._N, config.NCPU)

        # initialise pool and apply function
        with mp.Pool(processes=config.NCPU) as p:
            p.map(func, chunks)

        # retrieve real and imag components
        result_r = np.ctypeslib.as_array(shared_array_r)
        result_i = np.ctypeslib.as_array(shared_array_i)
        D = result_r + 1j * result_i

        return D

    def _matrix_helper(self, D_r: np.ndarray, D_i: np.ndarray, i: int) -> None:
        """
        used in both serial and parallel calculations

        the hack with splitting into real and imaginary parts
        is not required for the serial case but here for ease
        """
        # fill in diagonal components
        integral = self._integral(i, i)
        D_r[i][i] = integral.real
        D_i[i][i] = integral.imag
        _, m_i = ssht.ind2elm(i)

        for j in range(i + 1, self._N):
            ell_j, m_j = ssht.ind2elm(j)
            # if possible to use previous calculations
            if m_i == 0 and m_j != 0 and ell_j < self.L:
                # if positive m then use conjugate relation
                if m_j > 0:
                    integral = self._integral(i, j)
                    D_r[i][j] = integral.real
                    D_i[i][j] = integral.imag
                    D_r[j][i] = D_r[i][j]
                    D_i[j][i] = -D_i[i][j]
                    k = ssht.elm2ind(ell_j, -m_j)
                    D_r[i][k] = (-1) ** m_j * D_r[i][j]
                    D_i[i][k] = (-1) ** (m_j + 1) * D_i[i][j]
                    D_r[k][i] = D_r[i][k]
                    D_i[k][i] = -D_i[i][k]
            else:
                integral = self._integral(i, j)
                D_r[i][j] = integral.real
                D_i[i][j] = integral.imag
                D_r[j][i] = D_r[i][j]
                D_i[j][i] = -D_i[i][j]

    def _integral(self, i: int, j: int) -> complex:
        F = (self._f(i, j) * self._weight()).sum()
        return F

    def _f(self, i: int, j: int) -> np.ndarray:
        f = self._ylm[i] * self._ylm[j].conj()
        return f

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

        # find repeating eigenvalues and ensure orthorgonality
        pairs = np.where(np.abs(np.diff(eigenvalues)) < 1e-14)[0] + 1
        eigenvectors[pairs] *= 1j

        return eigenvalues, eigenvectors

    @property  # type: ignore
    def mask_name(self) -> str:
        return self._mask_name

    @mask_name.setter
    def mask_name(self, mask_name: str) -> None:
        self._mask_name = mask_name
        logger.info(f"mask_name={mask_name}")
