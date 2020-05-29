from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pyssht as ssht
from multiprocess import Pool
from multiprocess.shared_memory import SharedMemory

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
    ncpu: int
    _mask_name: str = field(init=False, repr=False)
    _N: int = field(init=False, repr=False)
    _name_ending: str = field(init=False, repr=False)
    _ncpu: int = field(default=config.NCPU, init=False, repr=False)
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

    def _create_fn_name(self) -> None:
        self.name = f"slepian{self._name_ending}"

    def _create_mask(self) -> None:
        self.mask = self._load_mask()

    def _create_matrix_location(self) -> None:
        self.matrix_location = (
            _arbitrary_path / "matrices" / f"D_L{self.L}{self._name_ending}.npy"
        )

    def _solve_eigenproblem(self) -> None:
        logger.info("start solving eigenproblem")
        D = self._load_D_matrix()

        eigenvalues, eigenvectors = np.linalg.eigh(D)

        self.eigenvalues, self.eigenvectors = self._clean_evals_and_evecs(
            eigenvalues, eigenvectors
        )
        logger.info("finished solving eigenproblem")

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

    def _load_D_matrix(self) -> np.ndarray:
        """
        if the D matrix already exists load it
        otherwise create it and save the result
        """
        # check if matrix already exists
        if Path(self.matrix_location).exists():
            D = np.load(self.matrix_location)
        else:
            if self.ncpu == 1:
                D = self.matrix_serial()
            else:
                D = self.matrix_parallel()

            # save to speed up for future
            if config.SAVE_MATRICES:
                np.save(self.matrix_location, D)

        return D

    def _matrix_serial(self) -> np.ndarray:
        D = np.zeros((self._N, self._N), dtype=complex)

        for i in range(self._N):
            self._matrix_helper(D, i)

        return D

    def _matrix_parallel(self) -> np.ndarray:
        D = np.zeros((self._N, self._N), dtype=complex)

        # create shared memory block
        shm = SharedMemory(create=True, size=D.nbytes)
        # create a array backed by shared memory
        D_ext = np.ndarray(D.shape, dtype=D.dtype, buffer=shm.buf)

        def func(chunk: List[int]) -> None:
            """
            calculate D matrix components for each chunk
            """
            # attach to the existing shared memory block
            ex_shm = SharedMemory(name=shm.name)
            D_int = np.ndarray(D.shape, dtype=D.dtype, buffer=ex_shm.buf)

            # deal with chunk
            for i in chunk:
                self._matrix_helper(D_int, i)

            # clean up shared memory
            ex_shm.close()

        # split up L range to maximise effiency
        chunks = split_L_into_chunks(self._N, self.ncpu)

        # initialise pool and apply function
        with Pool(processes=self.ncpu) as p:
            p.map(func, chunks)

        # retrieve from parallel function
        D = D_ext

        # Free and release the shared memory block at the very end
        shm.close()
        shm.unlink()

        return D

    def _matrix_helper(self, D: np.ndarray, i: int) -> None:
        """
        used in both serial and parallel calculations

        the hack with splitting into real and imaginary parts
        is not required for the serial case but here for ease
        """
        # fill in diagonal components
        D[i][i] = self.integral(i, i)
        _, m_i = ssht.ind2elm(i)
        for j in range(i + 1, self._N):
            ell_j, m_j = ssht.ind2elm(j)
            # if possible to use previous calculations
            if m_i == 0 and m_j != 0 and ell_j < self.L:
                # if positive m then use conjugate relation
                if m_j > 0:
                    D[i][j] = self.integral(i, j)
                    D[j][i] = D[i][j].conj()
                    k = ssht.elm2ind(ell_j, -m_j)
                    D[i][k] = (-1) ** m_j * D[i][j].conj()
                    D[k][i] = D[i][k].conj()
            else:
                D[i][j] = self.integral(i, j)
                D[j][i] = D[i][j].conj()

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
        logger.info(f"mask_name={self.mask_name}")

    @property  # type: ignore
    def ncpu(self) -> int:
        return self._ncpu

    @ncpu.setter
    def ncpu(self, ncpu: int) -> None:
        if isinstance(ncpu, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            ncpu = SlepianArbitrary._ncpu
        self._ncpu = ncpu
        logger.info(f"ncpu={self.ncpu}")
