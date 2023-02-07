from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyssht as ssht
from multiprocess import Pool
from numpy import linalg as LA

from sleplet.slepian.slepian_functions import SlepianFunctions
from sleplet.utils.array_methods import fill_upper_triangle_of_hermitian_matrix
from sleplet.utils.config import settings
from sleplet.utils.harmonic_methods import create_spherical_harmonic, invert_flm_boosted
from sleplet.utils.integration_methods import (
    calc_integration_weight,
    integrate_region_sphere,
)
from sleplet.utils.logger import logger
from sleplet.utils.mask_methods import create_mask_region
from sleplet.utils.parallel_methods import (
    attach_to_shared_memory_block,
    create_shared_memory_array,
    free_shared_memory,
    release_shared_memory,
    split_arr_into_chunks,
)
from sleplet.utils.region import Region
from sleplet.utils.slepian_arbitrary_methods import clean_evals_and_evecs
from sleplet.utils.vars import L_MAX_DEFAULT, L_MIN_DEFAULT

_file_location = Path(__file__).resolve()
_slepian_path = _file_location.parents[2] / "data" / "slepian"


@dataclass
class SlepianArbitrary(SlepianFunctions):
    mask_name: str
    L_min: int
    L_max: int
    _mask_name: str = field(init=False, repr=False)
    _L_max: int = field(default=settings.L_MAX, init=False, repr=False)
    _L_min: int = field(default=settings.L_MIN, init=False, repr=False)
    _region: Region = field(init=False, repr=False)
    _weight: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.region = Region(mask_name=self.mask_name)
        self.resolution = settings.SAMPLES * self.L
        super().__post_init__()

    def _create_fn_name(self) -> None:
        self.name = f"slepian_{self.mask_name}"

    def _create_mask(self) -> None:
        self.mask = create_mask_region(self.resolution, self.region)

    def _calculate_area(self) -> None:
        self.weight = calc_integration_weight(self.resolution)
        self.area = (self.mask * self.weight).sum()

    def _create_matrix_location(self) -> None:
        self.matrix_location = (
            _slepian_path / "eigensolutions" / f"D_{self.mask_name}_L{self.L}_N{self.N}"
        )

    def _solve_eigenproblem(self) -> None:
        eval_loc = self.matrix_location / "eigenvalues.npy"
        evec_loc = self.matrix_location / "eigenvectors.npy"
        if eval_loc.exists() and evec_loc.exists():
            logger.info("binaries found - loading...")
            self.eigenvalues = np.load(eval_loc)
            self.eigenvectors = np.load(evec_loc)
        else:
            D = self._create_D_matrix()

            # check whether the large job has been split up
            if (
                self.L_min != L_MIN_DEFAULT or self.L_max != self.L
            ) and settings.SAVE_MATRICES:
                logger.info("large job has been used, saving intermediate matrix")
                inter_loc = (
                    self.matrix_location / f"D_min{self.L_min}_max{self.L_max}.npy"
                )
                np.save(inter_loc, D)
                return

            # fill in remaining triangle section
            fill_upper_triangle_of_hermitian_matrix(D)

            # solve eigenproblem
            self.eigenvalues, self.eigenvectors = clean_evals_and_evecs(LA.eigh(D))
            if settings.SAVE_MATRICES:
                np.save(eval_loc, self.eigenvalues)
                np.save(evec_loc, self.eigenvectors[: self.N])

    def _create_D_matrix(self) -> np.ndarray:
        """
        computes the D matrix in parallel
        """
        # create dictionary for the integrals
        self._fields: dict[int, np.ndarray] = {}

        # initialise real and imaginary matrices
        D_r = np.zeros((self.L**2, self.L**2))
        D_i = np.zeros((self.L**2, self.L**2))

        D_r_ext, shm_r_ext = create_shared_memory_array(D_r)
        D_i_ext, shm_i_ext = create_shared_memory_array(D_i)

        def func(chunk: list[int]) -> None:
            """
            calculate D matrix components for each chunk
            """
            D_r_int, shm_r_int = attach_to_shared_memory_block(D_r, shm_r_ext)
            D_i_int, shm_i_int = attach_to_shared_memory_block(D_i, shm_i_ext)

            for i in chunk:
                logger.info(f"start ell: {i}")
                self._matrix_helper(D_r_int, D_i_int, i)
                logger.info(f"finish ell: {i}")

            free_shared_memory(shm_r_int, shm_i_int)

        # split up L range to maximise effiency
        chunks = split_arr_into_chunks(
            self.L_max**2, settings.NCPU, arr_min=self.L_min**2
        )

        # initialise pool and apply function
        with Pool(processes=settings.NCPU) as p:
            p.map(func, chunks)

        # retrieve from parallel function
        D = D_r_ext + 1j * D_i_ext

        # Free and release the shared memory block at the very end
        free_shared_memory(shm_r_ext, shm_i_ext)
        release_shared_memory(shm_r_ext, shm_i_ext)
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

        for j in range(i + 1, D_r.shape[0]):
            ell_j, m_j = ssht.ind2elm(j)
            # if possible to use previous calculations
            if m_i == 0 and m_j != 0 and ell_j < self.L:
                # if positive m then use conjugate relation
                if m_j > 0:
                    integral = self._integral(j, i)
                    D_r[j][i] = integral.real
                    D_i[j][i] = integral.imag
                    k = ssht.elm2ind(ell_j, -m_j)
                    D_r[k][i] = (-1) ** m_j * D_r[j][i]
                    D_i[k][i] = (-1) ** (m_j + 1) * D_i[j][i]
            else:
                integral = self._integral(j, i)
                D_r[j][i] = integral.real
                D_i[j][i] = integral.imag

    def _integral(self, i: int, j: int) -> complex:
        """
        calculates the D integral between two spherical harmonics
        """
        if i not in self._fields:
            self._fields[i] = invert_flm_boosted(
                create_spherical_harmonic(self.L, i), self.L, self.resolution
            )
        if j not in self._fields:
            self._fields[j] = invert_flm_boosted(
                create_spherical_harmonic(self.L, j), self.L, self.resolution
            )
        return integrate_region_sphere(
            self.mask, self.weight, self._fields[i], self._fields[j].conj()
        )

    @L_max.setter
    def L_max(self, L_max: int) -> None:
        if L_max > self.L:
            raise ValueError(f"L_max cannot be greater than L: {self.L}")
        if not isinstance(L_max, int):
            raise TypeError("L_max must be an integer")
        self._L_max = L_max if L_max != L_MAX_DEFAULT else self.L

    @L_min.setter
    def L_min(self, L_min: int) -> None:
        if L_min < 0:
            raise ValueError("L_min cannot be negative")
        if not isinstance(L_min, int):
            raise TypeError("L_min must be an integer")
        self._L_min = L_min
