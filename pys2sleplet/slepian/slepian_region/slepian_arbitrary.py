from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import pyssht as ssht
from multiprocess import Pool
from numpy import linalg as LA

from pys2sleplet.slepian.slepian_functions import SlepianFunctions
from pys2sleplet.utils.array_methods import fill_upper_triangle_of_hermitian_matrix
from pys2sleplet.utils.config import settings
from pys2sleplet.utils.harmonic_methods import (
    create_spherical_harmonic,
    invert_flm_boosted,
)
from pys2sleplet.utils.integration_methods import (
    calc_integration_weight,
    integrate_sphere,
)
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.mask_methods import create_mask_region
from pys2sleplet.utils.parallel_methods import (
    attach_to_shared_memory_block,
    create_shared_memory_array,
    free_shared_memory,
    release_shared_memory,
    split_L_into_chunks,
)
from pys2sleplet.utils.region import Region
from pys2sleplet.utils.slepian_arbitrary_methods import clean_evals_and_evecs
from pys2sleplet.utils.vars import (
    ANNOTATION_COLOUR,
    ARROW_STYLE,
    L_MAX_DEFAULT,
    L_MIN_DEFAULT,
    SAMPLING_SCHEME,
)

_file_location = Path(__file__).resolve()
_slepian_path = _file_location.parents[2] / "data" / "slepian"


@dataclass
class SlepianArbitrary(SlepianFunctions):
    mask_name: str
    ncpu: int
    L_min: int
    L_max: int
    _mask_name: str = field(init=False, repr=False)
    _L_max: int = field(default=settings.L_MAX, init=False, repr=False)
    _L_min: int = field(default=settings.L_MIN, init=False, repr=False)
    _ncpu: int = field(default=settings.NCPU, init=False, repr=False)
    _region: Region = field(init=False, repr=False)
    _weight: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.region = Region(mask_name=self.mask_name)
        self.resolution = settings.SAMPLES * self.L
        super().__post_init__()

    def _create_annotations(self) -> None:
        outline = np.load(
            _slepian_path
            / self.region.region_type
            / "outlines"
            / f"{self.mask_name}_outline.npy"
        )
        for o in outline:
            self.annotations.append(
                {
                    **dict(x=o[0], y=o[1], z=o[2], arrowcolor=ANNOTATION_COLOUR),
                    **ARROW_STYLE,
                }
            )

    def _create_fn_name(self) -> None:
        self.name = f"slepian_{self.mask_name}"

    def _create_mask(self) -> None:
        self.mask = create_mask_region(self.resolution, self.region)

    def _calculate_area(self) -> None:
        self.weight = calc_integration_weight(self.resolution)
        self.area = np.where(self.mask, self.weight, 0).sum()

    def _create_matrix_location(self) -> None:
        self.matrix_location = (
            _slepian_path
            / self.region.region_type
            / "matrices"
            / f"D_{self.mask_name}_L{self.L}_N{self.N}"
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
                np.save(eval_loc, self.eigenvalues[: self.N])
                np.save(evec_loc, self.eigenvectors[: self.N])

    def _create_D_matrix(self) -> np.ndarray:
        """
        computes the D matrix either in parallel or serially
        """
        fields = (
            self._inverse_transforms_serial()
            if self.ncpu == 1
            else self._inverse_transforms_parallel()
        )
        return (
            self._matrix_serial(fields)
            if self.ncpu == 1
            else self._matrix_parallel(fields)
        )

    def _inverse_transforms_serial(self) -> np.ndarray:
        """
        computes all the inverse transforms once serially
        """
        shape = (self.L ** 2,) + ssht.sample_shape(self.L, Method=SAMPLING_SCHEME)
        fields = np.zeros(shape, dtype=np.complex128)
        for p in range(self.L ** 2):
            logger.info(f"compute field: {p}")
            fields[p] = invert_flm_boosted(
                create_spherical_harmonic(self.L, p), self.L, self.resolution
            )
        return fields

    def _inverse_transforms_parallel(self) -> np.ndarray:
        """
        computes all the inverse transforms once in parallel
        """
        shape = (self.L ** 2,) + ssht.sample_shape(self.L, Method=SAMPLING_SCHEME)

        # initialise real and imaginary matrices
        fields_r = np.zeros(shape, dtype=np.complex128)
        fields_i = np.zeros(shape, dtype=np.complex128)

        fields_r_ext, shm_r_ext = create_shared_memory_array(fields_r)
        fields_i_ext, shm_i_ext = create_shared_memory_array(fields_i)

        def func(chunk: List[int]) -> None:
            """
            compute inverse transforms for each chunk
            """
            fields_r_int, shm_r_int = attach_to_shared_memory_block(fields_r, shm_r_ext)
            fields_i_int, shm_i_int = attach_to_shared_memory_block(fields_i, shm_i_ext)

            for p in chunk:
                logger.info(f"compute field: {p}")
                transformed = invert_flm_boosted(
                    create_spherical_harmonic(self.L, p), self.L, self.resolution
                )
                fields_r_int[p] = transformed.real
                fields_i_int[p] = transformed.imag

            free_shared_memory(shm_r_int, shm_i_int)

        # split up L range to maximise effiency
        chunks = split_L_into_chunks(self.L_max ** 2, self.ncpu, L_min=self.L_min ** 2)

        # initialise pool and apply function
        with Pool(processes=self.ncpu) as p:
            p.map(func, chunks)

        # retrieve from parallel function
        fields = fields_r_ext + 1j * fields_i_ext

        # Free and release the shared memory block at the very end
        free_shared_memory(shm_r_ext, shm_i_ext)
        release_shared_memory(shm_r_ext, shm_i_ext)
        return fields

    def _matrix_serial(self, fields: np.ndarray) -> np.ndarray:
        """
        computes the D matrix in serial
        """
        # initialise real and imaginary matrices
        D_r = np.zeros((self.L ** 2, self.L ** 2))
        D_i = np.zeros((self.L ** 2, self.L ** 2))

        for i in range(self.L_max ** 2 - self.L_min ** 2):
            logger.info(f"start ell: {i}")
            self._matrix_helper(D_r, D_i, fields, i)
            logger.info(f"finish ell: {i}")

        # combine real and imaginary parts
        return D_r + 1j * D_i

    def _matrix_parallel(self, fields: np.ndarray) -> np.ndarray:
        """
        computes the D matrix in parallel
        """
        # initialise real and imaginary matrices
        D_r = np.zeros((self.L ** 2, self.L ** 2))
        D_i = np.zeros((self.L ** 2, self.L ** 2))

        D_r_ext, shm_r_ext = create_shared_memory_array(D_r)
        D_i_ext, shm_i_ext = create_shared_memory_array(D_i)

        def func(chunk: List[int]) -> None:
            """
            calculate D matrix components for each chunk
            """
            D_r_int, shm_r_int = attach_to_shared_memory_block(D_r, shm_r_ext)
            D_i_int, shm_i_int = attach_to_shared_memory_block(D_i, shm_i_ext)

            for i in chunk:
                logger.info(f"start ell: {i}")
                self._matrix_helper(D_r_int, D_i_int, fields, i)
                logger.info(f"finish ell: {i}")

            free_shared_memory(shm_r_int, shm_i_int)

        # split up L range to maximise effiency
        chunks = split_L_into_chunks(self.L_max ** 2, self.ncpu, L_min=self.L_min ** 2)

        # initialise pool and apply function
        with Pool(processes=self.ncpu) as p:
            p.map(func, chunks)

        # retrieve from parallel function
        D = D_r_ext + 1j * D_i_ext

        # Free and release the shared memory block at the very end
        free_shared_memory(shm_r_ext, shm_i_ext)
        release_shared_memory(shm_r_ext, shm_i_ext)
        return D

    def _matrix_helper(
        self, D_r: np.ndarray, D_i: np.ndarray, fields: np.ndarray, i: int
    ) -> None:
        """
        used in both serial and parallel calculations

        the hack with splitting into real and imaginary parts
        is not required for the serial case but here for ease
        """
        # fill in diagonal components
        integral = self._integral(fields, i, i)
        D_r[i][i] = integral.real
        D_i[i][i] = integral.imag
        _, m_i = ssht.ind2elm(i)

        for j in range(i + 1, self.L ** 2):
            ell_j, m_j = ssht.ind2elm(j)
            # if possible to use previous calculations
            if m_i == 0 and m_j != 0 and ell_j < self.L:
                # if positive m then use conjugate relation
                if m_j > 0:
                    integral = self._integral(fields, j, i)
                    D_r[j][i] = integral.real
                    D_i[j][i] = integral.imag
                    k = ssht.elm2ind(ell_j, -m_j)
                    D_r[k][i] = (-1) ** m_j * D_r[j][i]
                    D_i[k][i] = (-1) ** (m_j + 1) * D_i[j][i]
            else:
                integral = self._integral(fields, j, i)
                D_r[j][i] = integral.real
                D_i[j][i] = integral.imag

    def _integral(self, fields: np.ndarray, i: int, j: int) -> complex:
        """
        calculates the D integral between two spherical harmonics
        """
        return integrate_sphere(
            self.resolution, fields[i], fields[j].conj(), self.weight, mask=self.mask
        )

    @property  # type:ignore
    def mask_name(self) -> str:
        return self._mask_name

    @mask_name.setter
    def mask_name(self, mask_name: str) -> None:
        self._mask_name = mask_name

    @property  # type:ignore
    def L_max(self) -> int:
        return self._L_max

    @L_max.setter
    def L_max(self, L_max: int) -> None:
        if isinstance(L_max, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            L_max = SlepianArbitrary._L_max
        if L_max > self.L:
            raise ValueError(f"L_max cannot be greater than L: {self.L}")
        if not isinstance(L_max, int):
            raise TypeError("L_max must be an integer")
        self._L_max = L_max if L_max != L_MAX_DEFAULT else self.L

    @property  # type:ignore
    def L_min(self) -> int:
        return self._L_min

    @L_min.setter
    def L_min(self, L_min: int) -> None:
        if isinstance(L_min, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            L_min = SlepianArbitrary._L_min
        if L_min < 0:
            raise ValueError("L_min cannot be negative")
        if not isinstance(L_min, int):
            raise TypeError("L_min must be an integer")
        self._L_min = L_min

    @property  # type:ignore
    def ncpu(self) -> int:
        return self._ncpu

    @ncpu.setter
    def ncpu(self, ncpu: int) -> None:
        if isinstance(ncpu, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            ncpu = SlepianArbitrary._ncpu
        self._ncpu = ncpu

    @property
    def region(self) -> Region:
        return self._region

    @region.setter
    def region(self, region: Region) -> None:
        self._region = region

    @property
    def weight(self) -> np.ndarray:
        return self._weight

    @weight.setter
    def weight(self, weight: np.ndarray) -> None:
        self._weight = weight
