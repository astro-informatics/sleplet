from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pyssht as ssht
from multiprocess import Pool
from multiprocess.shared_memory import SharedMemory
from numpy import linalg as LA

from pys2sleplet.slepian.slepian_functions import SlepianFunctions
from pys2sleplet.utils.array_methods import fill_upper_triangle_of_hermitian_matrix
from pys2sleplet.utils.config import settings
from pys2sleplet.utils.harmonic_methods import create_spherical_harmonic
from pys2sleplet.utils.integration_methods import (
    calc_integration_weight,
    integrate_sphere,
)
from pys2sleplet.utils.mask_methods import create_mask_region
from pys2sleplet.utils.parallel_methods import split_L_into_chunks
from pys2sleplet.utils.region import Region
from pys2sleplet.utils.vars import ANNOTATION_COLOUR, ARROW_STYLE

_file_location = Path(__file__).resolve()
_arbitrary_path = _file_location.parents[2] / "data" / "slepian" / "arbitrary"


@dataclass
class SlepianArbitrary(SlepianFunctions):
    mask_name: str
    ncpu: int
    _mask_name: str = field(init=False, repr=False)
    _ncpu: int = field(default=settings.NCPU, init=False, repr=False)
    _region: Region = field(init=False, repr=False)
    _weight: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.region = Region(mask_name=self.mask_name)
        super().__post_init__()

    def _create_annotations(self) -> None:
        self.mask: np.ndarray
        thetas, phis = ssht.sample_positions(self.resolution, Grid=True)
        for i in range(len(self.mask)):
            for j in range(self.mask.shape[1]):
                if not self.mask[i, j] and thetas[i, j] <= np.pi / 3:
                    self._add_to_annotation(thetas[i, j], phis[i, j])

    def _create_fn_name(self) -> None:
        self.name = f"slepian_{self.mask_name}"

    def _create_mask(self) -> None:
        self.mask = create_mask_region(self.resolution, self.region)

    def _calculate_area(self) -> None:
        self.weight = calc_integration_weight(self.resolution)
        self.area = np.where(self.mask, self.weight, 0).sum()

    def _create_matrix_location(self) -> None:
        self.matrix_location = (
            _arbitrary_path / "matrices" / f"D_{self.mask_name}_L{self.L}"
        )

    def _solve_eigenproblem(self) -> None:
        eval_loc = self.matrix_location / "eigenvalues.npy"
        evec_loc = self.matrix_location / "eigenvectors.npy"
        if eval_loc.exists() and evec_loc.exists():
            self.eigenvalues = np.load(eval_loc)
            self.eigenvectors = np.load(evec_loc)
        else:
            D = self._create_D_matrix()
            self.eigenvalues, self.eigenvectors = self._clean_evals_and_evecs(
                LA.eigh(D)
            )
            if settings.SAVE_MATRICES:
                np.save(eval_loc, self.eigenvalues)
                np.save(evec_loc, self.eigenvectors)

    def _add_to_annotation(self, theta: float, phi: float) -> None:
        """
        add to annotation list for given theta
        """
        x, y, z = ssht.s2_to_cart(theta, phi)
        self.annotations.append(
            {**dict(x=x, y=y, z=z, arrowcolor=ANNOTATION_COLOUR), **ARROW_STYLE}
        )

    def _create_D_matrix(self) -> np.ndarray:
        """
        computes the D matrix either in parallel or serially
        """
        return self._matrix_serial() if self.ncpu == 1 else self._matrix_parallel()

    def _matrix_serial(self) -> np.ndarray:
        """
        computes the D matrix in serial
        """
        # initialise real and imaginary matrices
        D_r = np.zeros((self.L ** 2, self.L ** 2))
        D_i = np.zeros((self.L ** 2, self.L ** 2))

        for i in range(self.L ** 2):
            self._matrix_helper(D_r, D_i, i)

        # combine real and imaginary parts
        D = D_r + 1j * D_i

        # fill in remaining triangle section
        fill_upper_triangle_of_hermitian_matrix(D)

        return D

    def _matrix_parallel(self) -> np.ndarray:
        """
        computes the D matrix in parallel
        """
        # initialise real and imaginary matrices
        D_r = np.zeros((self.L ** 2, self.L ** 2))
        D_i = np.zeros((self.L ** 2, self.L ** 2))

        # create shared memory block
        shm_r = SharedMemory(create=True, size=D_r.nbytes)
        shm_i = SharedMemory(create=True, size=D_i.nbytes)
        # create a array backed by shared memory
        D_r_ext = np.ndarray(D_r.shape, dtype=D_r.dtype, buffer=shm_r.buf)
        D_i_ext = np.ndarray(D_i.shape, dtype=D_i.dtype, buffer=shm_i.buf)

        def func(chunk: List[int]) -> None:
            """
            calculate D matrix components for each chunk
            """
            # attach to the existing shared memory block
            ex_shm_r = SharedMemory(name=shm_r.name)
            ex_shm_i = SharedMemory(name=shm_i.name)
            D_r_int = np.ndarray(D_r.shape, dtype=D_r.dtype, buffer=ex_shm_r.buf)
            D_i_int = np.ndarray(D_i.shape, dtype=D_i.dtype, buffer=ex_shm_i.buf)

            # deal with chunk
            for i in chunk:
                self._matrix_helper(D_r_int, D_i_int, i)

            # clean up shared memory
            ex_shm_r.close()
            ex_shm_i.close()

        # split up L range to maximise effiency
        chunks = split_L_into_chunks(self.L ** 2, self.ncpu)

        # initialise pool and apply function
        with Pool(processes=self.ncpu) as p:
            p.map(func, chunks)

        # retrieve from parallel function
        D = D_r_ext + 1j * D_i_ext

        # Free and release the shared memory block at the very end
        shm_r.close()
        shm_r.unlink()
        shm_i.close()
        shm_i.unlink()

        # fill in remaining triangle section
        fill_upper_triangle_of_hermitian_matrix(D)

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

        for j in range(i + 1, self.L ** 2):
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
        flm = create_spherical_harmonic(self.L, i)
        glm = create_spherical_harmonic(self.L, j)
        return integrate_sphere(
            self.L,
            self.resolution,
            flm,
            glm,
            self.weight,
            mask_boosted=self.mask,
            glm_conj=True,
        )

    @staticmethod
    def _clean_evals_and_evecs(
        eigendecomposition: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
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

        # find repeating eigenvalues and ensure orthorgonality
        pairs = np.where(np.abs(np.diff(eigenvalues)) < 1e-14)[0] + 1
        eigenvectors[pairs] *= 1j

        return eigenvalues, eigenvectors

    @property  # type:ignore
    def mask_name(self) -> str:
        return self._mask_name

    @mask_name.setter
    def mask_name(self, mask_name: str) -> None:
        self._mask_name = mask_name

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
