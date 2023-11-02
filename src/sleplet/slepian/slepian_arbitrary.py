"""Contains the `SlepianArbitrary` class."""
import concurrent.futures
import logging
import os

import numpy as np
import numpy.linalg as LA  # noqa: N812
import numpy.typing as npt
import platformdirs
import pydantic
import typing_extensions

import pyssht as ssht

import sleplet._array_methods
import sleplet._data.setup_pooch
import sleplet._integration_methods
import sleplet._mask_methods
import sleplet._parallel_methods
import sleplet._slepian_arbitrary_methods
import sleplet._validation
import sleplet.harmonic_methods
import sleplet.slepian.region
from sleplet.slepian.slepian_functions import SlepianFunctions

_logger = logging.getLogger(__name__)

_SAMPLES = 2


@pydantic.dataclasses.dataclass(config=sleplet._validation.validation)
class SlepianArbitrary(SlepianFunctions):
    """Class to create an arbitrary Slepian region on the sphere."""

    mask_name: str
    """The name of the mask of the arbitrary region."""
    _weight: npt.NDArray[np.float_] = pydantic.Field(
        default_factory=lambda: np.empty(0),
        init_var=False,
        repr=False,
    )

    def __post_init__(self: typing_extensions.Self) -> None:
        self._resolution = _SAMPLES * self.L
        super().__post_init__()

    def _create_fn_name(self: typing_extensions.Self) -> str:
        return f"slepian_{self.mask_name}"

    def _create_region(self: typing_extensions.Self) -> "sleplet.slepian.region.Region":
        return sleplet.slepian.region.Region(mask_name=self.mask_name)

    def _create_mask(self: typing_extensions.Self) -> npt.NDArray[np.float_]:
        return sleplet._mask_methods.create_mask_region(self._resolution, self.region)

    def _calculate_area(self: typing_extensions.Self) -> float:
        self._weight = sleplet._integration_methods.calc_integration_weight(
            self._resolution,
        )
        return (self.mask * self._weight).sum()

    def _create_matrix_location(self: typing_extensions.Self) -> str:
        return f"slepian_eigensolutions_D_{self.mask_name}_L{self.L}_N{self.N}"

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
            eigenvalues, eigenvectors = self._solve_D_matrix(eval_loc, evec_loc)
        return eigenvalues, eigenvectors

    def _solve_D_matrix(  # noqa: N802
        self: typing_extensions.Self,
        eval_loc: str,
        evec_loc: str,
    ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.complex_]]:
        D = self._create_D_matrix()

        # fill in remaining triangle section
        sleplet._array_methods.fill_upper_triangle_of_hermitian_matrix(D)

        # solve eigenproblem
        (
            eigenvalues,
            eigenvectors,
        ) = sleplet._slepian_arbitrary_methods.clean_evals_and_evecs(LA.eigh(D))
        np.save(platformdirs.user_data_path() / eval_loc, eigenvalues)
        np.save(platformdirs.user_data_path() / evec_loc, eigenvectors[: self.N])
        return eigenvalues, eigenvectors

    def _create_D_matrix(  # noqa: N802
        self: typing_extensions.Self,
    ) -> npt.NDArray[np.complex_]:
        """Compute the D matrix in parallel."""
        # create dictionary for the integrals
        self._fields: dict[int, npt.NDArray[np.complex_ | np.float_]] = {}

        # initialise real and imaginary matrices
        D_r = np.zeros((self.L**2, self.L**2))
        D_i = np.zeros((self.L**2, self.L**2))

        D_r_ext, shm_r_ext = sleplet._parallel_methods.create_shared_memory_array(D_r)
        D_i_ext, shm_i_ext = sleplet._parallel_methods.create_shared_memory_array(D_i)

        def func(chunk: list[int]) -> None:
            """Calculate D matrix components for each chunk."""
            (
                D_r_int,
                shm_r_int,
            ) = sleplet._parallel_methods.attach_to_shared_memory_block(D_r, shm_r_ext)
            (
                D_i_int,
                shm_i_int,
            ) = sleplet._parallel_methods.attach_to_shared_memory_block(D_i, shm_i_ext)

            for i in chunk:
                msg = f"start ell: {i}"
                _logger.info(msg)
                self._matrix_helper(D_r_int, D_i_int, i)
                msg = f"finish ell: {i}"
                _logger.info(msg)

            sleplet._parallel_methods.free_shared_memory(shm_r_int, shm_i_int)

        # split up L range to maximise efficiency
        ncpu = int(os.getenv("NCPU", "4"))
        msg = f"Number of CPU={ncpu}"
        _logger.info(msg)
        chunks = sleplet._parallel_methods.split_arr_into_chunks(
            self.L**2,
            ncpu,
        )

        # initialise pool and apply function
        with concurrent.futures.ThreadPoolExecutor(max_workers=ncpu) as e:
            e.map(func, chunks)

        # retrieve from parallel function
        D = D_r_ext + 1j * D_i_ext

        # Free and release the shared memory block at the very end
        sleplet._parallel_methods.free_shared_memory(shm_r_ext, shm_i_ext)
        sleplet._parallel_methods.release_shared_memory(shm_r_ext, shm_i_ext)
        return D

    def _matrix_helper(
        self: typing_extensions.Self,
        D_r: npt.NDArray[np.float_],
        D_i: npt.NDArray[np.float_],
        i: int,
    ) -> None:
        """
        Use in both serial and parallel calculations.

        The hack with splitting into real and imaginary parts
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

    def _integral(self: typing_extensions.Self, i: int, j: int) -> complex:
        """Calculate the D integral between two spherical harmonics."""
        if i not in self._fields:
            self._fields[i] = sleplet.harmonic_methods.invert_flm_boosted(
                sleplet.harmonic_methods._create_spherical_harmonic(self.L, i),
                self.L,
                self._resolution,
            )
        if j not in self._fields:
            self._fields[j] = sleplet.harmonic_methods.invert_flm_boosted(
                sleplet.harmonic_methods._create_spherical_harmonic(self.L, j),
                self.L,
                self._resolution,
            )
        return sleplet._integration_methods.integrate_region_sphere(
            self.mask,
            self._weight,
            self._fields[i],
            self._fields[j].conj(),
        )
