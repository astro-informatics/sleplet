"""Contains the `MeshSlepian` class."""
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy import linalg as LA  # noqa: N812
from numpy import typing as npt
from pydantic.dataclasses import dataclass

import sleplet._array_methods
import sleplet._data.setup_pooch
import sleplet._integration_methods
import sleplet._parallel_methods
import sleplet._slepian_arbitrary_methods
import sleplet._validation
import sleplet._vars
from sleplet.meshes.mesh import Mesh

_logger = logging.getLogger(__name__)


@dataclass(config=sleplet._validation.Validation)
class MeshSlepian:
    """Creates Slepian object of a given mesh."""

    mesh: Mesh
    """A mesh object."""

    def __post_init_post_parse__(self) -> None:
        self.N = sleplet._slepian_arbitrary_methods.compute_mesh_shannon(self.mesh)
        self._compute_slepian_functions()

    def _compute_slepian_functions(self) -> None:
        """Computes the Slepian functions of the mesh."""
        _logger.info("computing slepian functions of mesh")

        # create filenames
        eigd_loc = (
            f"meshes_laplacians_slepian_functions_{self.mesh.name}_"
            f"b{self.mesh.mesh_eigenvalues.shape[0]}_N{self.N}"
        )
        eval_loc = f"{eigd_loc}_eigenvalues.npy"
        evec_loc = f"{eigd_loc}_eigenvectors.npy"

        try:
            self.slepian_eigenvalues = np.load(
                sleplet._data.setup_pooch.find_on_pooch_then_local(eval_loc),
            )
            self.slepian_functions = np.load(
                sleplet._data.setup_pooch.find_on_pooch_then_local(evec_loc),
            )
        except TypeError:
            self._compute_slepian_functions_from_scratch(eval_loc, evec_loc)

    def _compute_slepian_functions_from_scratch(self, eval_loc, evec_loc):
        D = self._create_D_matrix()
        _logger.info(
            f"Shannon number from vertices: {self.N}, "
            f"Trace of D matrix: {round(D.trace())}, "
            f"difference: {round(np.abs(self.N - D.trace()))}",
        )

        # fill in remaining triangle section
        sleplet._array_methods.fill_upper_triangle_of_hermitian_matrix(D)

        # solve eigenproblem
        (
            self.slepian_eigenvalues,
            self.slepian_functions,
        ) = self._clean_evals_and_evecs(LA.eigh(D))
        np.save(sleplet._vars.DATA_PATH / eval_loc, self.slepian_eigenvalues)
        np.save(sleplet._vars.DATA_PATH / evec_loc, self.slepian_functions[: self.N])

    def _create_D_matrix(self) -> npt.NDArray[np.float_]:  # noqa: N802
        """Computes the D matrix for the mesh eigenfunctions."""
        D = np.zeros(
            (self.mesh.mesh_eigenvalues.shape[0], self.mesh.mesh_eigenvalues.shape[0]),
        )

        D_ext, shm_ext = sleplet._parallel_methods.create_shared_memory_array(D)

        def func(chunk: list[int]) -> None:
            """Calculate D matrix components for each chunk."""
            D_int, shm_int = sleplet._parallel_methods.attach_to_shared_memory_block(
                D,
                shm_ext,
            )

            for i in chunk:
                _logger.info(f"start basis function: {i}")
                self._fill_D_elements(D_int, i)
                _logger.info(f"finish basis function: {i}")

            sleplet._parallel_methods.free_shared_memory(shm_int)

        # split up L range to maximise effiency
        ncpu = int(os.getenv("NCPU", "4"))
        _logger.info(f"Number of CPU={ncpu}")
        chunks = sleplet._parallel_methods.split_arr_into_chunks(
            self.mesh.mesh_eigenvalues.shape[0],
            ncpu,
        )

        # initialise pool and apply function
        with ThreadPoolExecutor(max_workers=ncpu) as e:
            e.map(func, chunks)

        # retrieve from parallel function
        D = D_ext.copy()

        # Free and release the shared memory block at the very end
        sleplet._parallel_methods.free_shared_memory(shm_ext)
        sleplet._parallel_methods.release_shared_memory(shm_ext)
        return D

    def _fill_D_elements(self, D: npt.NDArray[np.float_], i: int) -> None:  # noqa: N802
        """Fill in the D matrix elements using symmetries."""
        D[i][i] = self._integral(i, i)
        for j in range(i + 1, self.mesh.mesh_eigenvalues.shape[0]):
            D[j][i] = self._integral(j, i)

    def _integral(self, i: int, j: int) -> float:
        """Calculates the D integral between two mesh basis functions."""
        return sleplet._integration_methods.integrate_region_mesh(
            self.mesh.region,
            self.mesh.vertices,
            self.mesh.faces,
            self.mesh.basis_functions[i],
            self.mesh.basis_functions[j],
        )

    @staticmethod
    def _clean_evals_and_evecs(
        eigendecomposition: tuple,
    ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """Need eigenvalues and eigenvectors to be in a certain format."""
        # access values
        eigenvalues, eigenvectors = eigendecomposition

        # sort eigenvalues and eigenvectors in descending order of eigenvalues
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx].T
        return eigenvalues, eigenvectors
