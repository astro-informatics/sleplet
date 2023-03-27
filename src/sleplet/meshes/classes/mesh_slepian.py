from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from numpy import linalg as LA  # noqa: N812
from numpy import typing as npt
from pydantic.dataclasses import dataclass

from sleplet import logger
from sleplet.data.setup_pooch import find_on_pooch_then_local
from sleplet.meshes.classes.mesh import Mesh
from sleplet.utils.array_methods import fill_upper_triangle_of_hermitian_matrix
from sleplet.utils.config import settings
from sleplet.utils.integration_methods import integrate_region_mesh
from sleplet.utils.parallel_methods import (
    attach_to_shared_memory_block,
    create_shared_memory_array,
    free_shared_memory,
    release_shared_memory,
    split_arr_into_chunks,
)
from sleplet.utils.slepian_arbitrary_methods import compute_mesh_shannon
from sleplet.utils.validation import Validation

_data_path = Path(__file__).resolve().parents[2] / "data"


@dataclass(config=Validation)
class MeshSlepian:
    mesh: Mesh

    def __post_init_post_parse__(self) -> None:
        self.N = compute_mesh_shannon(self.mesh)
        self._compute_slepian_functions()

    def _compute_slepian_functions(self) -> None:
        """
        computes the Slepian functions of the mesh
        """
        logger.info("computing slepian functions of mesh")

        # create filenames
        eigd_loc = (
            f"meshes_laplacians_slepian_functions_{self.mesh.name}_"
            f"b{self.mesh.mesh_eigenvalues.shape[0]}_N{self.N}"
        )
        eval_loc = f"{eigd_loc}_eigenvalues.npy"
        evec_loc = f"{eigd_loc}_eigenvectors.npy"

        try:
            self.slepian_eigenvalues = np.load(find_on_pooch_then_local(eval_loc))
            self.slepian_functions = np.load(find_on_pooch_then_local(evec_loc))
        except TypeError:
            D = self._create_D_matrix()
            logger.info(
                f"Shannon number from vertices: {self.N}, "
                f"Trace of D matrix: {round(D.trace())}, "
                f"difference: {round(np.abs(self.N - D.trace()))}"
            )

            # fill in remaining triangle section
            fill_upper_triangle_of_hermitian_matrix(D)

            # solve eigenproblem
            (
                self.slepian_eigenvalues,
                self.slepian_functions,
            ) = self._clean_evals_and_evecs(LA.eigh(D))
            np.save(_data_path / eval_loc, self.slepian_eigenvalues)
            np.save(_data_path / evec_loc, self.slepian_functions[: self.N])

    def _create_D_matrix(self) -> npt.NDArray[np.float_]:  # noqa: N802
        """
        computes the D matrix for the mesh eigenfunctions
        """
        D = np.zeros(
            (self.mesh.mesh_eigenvalues.shape[0], self.mesh.mesh_eigenvalues.shape[0])
        )

        D_ext, shm_ext = create_shared_memory_array(D)

        def func(chunk: list[int]) -> None:
            """
            calculate D matrix components for each chunk
            """
            D_int, shm_int = attach_to_shared_memory_block(D, shm_ext)

            for i in chunk:
                logger.info(f"start basis function: {i}")
                self._fill_D_elements(D_int, i)
                logger.info(f"finish basis function: {i}")

            free_shared_memory(shm_int)

        # split up L range to maximise effiency
        chunks = split_arr_into_chunks(
            self.mesh.mesh_eigenvalues.shape[0], settings["NCPU"]
        )

        # initialise pool and apply function
        with ThreadPoolExecutor(max_workers=settings["NCPU"]) as e:
            e.map(func, chunks)

        # retrieve from parallel function
        D = D_ext.copy()

        # Free and release the shared memory block at the very end
        free_shared_memory(shm_ext)
        release_shared_memory(shm_ext)
        return D

    def _fill_D_elements(self, D: npt.NDArray[np.float_], i: int) -> None:  # noqa: N802
        """
        fill in the D matrix elements using symmetries
        """
        D[i][i] = self._integral(i, i)
        for j in range(i + 1, self.mesh.mesh_eigenvalues.shape[0]):
            D[j][i] = self._integral(j, i)

    def _integral(self, i: int, j: int) -> float:
        """
        calculates the D integral between two mesh basis functions
        """
        return integrate_region_mesh(
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
        """
        need eigenvalues and eigenvectors to be in a certain format
        """
        # access values
        eigenvalues, eigenvectors = eigendecomposition

        # sort eigenvalues and eigenvectors in descending order of eigenvalues
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx].T
        return eigenvalues, eigenvectors
