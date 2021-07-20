from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from multiprocess import Pool
from numpy import linalg as LA

from pys2sleplet.meshes.classes.mesh import Mesh
from pys2sleplet.utils.array_methods import fill_upper_triangle_of_hermitian_matrix
from pys2sleplet.utils.config import settings
from pys2sleplet.utils.integration_methods import integrate_region_mesh
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.parallel_methods import (
    attach_to_shared_memory_block,
    create_shared_memory_array,
    free_shared_memory,
    release_shared_memory,
    split_arr_into_chunks,
)
from pys2sleplet.utils.slepian_arbitrary_methods import compute_mesh_shannon

_file_location = Path(__file__).resolve()
_meshes_path = _file_location.parents[2] / "data" / "meshes"


@dataclass
class SlepianMesh:
    mesh: Mesh
    _mesh: Mesh = field(init=False, repr=False)
    _N: int = field(init=False, repr=False)
    _slepian_eigenvalues: np.ndarray = field(init=False, repr=False)
    _slepian_functions: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.N = compute_mesh_shannon(self.mesh)
        self._compute_slepian_functions()

    def _compute_slepian_functions(self) -> None:
        """
        computes the Slepian functions of the mesh
        """
        logger.info("computing slepian functions of mesh")

        # create filenames
        laplacian_type = "mesh" if settings.MESH_LAPLACIAN else "graph"
        eigd_loc = (
            _meshes_path
            / "laplacians"
            / laplacian_type
            / "slepian_functions"
            / f"{self.mesh.name}_b{self.mesh.basis_functions.shape[0]}_N{self.N}"
        )
        eval_loc = eigd_loc / "eigenvalues.npy"
        evec_loc = eigd_loc / "eigenvectors.npy"

        if eval_loc.exists() and evec_loc.exists():
            logger.info("binaries found - loading...")
            self.slepian_eigenvalues = np.load(eval_loc)
            self.slepian_functions = np.load(evec_loc)
        else:
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
            if settings.SAVE_MATRICES:
                np.save(eval_loc, self.slepian_eigenvalues[: self.N])
                np.save(evec_loc, self.slepian_functions[: self.N])

    def _create_D_matrix(self) -> np.ndarray:
        """
        computes the D matrix for the mesh eigenfunctions
        """
        D = np.zeros(
            (self.mesh.basis_functions.shape[0], self.mesh.basis_functions.shape[0])
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
            self.mesh.basis_functions.shape[0], settings.NCPU
        )

        # initialise pool and apply function
        with Pool(processes=settings.NCPU) as p:
            p.map(func, chunks)

        # retrieve from parallel function
        D = D_ext.copy()

        # Free and release the shared memory block at the very end
        free_shared_memory(shm_ext)
        release_shared_memory(shm_ext)
        return D

    def _fill_D_elements(self, D: np.ndarray, i: int) -> None:
        """
        fill in the D matrix elements using symmetries
        """
        D[i][i] = self._integral(i, i)
        for j in range(i + 1, self.mesh.basis_functions.shape[0]):
            D[j][i] = self._integral(j, i)

    def _integral(self, i: int, j: int) -> float:
        """
        calculates the D integral between two mesh basis functions
        """
        return integrate_region_mesh(
            self.mesh.region,
            self.mesh.basis_functions[i],
            self.mesh.basis_functions[j],
        )

    @staticmethod
    def _clean_evals_and_evecs(
        eigendecomposition: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
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

    @property  # type:ignore
    def mesh(self) -> Mesh:
        return self._mesh

    @mesh.setter
    def mesh(self, mesh: Mesh) -> None:
        self._mesh = mesh

    @property
    def N(self) -> int:
        return self._N

    @N.setter
    def N(self, N: int) -> None:
        self._N = N

    @property
    def slepian_eigenvalues(self) -> np.ndarray:
        return self._slepian_eigenvalues

    @slepian_eigenvalues.setter
    def slepian_eigenvalues(self, slepian_eigenvalues: np.ndarray) -> None:
        self._slepian_eigenvalues = slepian_eigenvalues

    @property
    def slepian_functions(self) -> np.ndarray:
        return self._slepian_functions

    @slepian_functions.setter
    def slepian_functions(self, slepian_functions: np.ndarray) -> None:
        self._slepian_functions = slepian_functions