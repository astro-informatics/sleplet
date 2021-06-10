from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from multiprocess import Pool
from numpy import linalg as LA

from pys2sleplet.meshes.mesh import Mesh
from pys2sleplet.utils.array_methods import fill_upper_triangle_of_hermitian_matrix
from pys2sleplet.utils.config import settings
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.mesh_methods import clean_evals_and_evecs, integrate_region_mesh
from pys2sleplet.utils.parallel_methods import (
    attach_to_shared_memory_block,
    create_shared_memory_array,
    free_shared_memory,
    release_shared_memory,
    split_arr_into_chunks,
)

_file_location = Path(__file__).resolve()
_slepian_path = _file_location.parents[1] / "data" / "meshes" / "slepian_functions"


@dataclass  # type: ignore
class SlepianMesh:
    name: str
    _mesh: Mesh = field(init=False, repr=False)
    _N: int = field(init=False, repr=False)
    _name: str = field(init=False, repr=False)
    _slepian_eigenvalues: np.ndarray = field(init=False, repr=False)
    _slepian_functions: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.mesh = Mesh(self.name)
        self._compute_slepian_functions()

    def _compute_shannon(self, D: np.ndarray) -> int:
        """
        computes the Shannon number for the region
        """
        N = D.trace()
        logger.info(f"trace of D matrix: {N:e}")
        return np.floor(N).astype(int)

    def _compute_slepian_functions(self) -> None:
        """
        computes the Slepian functions of the mesh
        """
        logger.info("computing slepian functions of mesh")
        eval_loc = _slepian_path / self.name / "eigenvalues.npy"
        evec_loc = _slepian_path / self.name / "eigenvectors.npy"
        if eval_loc.exists() and evec_loc.exists():
            logger.info("binaries found - loading...")
            self.slepian_eigenvalues = np.load(eval_loc)
            self.slepian_functions = np.load(evec_loc)
        else:
            D = self._create_D_matrix()
            self.N = self._compute_shannon(D)

            # fill in remaining triangle section
            fill_upper_triangle_of_hermitian_matrix(D)

            # solve eigenproblem
            self.slepian_eigenvalues, self.slepian_functions = clean_evals_and_evecs(
                LA.eigh(D)
            )
            if settings.SAVE_MATRICES:
                np.save(eval_loc, self.slepian_eigenvalues[: self.N])
                np.save(evec_loc, self.slepian_functions[: self.N])

    def _create_D_matrix(self) -> np.ndarray:
        """
        computes the D matrix for the mesh eigenfunctions
        """
        D = np.zeros((self.mesh.vertices.shape[0], self.mesh.vertices.shape[0]))

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
        chunks = split_arr_into_chunks(self.mesh.vertices.shape[0], settings.NCPU)

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
        for j in range(i + 1, self.mesh.vertices.shape[0]):
            D[j][i] = self._integral(j, i)

    def _integral(self, i: int, j: int) -> float:
        """
        calculates the D integral between two mesh basis functions
        """
        return integrate_region_mesh(
            self.mesh.vertices,
            self.mesh.faces,
            self.mesh.basis_functions[i] * self.mesh.basis_functions[j],
            self.mesh.region,
        )

    @property
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

    @property  # type: ignore
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

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
