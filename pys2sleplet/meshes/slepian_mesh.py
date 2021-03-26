from dataclasses import dataclass, field

import numpy as np
from multiprocess import Pool

from pys2sleplet.meshes.mesh import Mesh
from pys2sleplet.utils.config import settings
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.mesh_methods import integrate_region_mesh
from pys2sleplet.utils.parallel_methods import (
    attach_to_shared_memory_block,
    create_shared_memory_array,
    free_shared_memory,
    release_shared_memory,
    split_arr_into_chunks,
)


@dataclass  # type: ignore
class SlepianMesh:
    name: str
    _D: np.ndarray = field(init=False, repr=False)
    _mesh: Mesh = field(init=False, repr=False)
    _name: str = field(init=False, repr=False)
    _slepian_eigenvalues: np.ndarray = field(init=False, repr=False)
    _slepian_functions: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.mesh = Mesh(self.name)
        self._compute_slepian_functions()
        self._compute_shannon()

    def _compute_shannon(self) -> None:
        """
        computes the Shannon number for the region
        """
        self.N = round(self.D.trace().sum())

    def _compute_slepian_functions(self) -> None:
        """
        computes the Slepian functions of the mesh
        """
        pass

    def _create_D_matrix(self) -> None:
        """
        computes the D matrix for the mesh eigenfunctions
        """
        # initialise matrix
        D = np.zeros((self.mesh.num_basis_fun, self.mesh.num_basis_fun))

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
        chunks = split_arr_into_chunks(self.mesh.num_basis_fun, settings.NCPU)

        # initialise pool and apply function
        with Pool(processes=settings.NCPU) as p:
            p.map(func, chunks)

        # Free and release the shared memory block at the very end
        free_shared_memory(shm_ext)
        release_shared_memory(shm_ext)
        self.D = D_ext

    def _fill_D_elements(self, D: np.ndarray, i: int) -> None:
        """
        fill in the D matrix elements using symmetries
        """
        D[i][i] = self._integral(i, i)
        for j in range(i + 1, D.shape[0]):
            D[j][i] = self._integral(j, i)

    def _integral(self, i: int, j: int) -> float:
        """
        calculates the D integral between two mesh basis functions
        """
        return integrate_region_mesh(
            self.mesh.vertices,
            self.mesh.faces,
            self.mesh.basis_functions[i] * self.mesh.basis_functions[j].conj(),
            self.mesh.region,
        )

    @property
    def D(self) -> np.ndarray:
        return self._D

    @D.setter
    def D(self, D: np.ndarray) -> None:
        self._D = D

    @property
    def mesh(self) -> Mesh:
        return self._mesh

    @mesh.setter
    def mesh(self, mesh: Mesh) -> None:
        self._mesh = mesh

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
