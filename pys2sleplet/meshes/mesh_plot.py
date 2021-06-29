from dataclasses import dataclass, field

import numpy as np

from pys2sleplet.meshes.mesh import Mesh
from pys2sleplet.meshes.slepian_mesh import SlepianMesh
from pys2sleplet.meshes.slepian_wavelets_mesh import SlepianWaveletsMesh
from pys2sleplet.utils.config import settings
from pys2sleplet.utils.mesh_methods import mesh_inverse
from pys2sleplet.utils.slepian_mesh_methods import slepian_mesh_inverse


@dataclass()
class MeshPlot:
    name: str
    index: int
    method: str
    B: int
    j_min: int
    _B: int = field(init=False, repr=False)
    _eigenvalue: float = field(init=False, repr=False)
    _eigenvector: np.ndarray = field(init=False, repr=False)
    _index: int = field(init=False, repr=False)
    _j_min: int = field(init=False, repr=False)
    _method: str = field(init=False, repr=False)
    _name: str = field(init=False, repr=False)
    _region: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        mesh = Mesh(self.name, laplacian_type=settings.LAPLACIAN)
        self.faces = mesh.faces
        self.region = mesh.region
        self.vertices = mesh.vertices
        if self.method == "basis":
            self.eigenvalue = mesh.mesh_eigenvalues[self.index]
            self.eigenvector = mesh.basis_functions[self.index]
        else:
            slepian_mesh = SlepianMesh(mesh)
            self.eigenvalue = slepian_mesh.slepian_eigenvalues[self.index]
            if self.method == "slepian":
                s_p_i = slepian_mesh.slepian_functions[self.index]
                self.eigenvector = mesh_inverse(mesh.basis_functions, s_p_i)
            else:
                slepian_wavelets_mesh = SlepianWaveletsMesh(
                    slepian_mesh, B=self.B, j_min=self.j_min
                )
                self.eigenvector = slepian_mesh_inverse(
                    mesh,
                    slepian_wavelets_mesh.wavelets[self.index],
                    slepian_mesh.slepian_functions,
                    slepian_mesh.N,
                )

    @property  # type: ignore
    def B(self) -> int:
        return self._B

    @B.setter
    def B(self, B: int) -> None:
        self._B = B

    @property
    def eigenvalue(self) -> float:
        return self._eigenvalue

    @eigenvalue.setter
    def eigenvalue(self, eigenvalue: float) -> None:
        self._eigenvalue = eigenvalue

    @property
    def eigenvector(self) -> np.ndarray:
        return self._eigenvector

    @eigenvector.setter
    def eigenvector(self, eigenvector: np.ndarray) -> None:
        self._eigenvector = eigenvector

    @property  # type:ignore
    def index(self) -> int:
        return self._index

    @index.setter
    def index(self, index: int) -> None:
        self._index = index

    @property  # type: ignore
    def j_min(self) -> int:
        return self._j_min

    @j_min.setter
    def j_min(self, j_min: int) -> None:
        self._j_min = j_min

    @property  # type: ignore
    def method(self) -> str:
        return self._method

    @method.setter
    def method(self, method: str) -> None:
        self._method = method

    @property  # type: ignore
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @property
    def region(self) -> np.ndarray:
        return self._region

    @region.setter
    def region(self, region: np.ndarray) -> None:
        self._region = region
