from dataclasses import dataclass, field

import numpy as np

from pys2sleplet.meshes.mesh import Mesh
from pys2sleplet.meshes.slepian_mesh import SlepianMesh
from pys2sleplet.utils.config import settings
from pys2sleplet.utils.mesh_methods import mesh_inverse


@dataclass()
class MeshPlot:
    name: str
    index: int
    slepian: bool
    _index: int = field(init=False, repr=False)
    _eigenvalue: float = field(init=False, repr=False)
    _eigenvector: np.ndarray = field(init=False, repr=False)
    _name: str = field(init=False, repr=False)
    _region: np.ndarray = field(init=False, repr=False)
    _slepian: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        mesh = Mesh(self.name, laplacian_type=settings.LAPLACIAN)
        self.faces = mesh.faces
        self.region = mesh.region
        self.vertices = mesh.vertices
        if self.slepian:
            slepian_mesh = SlepianMesh(mesh)
            s_p_i = slepian_mesh.slepian_functions[self.index]
            self.eigenvalue = slepian_mesh.slepian_eigenvalues[self.index]
            self.eigenvector = mesh_inverse(mesh.basis_functions, s_p_i)
        else:
            self.eigenvalue = mesh.mesh_eigenvalues[self.index]
            self.eigenvector = mesh.basis_functions[self.index]

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

    @property  # type: ignore
    def slepian(self) -> bool:
        return self._slepian

    @slepian.setter
    def slepian(self, slepian: bool) -> None:
        if isinstance(slepian, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            slepian = MeshPlot._slepian
        self._slepian = slepian