from dataclasses import dataclass, field

import numpy as np

from pys2sleplet.meshes.mesh import Mesh
from pys2sleplet.meshes.slepian_mesh import SlepianMesh


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
        if self.slepian:
            slepian_mesh = SlepianMesh(self.name)
            self.eigenvalue = slepian_mesh.slepian_eigenvalues[self.index]
            self.eigenvector = slepian_mesh.slepian_functions[self.index]
            self.faces = slepian_mesh.mesh.faces
            self.region = slepian_mesh.mesh.region
            self.vertices = slepian_mesh.mesh.vertices
        else:
            mesh = Mesh(self.name, num_basis_fun=self.index + 1)
            self.eigenvalue = mesh.mesh_eigenvalues[self.index]
            self.eigenvector = mesh.basis_functions[self.index]
            self.faces = mesh.faces
            self.region = mesh.region
            self.vertices = mesh.vertices

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
