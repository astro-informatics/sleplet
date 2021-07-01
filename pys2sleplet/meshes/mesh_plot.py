from dataclasses import dataclass, field

import numpy as np

from pys2sleplet.meshes.mesh import Mesh
from pys2sleplet.meshes.mesh_field import MeshField
from pys2sleplet.meshes.slepian_mesh import SlepianMesh
from pys2sleplet.meshes.slepian_wavelets_mesh import SlepianWaveletsMesh
from pys2sleplet.utils.config import settings
from pys2sleplet.utils.mesh_methods import mesh_inverse
from pys2sleplet.utils.slepian_mesh_methods import slepian_mesh_inverse
from pys2sleplet.utils.string_methods import wavelet_ending


@dataclass()
class MeshPlot:
    name: str
    index: int
    method: str
    B: int
    j_min: int
    _B: int = field(init=False, repr=False)
    _eigenvector: np.ndarray = field(init=False, repr=False)
    _index: int = field(init=False, repr=False)
    _j_min: int = field(init=False, repr=False)
    _mesh: Mesh = field(init=False, repr=False)
    _method: str = field(init=False, repr=False)
    _name: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._create_plot()

    def _create_plot(self) -> None:
        """
        master plotting method which initialises the eigenvalue and
        eigenvector depending on what the value of method is
        """
        # initialise mesh object
        self.mesh = Mesh(self.name, laplacian_type=settings.LAPLACIAN)

        if self.method == "region":
            self._plot_region()
        elif self.method == "basis":
            self._plot_basis_functions()
        elif self.method == "field":
            self._plot_field_on_mesh()
        else:
            # initialise Slepian mesh object
            slepian_mesh = SlepianMesh(self.mesh)

            if self.method == "slepian":
                self._plot_slepian_functions(slepian_mesh)
            else:
                self._plot_slepian_wavelets(slepian_mesh)

    def _plot_region(self) -> None:
        """
        method to just plot the region of interest
        """
        self.name = f"{self.name}_region"
        self.eigenvector = np.ones(self.mesh.vertices.shape[0])

    def _plot_basis_functions(self) -> None:
        """
        method to plot the basis functions of the mesh directly
        """
        self.name = (
            f"{self.name}_rank{self.index}_"
            f"lam{self.mesh.mesh_eigenvalues[self.index]:e}"
        )
        self.eigenvector = self.mesh.basis_functions[self.index]

    def _plot_field_on_mesh(self) -> None:
        """
        plots a field defined on the vertices of the mesh
        """
        self.name = f"{self.name}_field"
        mesh_field = MeshField(self.mesh)
        self.eigenvector = mesh_field.field_values

    def _plot_slepian_functions(self, slepian_mesh: SlepianMesh) -> None:
        """
        method to plot the Slepian functions of the mesh
        """
        self.name = (
            f"slepian_{self.name}_rank{self.index}_"
            f"lam{slepian_mesh.slepian_eigenvalues[self.index]:e}"
        )
        s_p_i = slepian_mesh.slepian_functions[self.index]
        self.eigenvector = mesh_inverse(slepian_mesh.mesh.basis_functions, s_p_i)

    def _plot_slepian_wavelets(self, slepian_mesh: SlepianMesh) -> None:
        """
        method to plot the Slepian wavelets of the mesh
        """
        # create file ending for wavelets
        j = None if self.index == 0 else self.index - 1
        name_end = wavelet_ending(self.j_min, j)
        self.name = (
            f"slepian_wavelets_{self.name}_" f"{self.B}B_{self.j_min}jmin{name_end}"
        )

        # initialise Slepian wavelets mesh object
        slepian_wavelets_mesh = SlepianWaveletsMesh(
            slepian_mesh, B=self.B, j_min=self.j_min
        )
        self.eigenvector = slepian_mesh_inverse(
            slepian_wavelets_mesh.wavelets[self.index],
            slepian_mesh.mesh,
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

    @property
    def mesh(self) -> Mesh:
        return self._mesh

    @mesh.setter
    def mesh(self, mesh: Mesh) -> None:
        self._mesh = mesh

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
