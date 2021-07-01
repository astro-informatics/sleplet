from dataclasses import dataclass, field

import numpy as np

from pys2sleplet.meshes.mesh import Mesh
from pys2sleplet.meshes.mesh_field import MeshField
from pys2sleplet.meshes.slepian_mesh import SlepianMesh
from pys2sleplet.meshes.slepian_mesh_field import SlepianMeshField
from pys2sleplet.meshes.slepian_wavelets_mesh import SlepianWaveletsMesh
from pys2sleplet.utils.config import settings
from pys2sleplet.utils.slepian_mesh_methods import (
    slepian_mesh_forward,
    slepian_mesh_inverse,
)
from pys2sleplet.utils.string_methods import wavelet_ending

REAL_VALUED_FUNCTIONS: set[str] = {"basis", "field", "region"}


@dataclass()
class MeshPlot:
    name: str
    index: int
    method: str
    B: int
    j_min: int
    _B: int = field(init=False, repr=False)
    _field_values: np.ndarray = field(init=False, repr=False)
    _index: int = field(init=False, repr=False)
    _j_min: int = field(init=False, repr=False)
    _mesh: Mesh = field(init=False, repr=False)
    _method: str = field(init=False, repr=False)
    _name: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._create_plot()

    def _create_plot(self) -> None:
        """
        master plotting method depending on the method value
        """
        self.mesh = Mesh(self.name, laplacian_type=settings.LAPLACIAN)
        mesh_field = MeshField(self.mesh)

        if self.method in REAL_VALUED_FUNCTIONS:
            self._plot_real_valued_functions(mesh_field)
        else:
            slepian_mesh = SlepianMesh(self.mesh)
            self._plot_slepian_coefficients(mesh_field, slepian_mesh)

    def _plot_real_valued_functions(self, mesh_field: MeshField) -> None:
        """
        master method to plot the functions defined directly on vertices
        """
        if self.method == "basis":
            self._plot_basis_functions()
        elif self.method == "field":
            self._plot_field_on_mesh(mesh_field)
        else:
            self._plot_region()

    def _plot_slepian_coefficients(
        self, mesh_field: MeshField, slepian_mesh: SlepianMesh
    ) -> None:
        """
        master method to plot the functions defined in Slepian space
        """
        if self.method == "slepian":
            slepian_coefficients = self._plot_slepian_functions(slepian_mesh)
        elif self.method == "slepian_field":
            slepian_coefficients = self._plot_slepian_field(mesh_field, slepian_mesh)
        else:
            slepian_coefficients = self._plot_slepian_wavelets(slepian_mesh)
        # convert to Slepian coefficients to real space
        self.field_values = slepian_mesh_inverse(
            slepian_coefficients,
            slepian_mesh.mesh,
            slepian_mesh.slepian_functions,
            slepian_mesh.N,
        )

    def _plot_basis_functions(self) -> None:
        """
        method to plot the basis functions of the mesh directly
        """
        self.name = (
            (
                f"{self.name}_rank{self.index}_"
                f"lam{self.mesh.mesh_eigenvalues[self.index]:e}"
            )
            .replace(".", "-")
            .replace("+", "")
        )
        self.field_values = self.mesh.basis_functions[self.index]

    def _plot_field_on_mesh(self, mesh_field: MeshField) -> None:
        """
        plots a field defined on the vertices of the mesh
        """
        self.name = f"{self.name}_field"
        self.field_values = mesh_field.field_values

    def _plot_region(self) -> None:
        """
        method to just plot the region of interest
        """
        self.name = f"{self.name}_region"
        self.field_values = np.ones(self.mesh.vertices.shape[0])

    def _plot_slepian_functions(self, slepian_mesh: SlepianMesh) -> np.ndarray:
        """
        method to plot the Slepian functions of the mesh
        """
        self.name = (
            (
                f"slepian_{self.name}_rank{self.index}_"
                f"lam{slepian_mesh.slepian_eigenvalues[self.index]:e}"
            )
            .replace(".", "-")
            .replace("+", "")
        )
        s_p_i = slepian_mesh.slepian_functions[self.index]
        return slepian_mesh_forward(
            self.mesh,
            slepian_mesh.slepian_eigenvalues,
            slepian_mesh.slepian_functions,
            slepian_mesh.N,
            u_i=s_p_i,
        )

    def _plot_slepian_field(self, mesh_field, slepian_mesh) -> np.ndarray:
        """
        method to plot Slepian coefficients of a field
        """
        self.name = f"slepian_{self.name}_field"
        slepian_mesh_field = SlepianMeshField(mesh_field, slepian_mesh)
        return slepian_mesh_field.slepian_coefficients

    def _plot_slepian_wavelets(self, slepian_mesh: SlepianMesh) -> np.ndarray:
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
        return slepian_wavelets_mesh.wavelets[self.index]

    @property  # type: ignore
    def B(self) -> int:
        return self._B

    @B.setter
    def B(self, B: int) -> None:
        self._B = B

    @property
    def field_values(self) -> np.ndarray:
        return self._field_values

    @field_values.setter
    def field_values(self, field_values: np.ndarray) -> None:
        self._field_values = field_values

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
