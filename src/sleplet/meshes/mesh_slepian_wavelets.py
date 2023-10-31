"""Contains the `MeshSlepianWavelets` class."""
import logging

import numpy as np
import numpy.typing as npt
import pydantic
import typing_extensions

import pys2let

import sleplet._string_methods
import sleplet._validation
import sleplet.wavelet_methods
from sleplet.meshes.mesh_slepian_coefficients import MeshSlepianCoefficients

_logger = logging.getLogger(__name__)


@pydantic.dataclasses.dataclass(config=sleplet._validation.validation, kw_only=True)
class MeshSlepianWavelets(MeshSlepianCoefficients):
    """Create Slepian wavelets of a given mesh."""

    B: int = 3
    r"""The wavelet parameter. Represented as \(\lambda\) in the papers."""
    j_min: int = 2
    r"""The minimum wavelet scale. Represented as \(J_{0}\) in the papers."""
    j: int | None = None
    """Option to select a given wavelet. `None` indicates the scaling function,
    whereas `0` would correspond to the selected `j_min`."""

    def __post_init__(self: typing_extensions.Self) -> None:
        super().__post_init__()

    def _create_coefficients(
        self: typing_extensions.Self,
    ) -> npt.NDArray[np.complex_ | np.float_]:
        _logger.info("start computing wavelets")
        self.wavelets = self._create_wavelets()
        _logger.info("finish computing wavelets")
        jth = 0 if self.j is None else self.j + 1
        return self.wavelets[jth]

    def _create_name(self: typing_extensions.Self) -> str:
        return (
            f"slepian_wavelets_{self.mesh.name}"
            f"{sleplet._string_methods.filename_args(self.B, 'B')}"
            f"{sleplet._string_methods.filename_args(self.j_min, 'jmin')}"
            f"{sleplet._string_methods.wavelet_ending(self.j_min, self.j)}"
        )

    def _setup_args(self: typing_extensions.Self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 3
            if len(self.extra_args) != num_args:
                msg = f"The number of extra arguments should be {num_args}"
                raise ValueError(msg)
            self.B, self.j_min, self.j = self.extra_args

    def _create_wavelets(self: typing_extensions.Self) -> npt.NDArray[np.float_]:
        """Create Slepian wavelets of the mesh."""
        return sleplet.wavelet_methods.create_kappas(
            self.mesh.mesh_eigenvalues.shape[0],
            self.B,
            self.j_min,
        )

    @pydantic.field_validator("j")
    def _check_j(
        cls,  # noqa: ANN101
        v: int | None,
        info: pydantic.ValidationInfo,
    ) -> int | None:
        j_max = pys2let.pys2let_j_max(
            info.data["B"],
            info.data["mesh"].mesh_eigenvalues.shape[0],
            info.data["j_min"],
        )
        if v is not None and v < 0:
            msg = "j should be positive"
            raise ValueError(msg)
        if v is not None and v > j_max - info.data["j_min"]:
            msg = (
                "j should be less than j_max - j_min: "
                f"{j_max - info.data['j_min'] + 1}"
            )
            raise ValueError(msg)
        return v
