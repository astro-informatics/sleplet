from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pyssht as ssht

from pys2sleplet.flm.functions import Functions
from pys2sleplet.utils.config import settings
from pys2sleplet.utils.pys2let import s2let
from pys2sleplet.utils.region import Region
from pys2sleplet.utils.slepian_methods import choose_slepian_method, slepian_inverse
from pys2sleplet.utils.string_methods import filename_args


@dataclass
class SlepianWavelets(Functions):
    B: int
    j_min: int
    j: Optional[int]
    region: Optional[Region]
    _B: int = field(default=2, init=False, repr=False)
    _j: Optional[int] = field(default=None, init=False, repr=False)
    _j_min: int = field(default=0, init=False, repr=False)
    _region: Optional[Region] = field(default=None, init=False, repr=False)
    _j_max: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.region = (
            Region(
                gap=settings.POLAR_GAP,
                mask_name=settings.SLEPIAN_MASK,
                phi_max=np.deg2rad(settings.PHI_MAX),
                phi_min=np.deg2rad(settings.PHI_MIN),
                theta_max=np.deg2rad(settings.THETA_MAX),
                theta_min=np.deg2rad(settings.THETA_MIN),
            )
            if self.region is None
            else self.region
        )
        self.slepian = choose_slepian_method(self.L, self.region)
        super().__post_init__()

    def _create_annotations(self) -> None:
        self.annotations = self.slepian.annotations

    def _create_flm(self) -> None:
        kappa0, kappa = s2let.axisym_wav_l(self.B, self.L ** 2, self.j_min)
        k = kappa0 if self.j is None else kappa[:, self.j]
        f = slepian_inverse(
            self.L, k, self.slepian.eigenvectors, coefficients=self.slepian.N
        )
        self.multipole = ssht.forward(f, self.L)

    def _create_name(self) -> None:
        coefficient = (
            "_scaling"
            if self.j is None
            else f"{filename_args(self.j + self.j_min, 'j')}"
        )
        self.name = (
            f"slepian_wavelets_{self.slepian.region.name_ending}"
            f"{filename_args(self.B, 'B')}"
            f"{filename_args(self.j_min, 'jmin')}"
            f"{coefficient}"
        )

    def _set_reality(self) -> None:
        self.reality = True

    def _set_spin(self) -> None:
        self.spin = 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 3
            if len(self.extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.B, self.j_min, self.j = self.extra_args[:num_args]

    @property  # type:ignore
    def B(self) -> int:
        return self._B

    @B.setter
    def B(self, B: int) -> None:
        if isinstance(B, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            B = SlepianWavelets._B
        self._B = B

    @property  # type:ignore
    def j(self) -> Optional[int]:
        return self._j

    @j.setter
    def j(self, j: Optional[int]) -> None:
        if isinstance(j, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            j = SlepianWavelets._j
        self.j_max = s2let.pys2let_j_max(self.B, self.L ** 2, self.j_min)
        if j is not None and j < 0:
            raise ValueError("j should be positive")
        if j is not None and j > self.j_max - self.j_min:
            raise ValueError(
                f"j should be less than j_max - j_min: {self.j_max - self.j_min + 1}"
            )
        self._j = j

    @property  # type:ignore
    def j_max(self) -> int:
        return self._j_max

    @j_max.setter
    def j_max(self, j_max: int) -> None:
        self._j_max = j_max

    @property  # type:ignore
    def j_min(self) -> int:
        return self._j_min

    @j_min.setter
    def j_min(self, j_min: int) -> None:
        if isinstance(j_min, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            j_min = SlepianWavelets._j_min
        self._j_min = j_min

    @property  # type:ignore
    def region(self) -> Optional[Region]:
        return self._region

    @region.setter
    def region(self, region: Optional[Region]) -> None:
        if isinstance(region, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            region = SlepianWavelets._region
        self._region = region
