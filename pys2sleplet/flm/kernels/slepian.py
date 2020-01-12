from pathlib import Path
from typing import List, Optional

import numpy as np

from pys2sleplet.slepian.slepian_arbitrary import SlepianArbitrary
from pys2sleplet.slepian.slepian_functions import SlepianFunctions
from pys2sleplet.slepian.slepian_region.specific_region.slepian_limit_lat_long import (
    SlepianLimitLatLong,
)
from pys2sleplet.slepian.slepian_region.specific_region.slepian_polar_cap import (
    SlepianPolarCap,
)
from pys2sleplet.utils.bool_methods import is_limited_lat_lon, is_polar_cap
from pys2sleplet.utils.string_methods import verify_args
from pys2sleplet.utils.vars import SLEPIAN

from ..functions import Functions


class Slepian(Functions):
    # [0, 360, 0, 180, 0, 0, 0]
    def __init__(self, L: int, args: List[int] = None):
        self.s = self._create_slepian()
        super().__init__(L, args)

    def _setup_args(self, args: Optional[List[int]]) -> None:
        if args is not None:
            verify_args(args, 1)
            rank = args[0]
        else:
            rank = 0
        self.rank = rank

    def _create_name(self) -> str:
        name = "slepian"
        return name

    def _create_flm(self, L: int) -> np.ndarray:
        flm = self.s.eigenvectors[self.rank]
        print(f"Eigenvalue {self.rank}: {self.s.eigenvalues[self.rank]:e}")
        return flm

    def _create_slepian(self) -> SlepianFunctions:
        """
        initialise Slepian object depending on input
        """
        if is_polar_cap:
            slepian = SlepianPolarCap(self.L, np.deg2rad(SLEPIAN["THETA_MAX"]))
        elif is_limited_lat_lon:
            slepian = SlepianLimitLatLong(
                self.L,
                np.deg2rad(SLEPIAN["THETA_MIN"]),
                np.deg2rad(SLEPIAN["THETA_MAX"]),
                np.deg2rad(SLEPIAN["PHI_MIN"]),
                np.deg2rad(SLEPIAN["PHI_MAX"]),
            )
        else:
            location = (
                Path(__file__).resolve().parents[2]
                / "data"
                / "slepian"
                / "arbitrary"
                / "masks"
                / SLEPIAN["MASK"]
            )
            mask = np.load(location)
            slepian = SlepianArbitrary(self.L, mask)
        return slepian

    @property
    def rank(self) -> int:
        return self.__rank

    @rank.setter
    def rank(self, var: int) -> None:
        self.__rank = var
