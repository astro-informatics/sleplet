from pathlib import Path
from typing import List, Optional

import numpy as np
from dynaconf import settings

from pys2sleplet.slepian.slepian_functions import SlepianFunctions
from pys2sleplet.slepian.slepian_region.slepian_arbitrary import SlepianArbitrary
from pys2sleplet.slepian.slepian_region.specific_region.slepian_limit_lat_long import (
    SlepianLimitLatLong,
)
from pys2sleplet.slepian.slepian_region.specific_region.slepian_polar_cap import (
    SlepianPolarCap,
)
from pys2sleplet.utils.bool_methods import is_limited_lat_lon, is_polar_cap
from pys2sleplet.utils.string_methods import verify_args

from ..functions import Functions


class Slepian(Functions):
    def __init__(self, L: int, args: List[int] = None):
        self.phi_min = np.deg2rad(settings.PHI_MIN)
        self.phi_max = np.deg2rad(settings.PHI_MAX)
        self.theta_min = np.deg2rad(settings.THETA_MIN)
        self.theta_max = np.deg2rad(settings.THETA_MAX)
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
            slepian = SlepianPolarCap(self.L, self.theta_max)
        elif is_limited_lat_lon:
            slepian = SlepianLimitLatLong(
                self.L, self.theta_min, self.theta_max, self.phi_min, self.phi_max
            )
        else:
            location = (
                Path(__file__).resolve().parents[2]
                / "data"
                / "slepian"
                / "arbitrary"
                / "masks"
                / settings.MASK_FILE
            )
            try:
                mask = np.load(location)
                slepian = SlepianArbitrary(self.L, mask)
            except FileNotFoundError:
                print("specify valid mask file")
        return slepian

    @property
    def rank(self) -> int:
        return self.__rank

    @rank.setter
    def rank(self, var: int) -> None:
        self.__rank = var

    @property
    def order(self) -> int:
        return self.__order

    @order.setter
    def order(self, var: int) -> None:
        self.__order = var
