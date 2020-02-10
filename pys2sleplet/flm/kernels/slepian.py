from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from pys2sleplet.slepian.slepian_region.slepian_arbitrary import SlepianArbitrary
from pys2sleplet.slepian.slepian_region.specific_region.slepian_limit_lat_long import (
    SlepianLimitLatLong,
)
from pys2sleplet.slepian.slepian_region.specific_region.slepian_polar_cap import (
    SlepianPolarCap,
)
from pys2sleplet.utils.bool_methods import is_limited_lat_lon, is_polar_cap
from pys2sleplet.utils.inputs import config
from pys2sleplet.utils.logging import logger

from ..functions import Functions


class Slepian(Functions):
    def __init__(self, L: int, args: List[int] = None):
        self.L = L
        self.reality = False
        self.phi_min = np.deg2rad(config.PHI_MIN)
        self.phi_max = np.deg2rad(config.PHI_MAX)
        self.theta_min = np.deg2rad(config.THETA_MIN)
        self.theta_max = np.deg2rad(config.THETA_MAX)
        self.s = self._create_slepian()
        super().__init__(L, args)

    def _setup_args(self, args: Optional[List[int]]) -> None:
        if args is not None:
            num_args = 1
            if len(args) != num_args:
                raise ValueError(
                    f"The number of extra arguments should be 1 or {num_args}"
                )
            rank = args[0]
        else:
            rank = 0
        self.rank = rank

    def _create_name(self) -> str:
        name = f"{self.s.name}_rank{self.rank}"
        return name

    def _create_flm(self, L: int) -> np.ndarray:
        flm = self.s.eigenvectors[self.rank]
        logger.info(f"Eigenvalue {self.rank}: {self.s.eigenvalues[self.rank]:e}")
        return flm

    def _create_annotations(self) -> List[Dict]:
        annotations = self.s.annotations
        return annotations

    def _create_slepian(
        self
    ) -> Union[SlepianPolarCap, SlepianLimitLatLong, SlepianArbitrary]:
        """
        initialise Slepian object depending on input
        """
        if is_polar_cap(self.phi_min, self.phi_max, self.theta_min, self.theta_max):
            logger.info("polar cap region detected")
            return SlepianPolarCap(self.L, self.theta_max, config.ORDER)

        elif is_limited_lat_lon(
            self.phi_min, self.phi_max, self.theta_min, self.theta_max
        ):
            logger.info("limited latitude longitude region detected")
            return SlepianLimitLatLong(
                self.L, self.theta_min, self.theta_max, self.phi_min, self.phi_max
            )

        elif config.SLEPIAN_MASK:
            logger.info("no angles specified, looking for a file with mask")
            location = (
                Path(__file__).resolve().parents[2]
                / "data"
                / "slepian"
                / "arbitrary"
                / "masks"
                / config.SLEPIAN_MASK
            )
            try:
                mask = np.load(location)
                return SlepianArbitrary(self.L, mask)
            except FileNotFoundError:
                logger.error(f"can not find the file: {config.SLEPIAN_MASK}")
                raise

        else:
            raise RuntimeError("no angle or file specified for Slepian region")

    @property
    def rank(self) -> int:
        return self.__rank

    @rank.setter
    def rank(self, var: int) -> None:
        if not isinstance(var, int):
            raise TypeError("rank should be an integer")
        if var < 0:
            raise ValueError("rank cannot be negative")
        if var >= self.L:
            raise ValueError(f"rank should be no more than {self.L}")
        self.__rank = var
