from dataclasses import dataclass, field

import numpy as np

from pys2sleplet.flm.functions import Functions
from pys2sleplet.slepian.slepian_functions import SlepianFunctions
from pys2sleplet.slepian.slepian_region.slepian_arbitrary import SlepianArbitrary
from pys2sleplet.slepian.slepian_region.slepian_limit_lat_lon import SlepianLimitLatLon
from pys2sleplet.slepian.slepian_region.slepian_polar_cap import SlepianPolarCap
from pys2sleplet.utils.config import config
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.region import Region


@dataclass
class Slepian(Functions):
    _rank: int = field(default=0, init=False, repr=False)
    _slepian: SlepianFunctions = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.slepian = self._choose_slepian_method()
        super().__post_init__()

    def _create_annotations(self) -> None:
        self.annotations = self.slepian.annotations

    def _create_name(self) -> None:
        self.name = f"{self.slepian.name}_rank{self.rank}"

    def _create_flm(self) -> None:
        self.multipole = self.slepian.eigenvectors[self.rank]
        logger.info(f"Eigenvalue {self.rank}: {self.slepian.eigenvalues[self.rank]:e}")

    def _set_reality(self) -> None:
        self.reality = False

    def _setup_args(self) -> None:
        if self.extra_args is not None:
            num_args = 1
            if len(self.extra_args) != num_args:
                raise ValueError(
                    f"The number of extra arguments should be 1 or {num_args}"
                )
            self.rank = self.extra_args[0]

    @staticmethod
    def _choose_slepian_method() -> SlepianFunctions:
        """
        initialise Slepian object depending on input
        """
        region = Region(
            phi_min=np.deg2rad(config.PHI_MIN),
            phi_max=np.deg2rad(config.PHI_MAX),
            theta_min=np.deg2rad(config.THETA_MIN),
            theta_max=np.deg2rad(config.THETA_MAX),
            mask_name=config.SLEPIAN_MASK,
        )

        if region.region_type == "polar":
            logger.info("polar cap region detected")
            slepian = SlepianPolarCap(config.L, region.theta_max, order=config.ORDER)

        elif region.region_type == "lim_lat_lon":
            logger.info("limited latitude longitude region detected")
            slepian = SlepianLimitLatLon(
                config.L,
                region.theta_min,
                region.theta_max,
                region.phi_min,
                region.phi_max,
            )

        elif region.region_type == "arbitrary":
            logger.info("mask specified in file detected")
            slepian = SlepianArbitrary(config.L, region.mask_name)

        return slepian

    @property
    def rank(self) -> int:
        return self._rank

    @rank.setter
    def rank(self, rank: int) -> None:
        if not isinstance(rank, int):
            raise TypeError("rank should be an integer")
        if rank < 0:
            raise ValueError("rank cannot be negative")
        if rank >= self.L:
            raise ValueError(f"rank should be no more than {self.L}")
        self._rank = rank
        logger.info(f"rank={self.rank}")

    @property
    def slepian(self) -> SlepianFunctions:
        return self._slepian

    @slepian.setter
    def slepian(self, slepian: SlepianFunctions) -> None:
        self._slepian = slepian
