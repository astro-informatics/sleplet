from dataclasses import dataclass, field
from typing import Optional

from pys2sleplet.utils.bool_methods import is_limited_lat_lon, is_polar_cap
from pys2sleplet.utils.string_methods import angle_as_degree, multiples_of_pi
from pys2sleplet.utils.vars import (
    GAP_DEFAULT,
    PHI_MAX_DEFAULT,
    PHI_MIN_DEFAULT,
    THETA_MAX_DEFAULT,
    THETA_MIN_DEFAULT,
)


@dataclass
class Region:
    theta_min: float
    theta_max: float
    phi_min: float
    phi_max: float
    mask_name: Optional[str]
    gap: bool
    _gap: bool = field(default=GAP_DEFAULT, init=False, repr=False)
    _mask_name: Optional[str] = field(default=None, init=False, repr=False)
    _name_ending: str = field(init=False, repr=False)
    _phi_max: float = field(default=PHI_MAX_DEFAULT, init=False, repr=False)
    _phi_min: float = field(default=PHI_MIN_DEFAULT, init=False, repr=False)
    _region_type: str = field(init=False, repr=False)
    _theta_max: float = field(default=THETA_MAX_DEFAULT, init=False, repr=False)
    _theta_min: float = field(default=THETA_MIN_DEFAULT, init=False, repr=False)

    def __post_init__(self) -> None:
        self._identify_region()

    def _identify_region(self) -> None:
        """
        identify region type based on the angle inputs or a mask name
        """
        if is_polar_cap(self.phi_min, self.phi_max, self.theta_min, self.theta_max):
            self.region_type = "polar"
            self.name_ending = (
                f"polar{'_gap' if self.gap else ''}"
                f"{angle_as_degree(self.theta_max)}"
            )

        elif is_limited_lat_lon(
            self.phi_min, self.phi_max, self.theta_min, self.theta_max
        ):
            self.region_type = "lim_lat_lon"
            self.name_ending = (
                f"theta{angle_as_degree(self.theta_min)}"
                f"-{angle_as_degree(self.theta_max)}"
                f"_phi{angle_as_degree(self.phi_min)}"
                f"-{angle_as_degree(self.phi_max)}"
            )

        elif self.mask_name:
            self.region_type = "arbitrary"
            self.name_ending = self.mask_name

        else:
            raise AttributeError(
                "need to specify either a polar cap, a limited latitude "
                "longitude region, or a file with a mask"
            )

    @property  # type: ignore
    def gap(self) -> bool:
        return self._gap

    @gap.setter
    def gap(self, gap: bool) -> None:
        if isinstance(gap, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            gap = Region._gap
        self._gap = gap

    @property  # type: ignore
    def mask_name(self) -> Optional[str]:
        return self._mask_name

    @mask_name.setter
    def mask_name(self, mask_name: Optional[str]) -> None:
        if isinstance(mask_name, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            mask_name = Region._mask_name
        self._mask_name = mask_name

    @property
    def name_ending(self) -> str:
        return self._name_ending

    @name_ending.setter
    def name_ending(self, name_ending: str) -> None:
        self._name_ending = name_ending

    @property  # type: ignore
    def phi_max(self) -> float:
        return self._phi_max

    @phi_max.setter
    def phi_max(self, phi_max: float) -> None:
        if isinstance(phi_max, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            phi_max = Region._phi_max
        if phi_max < PHI_MIN_DEFAULT:
            raise ValueError("phi_max cannot be negative")
        if phi_max > PHI_MAX_DEFAULT:
            raise ValueError(
                f"phi_max cannot be greater than {multiples_of_pi(PHI_MAX_DEFAULT)}"
            )
        self._phi_max = phi_max

    @property  # type: ignore
    def phi_min(self) -> float:
        return self._phi_min

    @phi_min.setter
    def phi_min(self, phi_min: float) -> None:
        if isinstance(phi_min, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            phi_min = Region._phi_min
        if phi_min < PHI_MIN_DEFAULT:
            raise ValueError("phi_min cannot be negative")
        if phi_min > PHI_MAX_DEFAULT:
            raise ValueError(
                f"phi_min cannot be greater than {multiples_of_pi(PHI_MAX_DEFAULT)}"
            )
        self._phi_min = phi_min

    @property
    def region_type(self) -> str:
        return self._region_type

    @region_type.setter
    def region_type(self, region_type: str) -> None:
        self._region_type = region_type

    @property  # type: ignore
    def theta_max(self) -> float:
        return self._theta_max

    @theta_max.setter
    def theta_max(self, theta_max: float) -> None:
        if isinstance(theta_max, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            theta_max = Region._theta_max
        if theta_max < THETA_MIN_DEFAULT:
            raise ValueError("theta_max cannot be negative")
        if theta_max > THETA_MAX_DEFAULT:
            raise ValueError(
                f"theta_max cannot be greater than {multiples_of_pi(THETA_MAX_DEFAULT)}"
            )
        self._theta_max = theta_max

    @property  # type: ignore
    def theta_min(self) -> float:
        return self._theta_min

    @theta_min.setter
    def theta_min(self, theta_min: float) -> None:
        if isinstance(theta_min, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            theta_min = Region._theta_min
        if theta_min < THETA_MIN_DEFAULT:
            raise ValueError("theta_min cannot be negative")
        if theta_min > THETA_MAX_DEFAULT:
            raise ValueError(
                f"theta_min cannot be greater than {multiples_of_pi(THETA_MAX_DEFAULT)}"
            )
        self._theta_min = theta_min
