from pydantic import validator
from pydantic.dataclasses import dataclass

from sleplet.utils.bool_methods import is_limited_lat_lon, is_polar_cap
from sleplet.utils.string_methods import angle_as_degree, multiples_of_pi
from sleplet.utils.validation import Validation
from sleplet.utils.vars import (
    PHI_MAX_DEFAULT,
    PHI_MIN_DEFAULT,
    THETA_MAX_DEFAULT,
    THETA_MIN_DEFAULT,
)


@dataclass(config=Validation, kw_only=True)
class Region:
    gap: bool = False
    mask_name: str = ""
    phi_max: float = PHI_MAX_DEFAULT
    phi_min: float = PHI_MIN_DEFAULT
    theta_max: float = THETA_MAX_DEFAULT
    theta_min: float = THETA_MIN_DEFAULT

    def __post_init_post_parse__(self) -> None:
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

    @validator("phi_max")
    def check_phi_max(cls, v):
        if v < PHI_MIN_DEFAULT:
            raise ValueError("phi_max cannot be negative")
        if v > PHI_MAX_DEFAULT:
            raise ValueError(
                f"phi_max cannot be greater than {multiples_of_pi(PHI_MAX_DEFAULT)}"
            )
        return v

    @validator("phi_min")
    def check_phi_min(cls, v):
        if v < PHI_MIN_DEFAULT:
            raise ValueError("phi_min cannot be negative")
        if v > PHI_MAX_DEFAULT:
            raise ValueError(
                f"phi_min cannot be greater than {multiples_of_pi(PHI_MAX_DEFAULT)}"
            )
        return v

    @validator("theta_max")
    def check_theta_max(cls, v):
        if v < THETA_MIN_DEFAULT:
            raise ValueError("theta_max cannot be negative")
        if v > THETA_MAX_DEFAULT:
            raise ValueError(
                f"theta_max cannot be greater than {multiples_of_pi(THETA_MAX_DEFAULT)}"
            )
        return v

    @validator("theta_min")
    def check_theta_min(cls, v):
        if v < THETA_MIN_DEFAULT:
            raise ValueError("theta_min cannot be negative")
        if v > THETA_MAX_DEFAULT:
            raise ValueError(
                f"theta_min cannot be greater than {multiples_of_pi(THETA_MAX_DEFAULT)}"
            )
        return v
