"""Contains the `Region` class."""
import logging

import pydantic
import typing_extensions

import sleplet._bool_methods
import sleplet._string_methods
import sleplet._validation
import sleplet._vars

_logger = logging.getLogger(__name__)


@pydantic.dataclasses.dataclass(config=sleplet._validation.validation, kw_only=True)
class Region:
    """Identify and create the appropriate Slepian region for the sphere."""

    gap: bool = False
    """Whether to enable a double ended polar cap, set by the `POLAR_GAP`
    environment variable. Only relevant if `theta_max` is not `180` and the
    other angles are at their default values."""
    mask_name: str = ""
    """The name of the mask of the arbitrary region, set by the `SLEPIAN_MASK`
    environment variable. Current options are `africa` and `south_america`."""
    phi_max: float = sleplet._vars.PHI_MAX_DEFAULT
    """For a limited latitude longitude region, set by the `PHI_MAX` environment
    variable."""
    phi_min: float = sleplet._vars.PHI_MIN_DEFAULT
    """For a limited latitude longitude region, set by the `PHI_MIN` environment
    variable."""
    theta_max: float = sleplet._vars.THETA_MAX_DEFAULT
    """Set by the `THETA_MAX` environment variable. When set without the
    other angles it controls a polar cap region. When in conjunction with the
    others it is for a limited latitude longitude region."""
    theta_min: float = sleplet._vars.THETA_MIN_DEFAULT
    """For a limited latitude longitude region, set by the `THETA_MIN` environment
    variable."""
    _name_ending: str = pydantic.Field(default="", init_var=False, repr=False)
    _region_type: str = pydantic.Field(default="", init_var=False, repr=False)

    def __post_init__(self: typing_extensions.Self) -> None:
        self._identify_region()

    def _identify_region(self: typing_extensions.Self) -> None:
        """Identify region type based on the angle inputs or a mask name."""
        msg = (
            "Slepian region values detected: "
            f"POLAR_GAP={self.gap}, "
            f"THETA_MAX={self.theta_max}, "
            f"THETA_MIN={self.theta_min}, "
            f"PHI_MAX={self.phi_max}, "
            f"PHI_MIN={self.phi_min}, "
            f"SLEPIAN_MASK={self.mask_name}.",
        )
        _logger.info(msg)
        if sleplet._bool_methods.is_polar_cap(
            self.phi_min,
            self.phi_max,
            self.theta_min,
            self.theta_max,
        ):
            self._region_type = "polar"
            self._name_ending = (
                f"polar{'_gap' if self.gap else ''}"
                f"{sleplet._string_methods.angle_as_degree(self.theta_max)}"
            )

        elif sleplet._bool_methods.is_limited_lat_lon(
            self.phi_min,
            self.phi_max,
            self.theta_min,
            self.theta_max,
        ):
            self._region_type = "lim_lat_lon"
            self._name_ending = (
                f"theta{sleplet._string_methods.angle_as_degree(self.theta_min)}"
                f"-{sleplet._string_methods.angle_as_degree(self.theta_max)}"
                f"_phi{sleplet._string_methods.angle_as_degree(self.phi_min)}"
                f"-{sleplet._string_methods.angle_as_degree(self.phi_max)}"
            )

        elif self.mask_name:
            self._region_type = "arbitrary"
            self._name_ending = self.mask_name

        else:
            msg = (
                "need to specify either a polar cap, a limited latitude "
                "longitude region, or a file with a mask",
            )
            raise AttributeError(msg)

    @pydantic.field_validator("phi_max")
    def _check_phi_max(
        cls,  # noqa: ANN101
        v: float,
    ) -> float:
        if v < sleplet._vars.PHI_MIN_DEFAULT:
            msg = "phi_max cannot be negative"
            raise ValueError(msg)
        if v > sleplet._vars.PHI_MAX_DEFAULT:
            msg = (
                f"phi_max cannot be greater than "
                f"{sleplet._string_methods.multiples_of_pi(sleplet._vars.PHI_MAX_DEFAULT)}"
            )
            raise ValueError(msg)
        return v

    @pydantic.field_validator("phi_min")
    def _check_phi_min(
        cls,  # noqa: ANN101
        v: float,
    ) -> float:
        if v < sleplet._vars.PHI_MIN_DEFAULT:
            msg = "phi_min cannot be negative"
            raise ValueError(msg)
        if v > sleplet._vars.PHI_MAX_DEFAULT:
            msg = (
                f"phi_min cannot be greater than "
                f"{sleplet._string_methods.multiples_of_pi(sleplet._vars.PHI_MAX_DEFAULT)}"
            )
            raise ValueError(msg)
        return v

    @pydantic.field_validator("theta_max")
    def _check_theta_max(
        cls,  # noqa: ANN101
        v: float,
    ) -> float:
        if v < sleplet._vars.THETA_MIN_DEFAULT:
            msg = "theta_max cannot be negative"
            raise ValueError(msg)
        if v > sleplet._vars.THETA_MAX_DEFAULT:
            msg = (
                "theta_max cannot be greater than "
                f"{sleplet._string_methods.multiples_of_pi(sleplet._vars.THETA_MAX_DEFAULT)}"
            )
            raise ValueError(msg)
        return v

    @pydantic.field_validator("theta_min")
    def _check_theta_min(
        cls,  # noqa: ANN101
        v: float,
    ) -> float:
        if v < sleplet._vars.THETA_MIN_DEFAULT:
            msg = "theta_min cannot be negative"
            raise ValueError(msg)
        if v > sleplet._vars.THETA_MAX_DEFAULT:
            msg = (
                "theta_min cannot be greater than "
                f"{sleplet._string_methods.multiples_of_pi(sleplet._vars.THETA_MAX_DEFAULT)}"
            )
            raise ValueError(msg)
        return v
