import sleplet._vars


def is_polar_cap(
    phi_min: float,
    phi_max: float,
    theta_min: float,
    theta_max: float,
) -> bool:
    """Circular mask at the north pole."""
    return (
        phi_min == sleplet._vars.PHI_MIN_DEFAULT
        and phi_max == sleplet._vars.PHI_MAX_DEFAULT
        and theta_min == sleplet._vars.THETA_MIN_DEFAULT
        and theta_max != sleplet._vars.THETA_MAX_DEFAULT
    )


def is_limited_lat_lon(
    phi_min: float,
    phi_max: float,
    theta_min: float,
    theta_max: float,
) -> bool:
    """Define a region by angles, just need one to not be the default."""
    return (
        not is_polar_cap(phi_min, phi_max, theta_min, theta_max)
        and phi_min != sleplet._vars.PHI_MIN_DEFAULT
        or phi_max != sleplet._vars.PHI_MAX_DEFAULT
        or theta_min != sleplet._vars.THETA_MIN_DEFAULT
    )
