import numpy as np
from numpy.testing import assert_equal

import sleplet.slepian

MASK = "south_america"
PHI_0 = np.pi / 6
PHI_1 = np.pi
THETA_0 = np.pi / 6
THETA_1 = np.pi / 3
THETA_MAX = 2 * np.pi / 9


def test_polar_cap_region() -> None:
    """Tests that a polar cap is created."""
    region = sleplet.slepian.Region(theta_max=THETA_MAX)
    assert_equal(region.region_type, "polar")


def test_lim_lat_lon_region() -> None:
    """Tests that a limited latitude longitude region is created."""
    region = sleplet.slepian.Region(
        theta_min=THETA_0,
        theta_max=THETA_1,
        phi_min=PHI_0,
        phi_max=PHI_1,
    )
    assert_equal(region.region_type, "lim_lat_lon")


def test_arbitrary_region() -> None:
    """Tests that an arbitrary region is created."""
    region = sleplet.slepian.Region(mask_name=MASK)
    assert_equal(region.region_type, "arbitrary")


def test_polar_hierarchy_over_arbirary() -> None:
    """Ensures polar cap is made instead of arbitrary."""
    region = sleplet.slepian.Region(theta_max=THETA_MAX, mask_name=MASK)
    assert_equal(region.region_type, "polar")


def test_lim_lat_lon_hierarchy_over_arbirary() -> None:
    """Ensures limited latitude longitude region is made instead of arbitrary."""
    region = sleplet.slepian.Region(
        theta_min=THETA_0,
        theta_max=THETA_1,
        phi_min=PHI_0,
        phi_max=PHI_1,
        mask_name=MASK,
    )
    assert_equal(region.region_type, "lim_lat_lon")
