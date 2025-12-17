from __future__ import annotations

import numpy as np

import sleplet

MASK = "south_america"
PHI_0 = np.pi / 6
PHI_1 = np.pi
THETA_0 = np.pi / 6
THETA_1 = np.pi / 3
THETA_MAX = 2 * np.pi / 9


def test_polar_cap_region() -> None:
    """Test that a polar cap is created."""
    region = sleplet.slepian.Region(theta_max=THETA_MAX)
    np.testing.assert_equal(region._region_type, "polar")


def test_lim_lat_lon_region() -> None:
    """Test that a limited latitude longitude region is created."""
    region = sleplet.slepian.Region(
        theta_min=THETA_0,
        theta_max=THETA_1,
        phi_min=PHI_0,
        phi_max=PHI_1,
    )
    np.testing.assert_equal(region._region_type, "lim_lat_lon")


def test_arbitrary_region() -> None:
    """Test that an arbitrary region is created."""
    region = sleplet.slepian.Region(mask_name=MASK)
    np.testing.assert_equal(region._region_type, "arbitrary")


def test_polar_hierarchy_over_arbitrary() -> None:
    """Ensure polar cap is made instead of arbitrary."""
    region = sleplet.slepian.Region(theta_max=THETA_MAX, mask_name=MASK)
    np.testing.assert_equal(region._region_type, "polar")


def test_lim_lat_lon_hierarchy_over_arbitrary() -> None:
    """Ensure limited latitude longitude region is made instead of arbitrary."""
    region = sleplet.slepian.Region(
        theta_min=THETA_0,
        theta_max=THETA_1,
        phi_min=PHI_0,
        phi_max=PHI_1,
        mask_name=MASK,
    )
    np.testing.assert_equal(region._region_type, "lim_lat_lon")
