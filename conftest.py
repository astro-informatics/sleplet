import pytest

from pys2sleplet.flm.maps.earth import Earth
from pys2sleplet.slepian.slepian_decomposition import SlepianDecomposition
from pys2sleplet.slepian.slepian_region.slepian_limit_lat_lon import SlepianLimitLatLon
from pys2sleplet.slepian.slepian_region.slepian_polar_cap import SlepianPolarCap
from pys2sleplet.test.constants import L_SMALL as L
from pys2sleplet.test.constants import ORDER, PHI_0, PHI_1, THETA_0, THETA_1, THETA_MAX
from pys2sleplet.utils.region import Region


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def slepian_polar_cap() -> SlepianPolarCap:
    """
    create a Slepian polar cap class
    """
    slepian = SlepianPolarCap(L, THETA_MAX, order=ORDER)
    return slepian


@pytest.fixture(scope="session")
def slepian_lim_lat_lon() -> SlepianLimitLatLon:
    """
    create a Slepian limited latitude longitude class
    """
    slepian = SlepianLimitLatLon(
        L, theta_min=THETA_0, theta_max=THETA_1, phi_min=PHI_0, phi_max=PHI_1
    )
    return slepian


@pytest.fixture(scope="session")
def polar_cap_region() -> Region:
    """
    creates a polar cap region
    """
    region = Region(theta_max=THETA_MAX, order=ORDER)
    return region


@pytest.fixture(scope="session")
def lim_lat_lon_region() -> Region:
    """
    creates a limited latitude longitude region
    """
    region = Region(theta_min=THETA_0, theta_max=THETA_1, phi_min=PHI_0, phi_max=PHI_1)
    return region


@pytest.fixture(scope="session")
def polar_cap_decomposition(polar_cap_region) -> SlepianDecomposition:
    earth = Earth(L, region=polar_cap_region)
    coefficients = SlepianDecomposition(earth)
    return coefficients


@pytest.fixture(scope="session")
def lim_lat_lon_decomposition(lim_lat_lon_region) -> SlepianDecomposition:
    earth = Earth(L, region=lim_lat_lon_region)
    coefficients = SlepianDecomposition(earth)
    return coefficients
