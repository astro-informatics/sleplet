import pytest

from pys2sleplet.functions.flm.earth import Earth
from pys2sleplet.slepian.slepian_region.slepian_limit_lat_lon import SlepianLimitLatLon
from pys2sleplet.slepian.slepian_region.slepian_polar_cap import SlepianPolarCap
from pys2sleplet.test.constants import L_SMALL as L
from pys2sleplet.test.constants import PHI_0, PHI_1, THETA_0, THETA_1, THETA_MAX


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
    return SlepianPolarCap(L, THETA_MAX)


@pytest.fixture(scope="session")
def slepian_lim_lat_lon() -> SlepianLimitLatLon:
    """
    create a Slepian limited latitude longitude class
    """
    return SlepianLimitLatLon(
        L, theta_min=THETA_0, theta_max=THETA_1, phi_min=PHI_0, phi_max=PHI_1
    )


@pytest.fixture(scope="session")
def earth_polar_cap(slepian_polar_cap) -> Earth:
    return Earth(L, region=slepian_polar_cap.region)


@pytest.fixture(scope="session")
def earth_lim_lat_lon(slepian_lim_lat_lon) -> Earth:
    return Earth(L, region=slepian_lim_lat_lon.region)
