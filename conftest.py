import numpy as np
import pytest
from numpy.random import default_rng

from pys2sleplet.functions.flm.earth import Earth
from pys2sleplet.functions.fp.slepian_dirac_delta import SlepianDiracDelta
from pys2sleplet.slepian.slepian_region.slepian_limit_lat_lon import SlepianLimitLatLon
from pys2sleplet.slepian.slepian_region.slepian_polar_cap import SlepianPolarCap
from pys2sleplet.test.constants import (
    L_SMALL,
    PHI_0,
    PHI_1,
    THETA_0,
    THETA_1,
    THETA_MAX,
)
from pys2sleplet.utils.harmonic_methods import compute_random_signal
from pys2sleplet.utils.vars import RANDOM_SEED


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
    return SlepianPolarCap(L_SMALL, THETA_MAX)


@pytest.fixture(scope="session")
def slepian_lim_lat_lon() -> SlepianLimitLatLon:
    """
    create a Slepian limited latitude longitude class
    """
    return SlepianLimitLatLon(
        L_SMALL, theta_min=THETA_0, theta_max=THETA_1, phi_min=PHI_0, phi_max=PHI_1
    )


@pytest.fixture(scope="session")
def earth_polar_cap(slepian_polar_cap) -> Earth:
    """
    Earth with polar cap region
    """
    return Earth(L_SMALL, region=slepian_polar_cap.region)


@pytest.fixture(scope="session")
def earth_lim_lat_lon(slepian_lim_lat_lon) -> Earth:
    """
    Earth with limited latitude longitude region
    """
    return Earth(L_SMALL, region=slepian_lim_lat_lon.region)


@pytest.fixture(scope="session")
def slepian_dirac_delta_polar_cap(slepian_polar_cap) -> SlepianDiracDelta:
    """
    Creates a polar cap Slepian Dirac delta
    """
    return SlepianDiracDelta(L_SMALL, region=slepian_polar_cap.region)


@pytest.fixture(scope="session")
def slepian_dirac_delta_lim_lat_lon(slepian_lim_lat_lon) -> SlepianDiracDelta:
    """
    Creates a limited latitude longitude Slepian Dirac delta
    """
    return SlepianDiracDelta(L_SMALL, region=slepian_lim_lat_lon.region)


@pytest.fixture(scope="session")
def random_flm() -> np.ndarray:
    """
    creates random flm
    """
    rng = default_rng(RANDOM_SEED)
    return compute_random_signal(L_SMALL, rng, 1)
