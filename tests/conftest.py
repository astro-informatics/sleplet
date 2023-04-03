import numpy as np
import pytest
from numpy import typing as npt
from numpy.random import default_rng

import sleplet

ARRAY_DIM = 10
L = 16
MASK = "south_america"
PHI_0 = np.pi / 6
PHI_1 = np.pi / 3
THETA_0 = np.pi / 6
THETA_1 = np.pi / 3
THETA_MAX = 2 * np.pi / 9


@pytest.fixture(scope="session")
def slepian_polar_cap() -> sleplet.slepian.SlepianPolarCap:
    """Create a Slepian polar cap class."""
    return sleplet.slepian.SlepianPolarCap(L, THETA_MAX)


@pytest.fixture(scope="session")
def slepian_lim_lat_lon() -> sleplet.slepian.SlepianLimitLatLon:
    """Create a Slepian limited latitude longitude class."""
    return sleplet.slepian.SlepianLimitLatLon(
        L,
        theta_min=THETA_0,
        theta_max=THETA_1,
        phi_min=PHI_0,
        phi_max=PHI_1,
    )


@pytest.fixture(scope="session")
def slepian_arbitrary() -> sleplet.slepian.SlepianArbitrary:
    """Create a Slepian arbitrary class."""
    return sleplet.slepian.SlepianArbitrary(L, MASK)


@pytest.fixture(scope="session")
def earth_polar_cap(slepian_polar_cap) -> sleplet.functions.Earth:
    """Earth with polar cap region."""
    return sleplet.functions.Earth(slepian_polar_cap.L, region=slepian_polar_cap.region)


@pytest.fixture(scope="session")
def earth_lim_lat_lon(slepian_lim_lat_lon) -> sleplet.functions.Earth:
    """Earth with limited latitude longitude region."""
    return sleplet.functions.Earth(
        slepian_lim_lat_lon.L,
        region=slepian_lim_lat_lon.region,
    )


@pytest.fixture(scope="session")
def south_america_arbitrary(slepian_arbitrary) -> sleplet.functions.SouthAmerica:
    """South America already has region."""
    return sleplet.functions.SouthAmerica(slepian_arbitrary.L)


@pytest.fixture(scope="session")
def slepian_dirac_delta_polar_cap(
    slepian_polar_cap,
) -> sleplet.functions.SlepianDiracDelta:
    """Creates a polar cap Slepian Dirac delta."""
    return sleplet.functions.SlepianDiracDelta(
        slepian_polar_cap.L,
        region=slepian_polar_cap.region,
    )


@pytest.fixture(scope="session")
def slepian_dirac_delta_lim_lat_lon(
    slepian_lim_lat_lon,
) -> sleplet.functions.SlepianDiracDelta:
    """Creates a limited latitude longitude Slepian Dirac delta."""
    return sleplet.functions.SlepianDiracDelta(
        slepian_lim_lat_lon.L,
        region=slepian_lim_lat_lon.region,
    )


@pytest.fixture(scope="session")
def slepian_wavelets_polar_cap(slepian_polar_cap) -> sleplet.functions.SlepianWavelets:
    """Computes the Slepian wavelets for the polar cap region."""
    return sleplet.functions.SlepianWavelets(
        slepian_polar_cap.L,
        region=slepian_polar_cap.region,
    )


@pytest.fixture(scope="session")
def slepian_wavelets_lim_lat_lon(
    slepian_lim_lat_lon,
) -> sleplet.functions.SlepianWavelets:
    """Computes the Slepian wavelets for the lim_lat_lon region."""
    return sleplet.functions.SlepianWavelets(
        slepian_lim_lat_lon.L,
        region=slepian_lim_lat_lon.region,
    )


@pytest.fixture(scope="session")
def random_flm() -> npt.NDArray[np.complex_]:
    """Creates random flm."""
    rng = default_rng(sleplet._vars.RANDOM_SEED)
    return sleplet.harmonic_methods.compute_random_signal(L, rng, var_signal=1)


@pytest.fixture(scope="session")
def random_nd_flm() -> npt.NDArray[np.complex_]:
    """Creates multiple random flm."""
    rng = default_rng(sleplet._vars.RANDOM_SEED)
    return np.array(
        [
            sleplet.harmonic_methods.compute_random_signal(L, rng, var_signal=1)
            for _ in range(ARRAY_DIM)
        ],
    )


@pytest.fixture(scope="session")
def mesh() -> sleplet.meshes.Mesh:
    """Creates a bird mesh."""
    return sleplet.meshes.Mesh("bird")


@pytest.fixture(scope="session")
def mesh_slepian(mesh) -> sleplet.meshes.MeshSlepian:
    """Creates a Slepian bird mesh."""
    return sleplet.meshes.MeshSlepian(mesh)


@pytest.fixture(scope="session")
def mesh_field_region(mesh) -> sleplet.meshes.MeshField:
    """Creates a field on the mesh."""
    return sleplet.meshes.MeshField(mesh, region=True)


@pytest.fixture(scope="session")
def mesh_slepian_wavelets(mesh) -> sleplet.meshes.MeshSlepianWavelets:
    """Creates a field on the mesh."""
    return sleplet.meshes.MeshSlepianWavelets(mesh)
