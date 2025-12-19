from __future__ import annotations

import typing

import numpy as np
import pytest

import sleplet
import sleplet._vars

if typing.TYPE_CHECKING:
    import numpy.typing as npt


ARRAY_DIM = 10
L = 16
MASK = "south_america"
PHI_0 = np.pi / 6
PHI_1 = np.pi / 3
RNG = np.random.default_rng(sleplet._vars.RANDOM_SEED)
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
def earth_polar_cap(
    slepian_polar_cap: sleplet.slepian.SlepianPolarCap,
) -> sleplet.functions.Earth:
    """Earth with polar cap region."""
    return sleplet.functions.Earth(slepian_polar_cap.L, region=slepian_polar_cap.region)


@pytest.fixture(scope="session")
def earth_lim_lat_lon(
    slepian_lim_lat_lon: sleplet.slepian.SlepianLimitLatLon,
) -> sleplet.functions.Earth:
    """Earth with limited latitude longitude region."""
    return sleplet.functions.Earth(
        slepian_lim_lat_lon.L,
        region=slepian_lim_lat_lon.region,
    )


@pytest.fixture(scope="session")
def south_america_arbitrary(
    slepian_arbitrary: sleplet.slepian.SlepianLimitLatLon,
) -> sleplet.functions.SouthAmerica:
    """South America already has region."""
    return sleplet.functions.SouthAmerica(slepian_arbitrary.L)


@pytest.fixture(scope="session")
def slepian_dirac_delta_polar_cap(
    slepian_polar_cap: sleplet.slepian.SlepianPolarCap,
) -> sleplet.functions.SlepianDiracDelta:
    """Create a polar cap Slepian Dirac delta."""
    return sleplet.functions.SlepianDiracDelta(
        slepian_polar_cap.L,
        region=slepian_polar_cap.region,
    )


@pytest.fixture(scope="session")
def slepian_dirac_delta_lim_lat_lon(
    slepian_lim_lat_lon: sleplet.slepian.SlepianLimitLatLon,
) -> sleplet.functions.SlepianDiracDelta:
    """Create a limited latitude longitude Slepian Dirac delta."""
    return sleplet.functions.SlepianDiracDelta(
        slepian_lim_lat_lon.L,
        region=slepian_lim_lat_lon.region,
    )


@pytest.fixture(scope="session")
def slepian_wavelets_polar_cap(
    slepian_polar_cap: sleplet.slepian.SlepianPolarCap,
) -> sleplet.functions.SlepianWavelets:
    """Compute the Slepian wavelets for the polar cap region."""
    return sleplet.functions.SlepianWavelets(
        slepian_polar_cap.L,
        region=slepian_polar_cap.region,
    )


@pytest.fixture(scope="session")
def slepian_wavelets_lim_lat_lon(
    slepian_lim_lat_lon: sleplet.slepian.SlepianLimitLatLon,
) -> sleplet.functions.SlepianWavelets:
    """Compute the Slepian wavelets for the lim_lat_lon region."""
    return sleplet.functions.SlepianWavelets(
        slepian_lim_lat_lon.L,
        region=slepian_lim_lat_lon.region,
    )


@pytest.fixture(scope="session")
def random_flm() -> npt.NDArray[np.complex128]:
    """Create random flm."""
    return sleplet.harmonic_methods.compute_random_signal(L, RNG, var_signal=1)


@pytest.fixture(scope="session")
def mesh() -> sleplet.meshes.Mesh:
    """Create a bird mesh."""
    return sleplet.meshes.Mesh("bird")


@pytest.fixture(scope="session")
def mesh_slepian(mesh: sleplet.meshes.Mesh) -> sleplet.meshes.MeshSlepian:
    """Create a Slepian bird mesh."""
    return sleplet.meshes.MeshSlepian(mesh)


@pytest.fixture(scope="session")
def mesh_field_region(mesh: sleplet.meshes.Mesh) -> sleplet.meshes.MeshField:
    """Create a field on the mesh."""
    return sleplet.meshes.MeshField(mesh, region=True)


@pytest.fixture(scope="session")
def mesh_slepian_wavelets(
    mesh: sleplet.meshes.Mesh,
) -> sleplet.meshes.MeshSlepianWavelets:
    """Create a field on the mesh."""
    return sleplet.meshes.MeshSlepianWavelets(mesh)
