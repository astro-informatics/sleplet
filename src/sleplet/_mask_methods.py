import logging
import os
import typing

import numpy as np
import numpy.typing as npt
import platformdirs

import pyssht as ssht

import sleplet._data.create_earth_flm
import sleplet._data.setup_pooch
import sleplet._vars
import sleplet.harmonic_methods
import sleplet.meshes.mesh
import sleplet.slepian.region

_logger = logging.getLogger(__name__)

_AFRICA_RANGE = np.deg2rad(41)
_SOUTH_AMERICA_RANGE = np.deg2rad(40)


def create_mask_region(
    L: int,
    region: "sleplet.slepian.region.Region",
) -> npt.NDArray[np.float_]:
    """
    Create a mask of a region of interested, the output will be based
    on the value of the provided L. The mask could be either:
    * polar cap - if theta_max provided
    * limited latitude longitude - if one of theta_min, theta_max,
                                   phi_min or phi_max is provided
    * arbitrary - just checks the shape of the input mask.
    """
    thetas, phis = ssht.sample_positions(
        L,
        Grid=True,
        Method=sleplet._vars.SAMPLING_SCHEME,
    )

    match region._region_type:
        case "arbitrary":
            _logger.info("loading and checking shape of provided mask")
            name = f"{region.mask_name}_L{L}.npy"
            mask = _load_mask(L, name)
            assert mask.shape == thetas.shape, (  # noqa: S101
                f"mask {name} has shape {mask.shape} which does not match "
                f"the provided L={L}, the shape should be {thetas.shape}"
            )

        case "lim_lat_lon":
            _logger.info("creating limited latitude longitude mask")
            mask = (
                (thetas >= region.theta_min)
                & (thetas <= region.theta_max)
                & (phis >= region.phi_min)
                & (phis <= region.phi_max)
            )

        case "polar":
            _logger.info("creating polar cap mask")
            mask = thetas <= region.theta_max
            if region.gap:
                _logger.info("creating polar gap mask")
                mask += thetas >= np.pi - region.theta_max
    return mask


def _load_mask(L: int, mask_name: str) -> npt.NDArray[np.float_]:
    """Attempt to read the mask from the config file."""
    mask = sleplet._data.setup_pooch.find_on_pooch_then_local(
        f"slepian_masks_{mask_name}",
    )
    return create_mask(L, mask_name) if mask is None else np.load(mask)


def ensure_masked_flm_bandlimited(
    flm: npt.NDArray[np.complex_],
    L: int,
    region: "sleplet.slepian.region.Region",
    *,
    reality: bool,
    spin: int,
) -> npt.NDArray[np.complex_]:
    """Ensure the coefficients is bandlimited for a given region."""
    field = ssht.inverse(
        flm,
        L,
        Method=sleplet._vars.SAMPLING_SCHEME,
        Reality=reality,
        Spin=spin,
    )
    mask = create_mask_region(L, region)
    field = np.where(mask, field, 0)
    return ssht.forward(
        field,
        L,
        Method=sleplet._vars.SAMPLING_SCHEME,
        Reality=reality,
        Spin=spin,
    )


def create_default_region() -> "sleplet.slepian.region.Region":
    """Create default region."""
    return sleplet.slepian.region.Region(
        gap=os.getenv("POLAR_GAP", "False").lower() == "true",
        mask_name=os.getenv("SLEPIAN_MASK", "south_america"),
        phi_max=np.deg2rad(int(os.getenv("PHI_MAX", "360"))),
        phi_min=np.deg2rad(int(os.getenv("PHI_MIN", "0"))),
        theta_max=np.deg2rad(int(os.getenv("THETA_MAX", "180"))),
        theta_min=np.deg2rad(int(os.getenv("THETA_MIN", "0"))),
    )


def create_mesh_region(
    mesh_config: dict[str, typing.Any],
    vertices: npt.NDArray[np.float_],
) -> npt.NDArray[np.bool_]:
    """Create a boolean region for the given mesh."""
    return (
        (vertices[:, 0] >= mesh_config["XMIN"])
        & (vertices[:, 0] <= mesh_config["XMAX"])
        & (vertices[:, 1] >= mesh_config["YMIN"])
        & (vertices[:, 1] <= mesh_config["YMAX"])
        & (vertices[:, 2] >= mesh_config["ZMIN"])
        & (vertices[:, 2] <= mesh_config["ZMAX"])
    )


def ensure_masked_bandlimit_mesh_signal(
    mesh: "sleplet.meshes.mesh.Mesh",
    u_i: npt.NDArray[np.complex_ | np.float_],
) -> npt.NDArray[np.float_]:
    """Ensure that signal in pixel space is bandlimited."""
    field = sleplet.harmonic_methods.mesh_inverse(mesh, u_i)
    masked_field = np.where(mesh.mesh_region, field, 0)
    return sleplet.harmonic_methods.mesh_forward(mesh, masked_field)


def convert_region_on_vertices_to_faces(
    mesh: "sleplet.meshes.mesh.Mesh",
) -> npt.NDArray[np.float_]:
    """Convert the region on vertices to faces."""
    region_reshape = np.argwhere(mesh.mesh_region).reshape(-1)
    faces_in_region = np.isin(mesh.faces, region_reshape).all(axis=1)
    region_on_faces = np.zeros(mesh.faces.shape[0])
    region_on_faces[faces_in_region] = 1
    return region_on_faces


def _create_africa_mask(
    L: int,
    earth_flm: npt.NDArray[np.complex_],
) -> npt.NDArray[np.float_]:
    """Create the Africa region mask."""
    rot_flm = sleplet.harmonic_methods.rotate_earth_to_africa(earth_flm, L)
    earth_f = ssht.inverse(
        rot_flm,
        L,
        Method=sleplet._vars.SAMPLING_SCHEME,
        Reality=True,
    )
    thetas, _ = ssht.sample_positions(
        L,
        Grid=True,
        Method=sleplet._vars.SAMPLING_SCHEME,
    )
    return (thetas <= _AFRICA_RANGE) & (earth_f >= 0)


def _create_south_america_mask(
    L: int,
    earth_flm: npt.NDArray[np.complex_],
) -> npt.NDArray[np.float_]:
    """Create the Africa region mask."""
    rot_flm = sleplet.harmonic_methods.rotate_earth_to_south_america(earth_flm, L)
    earth_f = ssht.inverse(
        rot_flm,
        L,
        Method=sleplet._vars.SAMPLING_SCHEME,
        Reality=True,
    )
    thetas, _ = ssht.sample_positions(
        L,
        Grid=True,
        Method=sleplet._vars.SAMPLING_SCHEME,
    )
    return (thetas <= _SOUTH_AMERICA_RANGE) & (earth_f >= 0)


def create_mask(L: int, mask_name: str) -> npt.NDArray[np.float_]:
    """Create the South America region mask."""
    earth_flm = sleplet._data.create_earth_flm.create_flm(L)
    if mask_name == f"africa_L{L}.npy":
        mask = _create_africa_mask(L, earth_flm)
    elif mask_name == f"south_america_L{L}.npy":
        mask = _create_south_america_mask(L, earth_flm)
    else:
        msg = f"Mask name {mask_name} not recognised"
        raise ValueError(msg)
    np.save(platformdirs.user_data_path() / f"slepian_masks_{mask_name}", mask)
    return mask
