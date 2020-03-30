# from pys2sleplet.utils.slepian_methods import apply_slepian_mask
# from pys2sleplet.flm.maps.earth import Earth
# import pytest
# from pys2sleplet.utils.config import config
# from pys2sleplet.slepian.slepian_region.slepian_arbitrary import SlepianArbitrary
# from pys2sleplet.slepian.slepian_region.specific_region.slepian_limit_lat_long import (
#     SlepianLimitLatLong,
# )
# from pys2sleplet.slepian.slepian_region.specific_region.slepian_polar_cap import (
#     SlepianPolarCap,
# )
# import numpy as np


# @pytest.fixture
# def earth() -> Earth:
#     return Earth(config.L)


# @pytest.fixture
# def theta_min() -> float:
#     return np.deg2rad(20)


# @pytest.fixture
# def theta_max() -> float:
#     return np.deg2rad(40)


# @pytest.fixture
# def phi_min() -> float:
#     return np.deg2rad(20)


# @pytest.fixture
# def phi_max() -> float:
#     return np.deg2rad(40)


# def test_apply_mask_slepian_polar_cap(earth, theta_max) -> None:
#     """
#     """
#     slepian = SlepianPolarCap(config.L, theta_max)
#     apply_slepian_mask(earth, slepian)


# def test_apply_mask_slepian_lim_lat_lon(
#     earth, theta_min, theta_max, phi_min, phi_max
# ) -> None:
#     """
#     """
#     slepian = SlepianLimitLatLong(config.L, theta_min, theta_max, phi_min, phi_max)
#     apply_slepian_mask(earth, slepian)


# def test_apply_mask_slepian_arbitrary() -> None:
#     """
#     """
#     pass
