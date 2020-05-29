# from pys2sleplet.utils.integration_methods import integrate_whole_sphere
# from pys2sleplet.slepian.slepian_region.slepian_polar_cap import SlepianPolarCap
# import pytest
# from hypothesis.strategies import integers, SearchStrategy
# from hypothesis import given, settings


# @pytest.fixture(scope="module")
# def L() -> int:
#     return 32


# def valid_theta_max() -> SearchStrategy[int]:
#     """
#     theta can be in the range [0, 180]
#     """
#     return integers(min_value=1, max_value=60)


# def valid_orders() -> SearchStrategy[int]:
#     """
#     the order of the Dm matrix, needs to be less than L
#     """
#     return integers(min_value=0, max_value=7)


# @settings(max_examples=8, derandomize=True, deadline=None)
# @given(theta_max=valid_theta_max(), order=valid_orders())
# def test_integrate_two_slepian_functions_whole_sphere(L) -> None:
#     """
#     """
#     slepian = SlepianPolarCap(L, theta_max)
