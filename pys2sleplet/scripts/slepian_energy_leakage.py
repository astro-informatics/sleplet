import numpy as np
from numpy.testing import assert_allclose

from pys2sleplet.functions.f_lm import F_LM
from pys2sleplet.functions.flm.south_america import SouthAmerica
from pys2sleplet.slepian.slepian_region.slepian_arbitrary import SlepianArbitrary
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.slepian_methods import slepian_forward
from pys2sleplet.utils.string_methods import convert_camel_case_to_snake_case
from pys2sleplet.utils.vars import SMOOTHING

L = 16


def compute_slepian_energy_leakage(earth_region: F_LM) -> None:
    """
    compute the proportion of energy within the Shannon number for South America
    """
    # compute full set of Slepian coefficients
    region = convert_camel_case_to_snake_case(earth_region.__class__.__name__)
    slepian = SlepianArbitrary(earth_region.L, region)
    f_p = slepian_forward(
        earth_region.L,
        slepian,
        flm=earth_region.coefficients,
        n_coeffs=earth_region.L**2,
    )

    # calculate energy
    slepian_energy = np.abs(f_p) ** 2
    proportion_inside = slepian_energy[: slepian.N].sum() / slepian_energy.sum()

    assert_allclose(proportion_inside, 1, atol=0.032)
    logger.info(f"proportion inside={proportion_inside:.2%}")


if __name__ == "__main__":
    earth_region = SouthAmerica(L, smoothing=SMOOTHING)
    compute_slepian_energy_leakage(earth_region)
