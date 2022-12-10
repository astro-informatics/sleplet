import numpy as np
from numpy.testing import assert_allclose

from sleplet.functions.f_lm import F_LM
from sleplet.functions.flm.south_america import SouthAmerica
from sleplet.slepian.slepian_region.slepian_arbitrary import SlepianArbitrary
from sleplet.utils.logger import logger
from sleplet.utils.slepian_methods import slepian_forward
from sleplet.utils.string_methods import convert_camel_case_to_snake_case
from sleplet.utils.vars import SMOOTHING

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
    logger.info(f"leakage={1-proportion_inside:.2%}")


if __name__ == "__main__":
    earth_region = SouthAmerica(L, smoothing=SMOOTHING)
    compute_slepian_energy_leakage(earth_region)
