from typing import List

from pys2sleplet.flm.kernels.slepian import Slepian
from pys2sleplet.scripts.plotting import plot
from pys2sleplet.scripts.simons_fried_egg.simons_inputs import ORDER_DICT, THETA_MAX, L
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.region import Region


def produce_figure() -> None:
    """
    create fig 5.4 from Spatiospectral Concentration on a Sphere by Simons et al 2006
    """
    for order, ranks in ORDER_DICT.items():
        _helper(order, ranks)
        if order != 0:
            _helper(-order, ranks)


def _helper(order: int, ranks: List[int]) -> None:
    """
    helper which plots the required order and specified ranks
    """
    region = Region(theta_max=THETA_MAX, order=order)
    for rank in ranks:
        logger.info(f"plotting order={order}, rank={rank}")
        slepian = Slepian(L, rank=rank, region=region)
        plot(slepian)
