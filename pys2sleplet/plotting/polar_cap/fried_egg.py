import numpy as np

from pys2sleplet.flm.kernels.slepian import Slepian
from pys2sleplet.plotting.inputs import THETA_MAX
from pys2sleplet.scripts.plotting import plot
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.region import Region

L = 19
ORDER_RANK = {0: 4, 1: 3, 2: 3, 3: 2, 4: 2, 5: 2, 6: 1, 7: 1}


def main() -> None:
    """
    create fig 5.4 from Spatiospectral Concentration on a Sphere by Simons et al 2006
    """
    for order, rank in ORDER_RANK.items():
        _helper(order, rank)
        if order != 0:
            _helper(-order, rank)


def _helper(order: int, rank: int) -> None:
    """
    helper which plots the required order and specified ranks
    """
    region = Region(theta_max=np.deg2rad(THETA_MAX), order=order)
    for r in range(rank):
        logger.info(f"plotting order={order}, rank={r}")
        slepian = Slepian(L, rank=r, region=region)
        plot(slepian)


if __name__ == "__main__":
    main()
