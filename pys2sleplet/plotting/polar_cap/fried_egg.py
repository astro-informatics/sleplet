import numpy as np

from pys2sleplet.functions.fp.slepian import Slepian
from pys2sleplet.plotting.inputs import THETA_MAX
from pys2sleplet.scripts.plotting_on_sphere import plot
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.region import Region

L = 19
RANKS = 32


def main() -> None:
    """
    create fig 5.4 from Spatiospectral Concentration on a Sphere by Simons et al 2006
    """
    region = Region(theta_max=np.deg2rad(THETA_MAX))
    for r in range(RANKS):
        logger.info(f"plotting rank={r}")
        slepian = Slepian(L, rank=r, region=region)
        plot(slepian)


if __name__ == "__main__":
    main()
