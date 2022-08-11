from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pyssht as ssht

from pys2sleplet.data.other.earth.create_earth_flm import create_flm
from pys2sleplet.plotting.create_plot_sphere import Plot
from pys2sleplet.utils.config import settings
from pys2sleplet.utils.plot_methods import rotate_earth_to_africa
from pys2sleplet.utils.vars import AFRICA_RANGE, SAMPLING_SCHEME

_file_location = Path(__file__).resolve()
_mask_path = _file_location.parents[3] / "data" / "slepian" / "masks"


def create_mask(L: int) -> None:
    """
    creates the Africa region mask
    """
    earth_flm = create_flm(L)
    rot_flm = rotate_earth_to_africa(earth_flm, L)
    earth_f = ssht.inverse(rot_flm, L, Reality=True, Method=SAMPLING_SCHEME)
    thetas, _ = ssht.sample_positions(L, Grid=True, Method=SAMPLING_SCHEME)
    mask = (thetas <= AFRICA_RANGE) & (earth_f >= 0)
    filename = f"africa_L{L}"
    Plot(mask.astype(np.complex_), L, f"mask_{filename}").execute()
    np.save(_mask_path / f"{filename}.npy", mask)


if __name__ == "__main__":
    parser = ArgumentParser(description="create the Africa region mask")
    parser.add_argument(
        "--bandlimit", "-L", type=int, default=settings.L, help="bandlimit"
    )
    args = parser.parse_args()
    create_mask(args.bandlimit)
