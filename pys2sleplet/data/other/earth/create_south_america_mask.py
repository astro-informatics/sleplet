from pathlib import Path

import numpy as np
import pyssht as ssht

from pys2sleplet.data.other.earth.create_earth_flm import create_flm
from pys2sleplet.utils.plot_methods import rotate_earth_to_south_america
from pys2sleplet.utils.vars import SAMPLING_SCHEME, SOUTH_AMERICA_RANGE

_file_location = Path(__file__).resolve()
_mask_path = _file_location.parents[3] / "data" / "slepian" / "masks"


def create_mask(L: int) -> None:
    """
    creates the South America region mask
    """
    earth_flm = create_flm(L)
    rot_flm = rotate_earth_to_south_america(earth_flm, L)
    earth_f = ssht.inverse(rot_flm, L, Reality=True, Method=SAMPLING_SCHEME)
    thetas, _ = ssht.sample_positions(L, Grid=True, Method=SAMPLING_SCHEME)
    mask = (thetas <= SOUTH_AMERICA_RANGE) & (earth_f >= 0)
    np.save(_mask_path / f"south_america_L{L}.npy", mask)


if __name__ == "__main__":
    create_mask(2048)
