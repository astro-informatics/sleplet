from pathlib import Path

import numpy as np
import pyssht as ssht

from sleplet.data.other.earth.create_earth_flm import create_flm
from sleplet.utils.plot_methods import rotate_earth_to_south_america
from sleplet.utils.vars import SAMPLING_SCHEME, SOUTH_AMERICA_RANGE

_data_path = Path(__file__).resolve().parents[3] / "data"


def create_mask(L: int) -> None:
    """
    creates the South America region mask
    """
    earth_flm = create_flm(L)
    rot_flm = rotate_earth_to_south_america(earth_flm, L)
    earth_f = ssht.inverse(rot_flm, L, Reality=True, Method=SAMPLING_SCHEME)
    thetas, _ = ssht.sample_positions(L, Grid=True, Method=SAMPLING_SCHEME)
    mask = (thetas <= SOUTH_AMERICA_RANGE) & (earth_f >= 0)
    np.save(_data_path / f"slepian_masks_south_america_L{L}.npy", mask)


if __name__ == "__main__":
    create_mask(2048)
