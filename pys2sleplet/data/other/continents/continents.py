from pathlib import Path

import numpy as np
import pyssht as ssht
from plotly.graph_objs import Scatter3d
from plotly.graph_objs.scatter3d import Line

from pys2sleplet.utils.vars import (
    ANNOTATION_COLOUR,
    CONTINENT_ALPHA,
    CONTINENT_BETA,
    CONTINENT_GAMMA,
)

_file_location = Path(__file__).resolve()


def create_boundaries() -> Scatter3d:
    """
    creates continental boundaries of the Earth
    """
    # Recast data in good form
    cont = _load_data() / 100
    cont[cont == cont.max()] = np.nan

    # Convert to spherical coordinates
    lon = np.deg2rad(cont[:, 0])
    lat = np.deg2rad(cont[:, 1])
    x, y, z = ssht.s2_to_cart(lat, lon)

    # rotate to Earth orientation
    x, y, z = ssht.rot_cart_1d(
        x, y, z, [CONTINENT_ALPHA, CONTINENT_BETA, CONTINENT_GAMMA]
    )

    return Scatter3d(x=x, y=y, z=-z, mode="lines", line=Line(color=ANNOTATION_COLOUR))


def _load_data() -> np.ndarray:
    """
    load continental data from file
    """
    mtlfile = _file_location.parent / "cont.mtl"
    with open(mtlfile) as f:
        data = np.fromfile(f, dtype=">H").reshape((-1, 2), order="F")
    return data
