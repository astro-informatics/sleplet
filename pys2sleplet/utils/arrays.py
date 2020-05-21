import pyssht as ssht

from pys2sleplet.utils.config import config
from pys2sleplet.utils.vars import SAMPLING_SCHEME

THETA_SAMPLES, PHI_SAMPLES = ssht.sample_positions(config.L, Method=SAMPLING_SCHEME)
THETA_GRID, PHI_GRID = ssht.sample_positions(
    config.L, Grid=True, Method=SAMPLING_SCHEME
)
