from pathlib import Path

import numpy as np
import pyssht as ssht
import seaborn as sns

from pys2sleplet.functions.flm.earth import Earth
from pys2sleplet.plotting.create_plot import Plot
from pys2sleplet.utils.vars import EARTH_ALPHA, EARTH_BETA, EARTH_GAMMA, SAMPLING_SCHEME

file_location = Path(__file__).resolve()
fig_path = file_location.parents[2] / "figures"
mask_path = file_location.parents[2] / "data" / "slepian" / "masks"
sns.set(context="paper")

L = 128
SMOOTHING = 1_000


def plot_mask() -> None:
    """
    plots masks on the sphere without any harmonic transforms
    """
    earth_smoothed = Earth(L, smoothing=SMOOTHING)
    rot_flm = ssht.rotate_flms(
        earth_smoothed.coefficients, EARTH_ALPHA, EARTH_BETA, EARTH_GAMMA, L
    )
    field = ssht.inverse(
        rot_flm, L, Reality=earth_smoothed.reality, Method=SAMPLING_SCHEME
    )
    mask = np.load(mask_path / f"south_america_L{L}.npy").astype(np.complex_)
    south_america_smoothed = np.where(mask, field, 0)
    Plot(south_america_smoothed, L, f"south_america_smoothed_L{L}").execute()


if __name__ == "__main__":
    plot_mask()
