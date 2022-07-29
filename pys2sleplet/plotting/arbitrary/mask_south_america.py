from pathlib import Path

import numpy as np

from pys2sleplet.plotting.create_plot_sphere import Plot

file_location = Path(__file__).resolve()
fig_path = file_location.parents[2] / "figures"
mask_path = file_location.parents[2] / "data" / "slepian" / "masks"

L = 128


def plot_mask() -> None:
    """
    plots masks on the sphere without any harmonic transforms
    """
    f = np.load(mask_path / f"south_america_L{L}.npy").astype(np.complex_)
    Plot(f, L, f"mask_south_america_L{L}").execute()


if __name__ == "__main__":
    plot_mask()
