import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from numpy import typing as npt

from sleplet.plot_methods import save_plot

sns.set(context="paper")

DELTA_T = 0.001
FREQUENCIES = [5, 10, 15]
LENGTH = 0.512


def main() -> None:
    """Plots a Dirac impulse and it's Fourier transform."""
    size = int(LENGTH / DELTA_T)
    amplitude = np.zeros(size * len(FREQUENCIES))
    for c, f in enumerate(FREQUENCIES):
        amplitude[size * c : size * (c + 1)] = _ricker(f)
    plt.plot(amplitude)
    plt.xticks([])
    plt.xlabel(r"$t$")
    save_plot("ricker_wavelets")


def _ricker(freq: float) -> npt.NDArray[np.float_]:
    """Creates a Ricker wavelet."""
    t = np.arange(-LENGTH / 2, (LENGTH - DELTA_T) / 2, DELTA_T)
    return (1.0 - 2.0 * (np.pi**2) * (freq**2) * (t**2)) * np.exp(
        -(np.pi**2) * (freq**2) * (t**2),
    )


if __name__ == "__main__":
    main()
