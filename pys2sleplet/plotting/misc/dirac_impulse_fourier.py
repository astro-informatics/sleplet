from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import signal

from pys2sleplet.utils.plot_methods import save_plot

file_location = Path(__file__).resolve()
fig_path = file_location.parents[2] / "figures"
sns.set(
    context="paper", rc={"font.size": 20, "axes.titlesize": 20, "axes.labelsize": 20}
)

LIM = 25


def main() -> None:
    """
    plots a Dirac impulse and it's Fourier transform
    """
    impulse = signal.unit_impulse(2 * LIM, "mid")
    _plot_impulse(impulse)
    _plot_fft(impulse)


def _plot_impulse(impulse: np.ndarray) -> None:
    """
    plots the Dirac impulse
    """
    plt.figure()
    plt.plot(impulse)
    plt.xticks([])
    plt.xlabel(r"$t$")
    save_plot(fig_path, "dirac_impulse")


def _plot_fft(impulse: np.ndarray) -> None:
    """
    plots resultant Fourier transform
    """
    plt.figure()
    fourier = np.fft.fft(impulse)
    plt.plot(fourier.real)
    plt.xticks([])
    plt.xlabel(r"$\varpi$")
    save_plot(fig_path, "dirac_fft")


if __name__ == "__main__":
    main()
