from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.special import spherical_jn

from pys2sleplet.utils.plot_methods import save_plot

file_location = Path(__file__).resolve()
fig_path = file_location.parents[2] / "figures"
sns.set(context="paper")

ELL = 20
LENGTH = 100
STEP = 0.1


def main() -> None:
    """
    plots an example spherical bessel function
    """
    x = np.arange(0, LENGTH, STEP)
    j = spherical_jn(ELL, x)
    plt.plot(x, j)
    plt.xticks([0, ELL], [0, r"$\ell$"])
    plt.yticks([0, 0])
    plt.xlabel(r"$kr$")
    plt.show()
    save_plot(fig_path, "spherical_bessel")


if __name__ == "__main__":
    main()
