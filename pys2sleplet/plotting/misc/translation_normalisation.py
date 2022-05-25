from pathlib import Path

import numpy as np
import pyssht as ssht
import seaborn as sns
from matplotlib import pyplot as plt

from pys2sleplet.functions.flm.harmonic_gaussian import HarmonicGaussian
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.plot_methods import save_plot
from pys2sleplet.utils.vars import ALPHA_DEFAULT, SAMPLING_SCHEME

file_location = Path(__file__).resolve()
fig_path = file_location.parents[2] / "figures"
sns.set(context="paper")

L = 128


def compute_translation_normalisation_theta() -> None:
    """
    analysis of the translation norm for referee
    """
    hg = HarmonicGaussian(L)
    thetas, _ = ssht.sample_positions(L, Method=SAMPLING_SCHEME)
    norm = np.zeros(len(thetas))
    for i, theta in enumerate(thetas):
        logger.info(f"compute norm {i+1}/{len(thetas)}")
        ylm_omega_prime = ssht.create_ylm(theta, ALPHA_DEFAULT, L).reshape(L**2)
        norm[i] = np.sqrt((np.abs(hg.coefficients * ylm_omega_prime) ** 2).sum())
    plt.plot(np.rad2deg(thetas), norm)
    plt.xlabel(r"$\theta/^\circ$")
    plt.ylabel(r"${\Vert f_{\ell m} Y_{\ell m}(\omega')\Vert}_{2}$")
    plt.xlim(0, 180)
    plt.ylim(0, np.ceil(norm.max()))
    save_plot(fig_path, f"harmonic_gaussian_translation_normalisation_L{L}")


if __name__ == "__main__":
    compute_translation_normalisation_theta()
