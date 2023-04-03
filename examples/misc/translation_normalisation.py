import numpy as np
import pyssht as ssht
import seaborn as sns
from matplotlib import pyplot as plt

from sleplet.functions import HarmonicGaussian
from sleplet.plot_methods import save_plot

sns.set(context="paper")

ALPHA_DEFAULT = 0.75
L = 128
SAMPLING_SCHEME = "MWSS"


def compute_translation_normalisation_theta() -> None:
    """Analysis of the translation norm for referee."""
    hg = HarmonicGaussian(L)
    thetas, _ = ssht.sample_positions(L, Method=SAMPLING_SCHEME)
    norm = np.zeros(len(thetas))
    for i, theta in enumerate(thetas):
        print(f"compute norm {i+1}/{len(thetas)}")
        ylm_omega_prime = ssht.create_ylm(theta, ALPHA_DEFAULT, L).reshape(L**2)
        norm[i] = np.sqrt((np.abs(hg.coefficients * ylm_omega_prime) ** 2).sum())
    plt.plot(np.rad2deg(thetas), norm)
    plt.xlabel(r"$\theta/^\circ$")
    plt.ylabel(r"${\Vert f_{\ell m} Y_{\ell m}(\omega')\Vert}_{2}$")
    plt.xlim(0, 180)
    plt.ylim(0, np.ceil(norm.max()))
    save_plot(f"harmonic_gaussian_translation_normalisation_L{L}")


if __name__ == "__main__":
    compute_translation_normalisation_theta()
