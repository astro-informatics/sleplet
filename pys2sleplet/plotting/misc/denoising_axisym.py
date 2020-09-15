import numpy as np
import pyssht as ssht

from pys2sleplet.flm.kernels.axisymmetric_wavelets import AxisymmetricWavelets
from pys2sleplet.flm.maps.earth import Earth
from pys2sleplet.plotting.create_plot import Plot
from pys2sleplet.utils.noise import compute_sigma_j, compute_snr, hard_thresholding
from pys2sleplet.utils.plot_methods import calc_plot_resolution

B = 2
J_MIN = 0
L = 128
N_SIGMA = 3


def main() -> None:
    """
    reproduce the denoising demo from s2let paper
    """
    # create Earth & noised Earth
    earth = Earth(L)
    earth_noised = Earth(L, noise=True)

    # create wavelets
    aw = AxisymmetricWavelets(L, B=B, j_min=J_MIN)

    # compute wavelet noise
    sigma_j = compute_sigma_j(L, earth.multipole, aw.wavelets[1:])

    # compute wavelet coefficients
    w = np.zeros(aw.wavelets.shape, dtype=np.complex128)
    for ell in range(L):
        ind_m0 = ssht.elm2ind(ell, 0)
        wav_0 = np.sqrt((4 * np.pi) / (2 * ell + 1)) * aw.wavelets[:, ind_m0].conj()
        for m in range(-ell, ell + 1):
            ind = ssht.elm2ind(ell, m)
            w[:, ind] = wav_0 * earth_noised.multipole[ind]

    # hard thresholding
    w_denoised = hard_thresholding(L, w, sigma_j, N_SIGMA)

    # wavelet synthesis
    flm = np.zeros(L ** 2, dtype=np.complex128)
    for ell in range(L):
        ind_m0 = ssht.elm2ind(ell, 0)
        wav_0 = np.sqrt((4 * np.pi) / (2 * ell + 1)) * aw.wavelets[:, ind_m0]
        for m in range(-ell, ell + 1):
            ind = ssht.elm2ind(ell, m)
            flm[ind] = (w_denoised[:, ind] * wav_0).sum()

    # compute SNR
    compute_snr(L, earth.multipole, flm - earth.multipole)

    # make plot
    f = ssht.inverse(flm, L)
    resolution = calc_plot_resolution(L)
    name = f"earth_denoised_L{L}_res{resolution}"
    Plot(f, L, resolution, name).execute()


if __name__ == "__main__":
    main()
