import numpy as np
import pyssht as ssht

from pys2sleplet.flm.kernels.axisymmetric_wavelets import AxisymmetricWavelets
from pys2sleplet.flm.maps.earth import Earth
from pys2sleplet.plotting.create_plot import Plot
from pys2sleplet.utils.noise import compute_snr, signal_power
from pys2sleplet.utils.plot_methods import calc_plot_resolution

B = 2
J_MIN = 0
L = 128
N_SIGMA = 3
SNR_IN = 10


def main() -> None:
    """
    reproduce the denoising demo from s2let paper
    """
    # create Earth & noised Earth
    earth = Earth(L)
    earth_noised = Earth(L, noise=True)

    # compute noise sigma
    sigma_noise = np.sqrt(10 ** (-SNR_IN / 10) * signal_power(L, earth.multipole))

    # create wavelets
    aw = AxisymmetricWavelets(L, B=B, j_min=J_MIN)

    # compute wavelet coefficients
    w = np.zeros(aw.wavelets.shape, dtype=np.complex128)
    for ell in range(L):
        ind_m0 = ssht.elm2ind(ell, 0)
        wav_0 = np.sqrt((4 * np.pi) / (2 * ell + 1)) * aw.wavelets[:, ind_m0].conj()
        for m in range(-ell, ell + 1):
            ind = ssht.elm2ind(ell, m)
            w[:, ind] = wav_0 * earth_noised.multipole[ind]

    # compute sigma for wavelets
    sigma_j = np.apply_along_axis(
        lambda p: sigma_noise * L * np.sqrt(signal_power(L, p)), 1, aw.wavelets[1:]
    )

    # hard thresholding
    for j in range(1, aw.wavelets.shape[0]):
        f = ssht.inverse(w[j], L)
        cond = np.abs(f) < N_SIGMA * sigma_j[j - 1]
        f = np.where(cond, 0, f)
        w[j] = ssht.forward(f, L)

    # wavelet synthesis
    flm = np.zeros(L ** 2, dtype=np.complex128)
    for ell in range(L):
        ind_m0 = ssht.elm2ind(ell, 0)
        wav_0 = np.sqrt((4 * np.pi) / (2 * ell + 1)) * aw.wavelets[:, ind_m0]
        for m in range(-ell, ell + 1):
            ind = ssht.elm2ind(ell, m)
            flm[ind] = (w[:, ind] * wav_0).sum()

    # compute SNR
    compute_snr(L, earth.multipole, flm - earth.multipole)

    # make plot
    f = ssht.inverse(flm, L)
    resolution = calc_plot_resolution(L)
    name = f"earth_denoised_L{L}_res{resolution}"
    Plot(f, L, resolution, name).execute()


if __name__ == "__main__":
    main()
