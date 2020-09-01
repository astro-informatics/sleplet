import numpy as np
import pyssht as ssht

from pys2sleplet.flm.maps.earth import Earth
from pys2sleplet.plotting.create_plot import Plot
from pys2sleplet.utils.noise import compute_snr, signal_power
from pys2sleplet.utils.plot_methods import calc_plot_resolution
from pys2sleplet.utils.pys2let import s2let

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

    # create generating functions
    kappa0, kappa = s2let.axisym_wav_l(B, L, J_MIN)

    # compute noise sigma
    sigma_noise = np.sqrt(10 ** (-SNR_IN / 10) * signal_power(L, earth.multipole))

    # compute tiling functions
    phi = np.zeros(kappa0.shape)
    psi = np.zeros(kappa.shape)
    for ell in range(L):
        factor = np.sqrt((2 * ell + 1) / (4 * np.pi))
        phi[ell] = factor * kappa0[ell]
        psi[ell] = factor * kappa[ell]

    # compute wavelet coefficients
    w_phi = np.zeros(L ** 2, dtype=np.complex128)
    w_psi = np.zeros((L ** 2, kappa.shape[1]), dtype=np.complex128)
    for ell in range(L):
        factor = np.sqrt((4 * np.pi) / (2 * ell + 1))
        scal_0 = factor * phi[ell].conj()
        wav_0 = factor * psi[ell].conj()
        for m in range(-ell, ell + 1):
            ind = ssht.elm2ind(ell, m)
            w_phi[ind] = scal_0 * earth_noised.multipole[ind]
            w_psi[ind] = wav_0 * earth_noised.multipole[ind]

    # compute sigma for wavelets
    sigma_j = np.apply_along_axis(
        lambda p: sigma_noise * L * np.sqrt(signal_power(L, p)), 0, psi
    )

    # hard thresholding
    for j in range(kappa.shape[1]):
        f = ssht.inverse(np.ascontiguousarray(w_psi[:, j]), L)
        cond = np.abs(f) < N_SIGMA * sigma_j[j]
        f = np.where(cond, 0, f)
        w_psi[:, j] = ssht.forward(f, L)

    # wavelet synthesis
    flm = np.zeros(L ** 2, dtype=np.complex128)
    for ell in range(L):
        factor = np.sqrt((4 * np.pi) / (2 * ell + 1))
        scal_0 = factor * phi[ell]
        wav_0 = factor * psi[ell]
        for m in range(-ell, ell + 1):
            ind = ssht.elm2ind(ell, m)
            flm[ind] = w_phi[ind] * scal_0 + (w_psi[ind] * wav_0).sum()

    # compute SNR
    compute_snr(L, earth.multipole, flm - earth.multipole)

    # make plot
    f = ssht.inverse(flm, L)
    resolution = calc_plot_resolution(L)
    name = f"earth_denoised_L{L}_res{resolution}"
    Plot(f, L, resolution, name).execute()


if __name__ == "__main__":
    main()
