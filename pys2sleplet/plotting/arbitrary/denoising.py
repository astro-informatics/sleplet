import pyssht as ssht

from pys2sleplet.flm.kernels.slepian_wavelets import SlepianWavelets
from pys2sleplet.flm.maps.south_america import SouthAmerica
from pys2sleplet.plotting.create_plot import Plot
from pys2sleplet.utils.noise import compute_sigma_j, compute_snr, hard_thresholding
from pys2sleplet.utils.plot_methods import calc_plot_resolution
from pys2sleplet.utils.region import Region
from pys2sleplet.utils.wavelet_methods import wavelet_forward, wavelet_inverse

B = 2
J_MIN = 0
L = 64
N_SIGMA = 15


def main() -> None:
    """
    denoising demo using Slepian wavelets
    """
    # create South America & noised South America
    sa = SouthAmerica(L)
    sa_noised = SouthAmerica(L, noise=True)

    # create wavelets
    region = Region(mask_name="south_america")
    sw = SlepianWavelets(L, B=B, j_min=J_MIN, region=region)

    # compute wavelet noise
    sigma_j = compute_sigma_j(L, sa.multipole, sw.wavelets[1:])

    # compute wavelet coefficients
    w = wavelet_forward(sa_noised, sw.wavelets)

    # hard thresholding
    w_denoised = hard_thresholding(L, w, sigma_j, N_SIGMA)

    # wavelet synthesis
    flm = wavelet_inverse(w_denoised, sw.wavelets)

    # compute SNR
    compute_snr(L, sa.multipole, sa_noised.multipole - sa.multipole, region=region)
    compute_snr(L, sa.multipole, flm - sa.multipole, region=region)

    # make plot
    f = ssht.inverse(flm, L)
    resolution = calc_plot_resolution(L)
    name = f"south_america_denoised_L{L}_res{resolution}"
    Plot(f, L, resolution, name).execute()


if __name__ == "__main__":
    main()
