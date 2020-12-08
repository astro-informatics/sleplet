from pys2sleplet.plotting.create_plot import Plot
from pys2sleplet.utils.denoising import denoising_slepian
from pys2sleplet.utils.region import Region

B = 3
J_MIN = 2
L = 128
N_SIGMA = 2
SNR_IN = 1


def main() -> None:
    """
    denoising demo using Slepian wavelets
    """
    fun = "south_america"
    region = Region(mask_name=fun)
    f, annotations = denoising_slepian(
        f"slepian_{fun}", L, B, J_MIN, N_SIGMA, region, SNR_IN
    )
    name = f"{fun}_denoised_slepian_L{L}"
    Plot(f, L, name, annotations=annotations).execute()


if __name__ == "__main__":
    main()
