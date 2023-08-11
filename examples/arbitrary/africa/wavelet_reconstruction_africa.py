import sleplet

B = 3
J_MIN = 2
L = 128
NORMALISE = False
SMOOTHING = 2


def main() -> None:
    """The reconstruction of a signal in Slepian space."""
    region = sleplet.slepian.Region(mask_name="africa")
    swc = sleplet.functions.SlepianWaveletCoefficientsAfrica(
        L,
        B=B,
        j_min=J_MIN,
        region=region,
        smoothing=SMOOTHING,
    )
    f_p = sleplet.wavelet_methods.slepian_wavelet_inverse(
        swc.wavelet_coefficients,
        swc.wavelets,
        swc.slepian.N,
    )

    # plot
    f = sleplet.slepian_methods.slepian_inverse(f_p, L, swc.slepian)
    name = f"africa_wavelet_reconstruction_L{L}"
    sleplet.plotting.PlotSphere(
        f,
        L,
        name,
        normalise=NORMALISE,
        region=swc.region,
    ).execute()


if __name__ == "__main__":
    main()
