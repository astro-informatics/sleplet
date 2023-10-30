import numpy as np

import sleplet

B = 3
J_MIN = 2
L = 128
NORMALISE = False
SMOOTHING = 2


def main() -> None:
    """Reconstruct a signal in Slepian space."""
    region = sleplet.slepian.Region(mask_name="africa")
    swc = sleplet.functions.SlepianWaveletCoefficientsAfrica(
        L,
        B=B,
        j_min=J_MIN,
        region=region,
        smoothing=SMOOTHING,
    )

    # plot
    f_p = np.zeros(swc.slepian.N, dtype=np.complex_)
    for p, coeff in enumerate(swc.wavelet_coefficients):
        print(f"plot reconstruction: {p}")
        f_p += sleplet.wavelet_methods.slepian_wavelet_inverse(
            coeff,
            swc.wavelets,
            swc.slepian.N,
        )
        f = sleplet.slepian_methods.slepian_inverse(f_p, L, swc.slepian)
        name = f"africa_wavelet_reconstruction_progressive_{p}_L{L}"
        sleplet.plotting.PlotSphere(
            f,
            L,
            name,
            normalise=NORMALISE,
            region=swc.region,
        ).execute()


if __name__ == "__main__":
    main()
