import sleplet

L = 128
NORMALISE = False
SMOOTHING = 2


def main() -> None:
    """Reconstruct a signal in Slepian space."""
    region = sleplet.slepian.Region(mask_name="africa")
    slepian = sleplet.slepian_methods.choose_slepian_method(L, region)
    africa = sleplet.functions.SlepianAfrica(L, region=region, smoothing=SMOOTHING)

    # perform reconstruction
    f = sleplet.slepian_methods.slepian_inverse(africa.coefficients, L, slepian)

    # plot
    name = f"africa_slepian_reconstruction_L{L}"
    sleplet.plotting.PlotSphere(
        f,
        L,
        name,
        normalise=NORMALISE,
        region=slepian.region,
    ).execute()


if __name__ == "__main__":
    main()
