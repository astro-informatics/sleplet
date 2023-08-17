import numpy as np

import sleplet

L = 16
NORMALISE = False
RANKS = 16
THETA_MAX = 40


def main() -> None:
    """
    Create fig 5.4 from Spatiospectral Concentration on a Sphere
    by Simons et al 2006.
    """
    region = sleplet.slepian.Region(theta_max=np.deg2rad(THETA_MAX))
    for r in range(RANKS):
        print(f"plotting rank={r}")
        slepian_function = sleplet.functions.Slepian(L, rank=r, region=region)
        f = sleplet.slepian_methods.slepian_inverse(
            slepian_function.coefficients,
            L,
            slepian_function.slepian,
        )
        sleplet.plotting.PlotSphere(
            f,
            L,
            slepian_function.name,
            normalise=NORMALISE,
            region=slepian_function.region,
        ).execute()


if __name__ == "__main__":
    main()
