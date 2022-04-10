from pathlib import Path

import seaborn as sns

from pys2sleplet.functions.fp.slepian_wavelets import SlepianWavelets
from pys2sleplet.utils.region import Region

file_location = Path(__file__).resolve()
fig_path = file_location.parents[2] / "figures"
sns.set(context="paper")

B = 3
J_MIN = 2
L = 128
STEP = 0.01


def main() -> None:
    """
    Plot a comparison of the absolute values of the wavelet coefficients
    compared to the Slepian coefficients. Expect the Slepian coefficients to
    decay faster than the wavelet coefficients.
    """
    # compute wavelets
    region = Region(mask_name="south_america")
    SlepianWavelets(L, B=B, j_min=J_MIN, region=region)


if __name__ == "__main__":
    main()
