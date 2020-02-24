from typing import Mapping, Type

from flm.functions import Functions
from flm.kernels.dirac_delta import DiracDelta
from flm.kernels.elongated_gaussian import ElongatedGaussian
from flm.kernels.gaussian import Gaussian
from flm.kernels.harmonic_gaussian import HarmonicGaussian
from flm.kernels.identity import Identity
from flm.kernels.slepian import Slepian
from flm.kernels.spherical_harmonic import SphericalHarmonic
from flm.kernels.squashed_gaussian import SquashedGaussian
from flm.maps.earth import Earth
from flm.maps.wmap import WMAP

function_dict: Mapping[str, Type[Functions]] = {
    "dirac_delta": DiracDelta,
    "earth": Earth,
    "elongated_gaussian": ElongatedGaussian,
    "gaussian": Gaussian,
    "harmonic_gaussian": HarmonicGaussian,
    "identity": Identity,
    "slepian": Slepian,
    "spherical_harmonic": SphericalHarmonic,
    "squashed_gaussian": SquashedGaussian,
    "wmap": WMAP,
}
