from .dirac_delta import DiracDelta
from .elongated_gaussian import ElongatedGaussian
from .gaussian import Gaussian
from .harmonic_gaussian import HarmonicGaussian
from .identity import Identity
from .slepian import Slepian
from .spherical_harmonic import SphericalHarmonic
from .squashed_gaussian import SquashedGaussian


def kernels():
    return {
        "dirac_delta": DiracDelta,
        "elongated_gaussian": ElongatedGaussian,
        "gaussian": Gaussian,
        "harmonic_gaussian": HarmonicGaussian,
        "identity": Identity,
        "slepian": Slepian,
        "spherical_harmonic": SphericalHarmonic,
        "squashed_gaussian": SquashedGaussian,
    }
