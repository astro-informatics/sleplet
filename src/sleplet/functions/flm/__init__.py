"""Classes of functions on the sphere created in harmonic space."""

from .africa import Africa
from .axisymmetric_wavelet_coefficients_africa import (
    AxisymmetricWaveletCoefficientsAfrica,
)
from .axisymmetric_wavelet_coefficients_earth import (
    AxisymmetricWaveletCoefficientsEarth,
)
from .axisymmetric_wavelet_coefficients_south_america import (
    AxisymmetricWaveletCoefficientsSouthAmerica,
)
from .axisymmetric_wavelets import AxisymmetricWavelets
from .dirac_delta import DiracDelta
from .directional_spin_wavelets import DirectionalSpinWavelets
from .earth import Earth
from .elongated_gaussian import ElongatedGaussian
from .gaussian import Gaussian
from .harmonic_gaussian import HarmonicGaussian
from .identity import Identity
from .noise_earth import NoiseEarth
from .ridgelets import Ridgelets
from .south_america import SouthAmerica
from .spherical_harmonic import SphericalHarmonic
from .squashed_gaussian import SquashedGaussian
from .wmap import Wmap

__all__ = [
    "Africa",
    "AxisymmetricWaveletCoefficientsAfrica",
    "AxisymmetricWaveletCoefficientsEarth",
    "AxisymmetricWaveletCoefficientsSouthAmerica",
    "AxisymmetricWavelets",
    "DiracDelta",
    "DirectionalSpinWavelets",
    "Earth",
    "ElongatedGaussian",
    "Gaussian",
    "HarmonicGaussian",
    "Identity",
    "NoiseEarth",
    "Ridgelets",
    "SouthAmerica",
    "SphericalHarmonic",
    "SquashedGaussian",
    "Wmap",
]
