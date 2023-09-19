"""Set of classes to create functions on the sphere."""

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
from .earth import Earth
from .elongated_gaussian import ElongatedGaussian
from .gaussian import Gaussian
from .harmonic_gaussian import HarmonicGaussian
from .identity import Identity
from .noise_earth import NoiseEarth
from .ridgelets import Ridgelets
from .slepian import Slepian
from .slepian_africa import SlepianAfrica
from .slepian_dirac_delta import SlepianDiracDelta
from .slepian_identity import SlepianIdentity
from .slepian_noise_africa import SlepianNoiseAfrica
from .slepian_noise_south_america import SlepianNoiseSouthAmerica
from .slepian_south_america import SlepianSouthAmerica
from .slepian_wavelet_coefficients_africa import SlepianWaveletCoefficientsAfrica
from .slepian_wavelet_coefficients_south_america import (
    SlepianWaveletCoefficientsSouthAmerica,
)
from .slepian_wavelets import SlepianWavelets
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
    "Earth",
    "ElongatedGaussian",
    "Gaussian",
    "HarmonicGaussian",
    "Identity",
    "NoiseEarth",
    "Ridgelets",
    "Slepian",
    "SlepianAfrica",
    "SlepianDiracDelta",
    "SlepianIdentity",
    "SlepianNoiseAfrica",
    "SlepianNoiseSouthAmerica",
    "SlepianSouthAmerica",
    "SlepianWaveletCoefficientsAfrica",
    "SlepianWaveletCoefficientsSouthAmerica",
    "SlepianWavelets",
    "SouthAmerica",
    "SphericalHarmonic",
    "SquashedGaussian",
    "Wmap",
]
