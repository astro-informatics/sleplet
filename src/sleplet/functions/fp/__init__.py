"""classes of functions on the sphere created in Slepian space."""

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

__all__ = [
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
]
