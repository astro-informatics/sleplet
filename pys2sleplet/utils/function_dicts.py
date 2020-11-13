from typing import Dict

from pys2sleplet.functions.coefficients import Coefficients
from pys2sleplet.functions.flm.axisymmetric_wavelet_coefficients_earth import (
    AxisymmetricWaveletCoefficientsEarth,
)
from pys2sleplet.functions.flm.axisymmetric_wavelets import AxisymmetricWavelets
from pys2sleplet.functions.flm.dirac_delta import DiracDelta
from pys2sleplet.functions.flm.directional_spin_wavelets import DirectionalSpinWavelets
from pys2sleplet.functions.flm.earth import Earth
from pys2sleplet.functions.flm.elongated_gaussian import ElongatedGaussian
from pys2sleplet.functions.flm.gaussian import Gaussian
from pys2sleplet.functions.flm.harmonic_gaussian import HarmonicGaussian
from pys2sleplet.functions.flm.identity import Identity
from pys2sleplet.functions.flm.noise_earth import NoiseEarth
from pys2sleplet.functions.flm.ridgelets import Ridgelets
from pys2sleplet.functions.flm.south_america import SouthAmerica
from pys2sleplet.functions.flm.spherical_harmonic import SphericalHarmonic
from pys2sleplet.functions.flm.squashed_gaussian import SquashedGaussian
from pys2sleplet.functions.flm.wmap import Wmap
from pys2sleplet.functions.fp.slepian import Slepian
from pys2sleplet.functions.fp.slepian_dirac_delta import SlepianDiracDelta
from pys2sleplet.functions.fp.slepian_identity import SlepianIdentity
from pys2sleplet.functions.fp.slepian_noise_south_america import (
    SlepianNoiseSouthAmerica,
)
from pys2sleplet.functions.fp.slepian_south_america import SlepianSouthAmerica
from pys2sleplet.functions.fp.slepian_wavelet_coefficients_south_america import (
    SlepianWaveletCoefficientsSouthAmerica,
)
from pys2sleplet.functions.fp.slepian_wavelets import SlepianWavelets

FLM: Dict[str, Coefficients] = dict(
    axisymmetric_wavelet_coefficients_earth=AxisymmetricWaveletCoefficientsEarth,
    axisymmetric_wavelets=AxisymmetricWavelets,
    dirac_delta=DiracDelta,
    directional_spin_wavelets=DirectionalSpinWavelets,
    elongated_gaussian=ElongatedGaussian,
    earth=Earth,
    gaussian=Gaussian,
    harmonic_gaussian=HarmonicGaussian,
    identity=Identity,
    noise_earth=NoiseEarth,
    ridgelets=Ridgelets,
    south_america=SouthAmerica,
    spherical_harmonic=SphericalHarmonic,
    squashed_gaussian=SquashedGaussian,
    wmap=Wmap,
)

FP: Dict[str, Coefficients] = dict(
    slepian=Slepian,
    slepian_dirac_delta=SlepianDiracDelta,
    slepian_identity=SlepianIdentity,
    slepian_noise_south_america=SlepianNoiseSouthAmerica,
    slepian_south_america=SlepianSouthAmerica,
    slepian_wavelet_coefficients_south_america=SlepianWaveletCoefficientsSouthAmerica,
    slepian_wavelets=SlepianWavelets,
)

MAPS: Dict[str, Coefficients] = dict(earth=Earth, south_america=SouthAmerica, wmap=Wmap)

FUNCTIONS: Dict[str, Coefficients] = {**FLM, **FP}
