from glob import glob
from pathlib import Path
from typing import TYPE_CHECKING

from sleplet.functions.flm.africa import Africa
from sleplet.functions.flm.axisymmetric_wavelet_coefficients_africa import (
    AxisymmetricWaveletCoefficientsAfrica,
)
from sleplet.functions.flm.axisymmetric_wavelet_coefficients_earth import (
    AxisymmetricWaveletCoefficientsEarth,
)
from sleplet.functions.flm.axisymmetric_wavelet_coefficients_south_america import (
    AxisymmetricWaveletCoefficientsSouthAmerica,
)
from sleplet.functions.flm.axisymmetric_wavelets import AxisymmetricWavelets
from sleplet.functions.flm.dirac_delta import DiracDelta
from sleplet.functions.flm.directional_spin_wavelets import DirectionalSpinWavelets
from sleplet.functions.flm.earth import Earth
from sleplet.functions.flm.elongated_gaussian import ElongatedGaussian
from sleplet.functions.flm.gaussian import Gaussian
from sleplet.functions.flm.harmonic_gaussian import HarmonicGaussian
from sleplet.functions.flm.identity import Identity
from sleplet.functions.flm.noise_earth import NoiseEarth
from sleplet.functions.flm.ridgelets import Ridgelets
from sleplet.functions.flm.south_america import SouthAmerica
from sleplet.functions.flm.spherical_harmonic import SphericalHarmonic
from sleplet.functions.flm.squashed_gaussian import SquashedGaussian
from sleplet.functions.flm.wmap import Wmap
from sleplet.functions.fp.slepian import Slepian
from sleplet.functions.fp.slepian_africa import SlepianAfrica
from sleplet.functions.fp.slepian_dirac_delta import SlepianDiracDelta
from sleplet.functions.fp.slepian_identity import SlepianIdentity
from sleplet.functions.fp.slepian_noise_africa import SlepianNoiseAfrica
from sleplet.functions.fp.slepian_noise_south_america import SlepianNoiseSouthAmerica
from sleplet.functions.fp.slepian_south_america import SlepianSouthAmerica
from sleplet.functions.fp.slepian_wavelet_coefficients_africa import (
    SlepianWaveletCoefficientsAfrica,
)
from sleplet.functions.fp.slepian_wavelet_coefficients_south_america import (
    SlepianWaveletCoefficientsSouthAmerica,
)
from sleplet.functions.fp.slepian_wavelets import SlepianWavelets
from sleplet.meshes.harmonic_coefficients.mesh_basis_functions import MeshBasisFunctions
from sleplet.meshes.harmonic_coefficients.mesh_field import MeshField
from sleplet.meshes.harmonic_coefficients.mesh_noise_field import MeshNoiseField
from sleplet.meshes.slepian_coefficients.mesh_slepian_field import MeshSlepianField
from sleplet.meshes.slepian_coefficients.mesh_slepian_functions import (
    MeshSlepianFunctions,
)
from sleplet.meshes.slepian_coefficients.mesh_slepian_noise_field import (
    MeshSlepianNoiseField,
)
from sleplet.meshes.slepian_coefficients.mesh_slepian_wavelet_coefficients import (
    MeshSlepianWaveletCoefficients,
)
from sleplet.meshes.slepian_coefficients.mesh_slepian_wavelets import (
    MeshSlepianWavelets,
)

if TYPE_CHECKING:
    from sleplet.functions.coefficients import Coefficients
    from sleplet.meshes.mesh_coefficients import MeshCoefficients

_data_path = Path(__file__).resolve().parents[1] / "data"
MESHES: list[str] = [
    Path(x).stem.removeprefix("meshes_regions_")
    for x in glob(str(_data_path / "*.toml"))
]

FLM: list[type[Coefficients]] = [
    Africa,
    AxisymmetricWaveletCoefficientsAfrica,
    AxisymmetricWaveletCoefficientsEarth,
    AxisymmetricWaveletCoefficientsSouthAmerica,
    AxisymmetricWavelets,
    DiracDelta,
    DirectionalSpinWavelets,
    Earth,
    ElongatedGaussian,
    Gaussian,
    HarmonicGaussian,
    Identity,
    NoiseEarth,
    Ridgelets,
    SouthAmerica,
    SphericalHarmonic,
    SquashedGaussian,
    Wmap,
]

FP: list[type[Coefficients]] = [
    Slepian,
    SlepianAfrica,
    SlepianDiracDelta,
    SlepianIdentity,
    SlepianNoiseAfrica,
    SlepianNoiseSouthAmerica,
    SlepianSouthAmerica,
    SlepianWaveletCoefficientsAfrica,
    SlepianWaveletCoefficientsSouthAmerica,
    SlepianWavelets,
]

COEFFICIENTS: list[type[Coefficients]] = FLM + FP

MAPS_LM: list[type[Coefficients]] = [Earth, SouthAmerica, Wmap]

MESH_HARMONIC: list[type[MeshCoefficients]] = [
    MeshBasisFunctions,
    MeshField,
    MeshNoiseField,
]

MESH_SLEPIAN: list[type[MeshCoefficients]] = [
    MeshSlepianField,
    MeshSlepianFunctions,
    MeshSlepianNoiseField,
    MeshSlepianWaveletCoefficients,
    MeshSlepianWavelets,
]

MESH_COEFFICIENTS: list[type[MeshCoefficients]] = MESH_HARMONIC + MESH_SLEPIAN
