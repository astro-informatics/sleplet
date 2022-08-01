from glob import glob
from pathlib import Path

from pys2sleplet.functions.coefficients import Coefficients
from pys2sleplet.functions.flm.africa import Africa
from pys2sleplet.functions.flm.axisymmetric_wavelet_coefficients_earth import (
    AxisymmetricWaveletCoefficientsEarth,
)
from pys2sleplet.functions.flm.axisymmetric_wavelet_coefficients_south_america import (
    AxisymmetricWaveletCoefficientsSouthAmerica,
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
from pys2sleplet.meshes.harmonic_coefficients.mesh_basis_functions import (
    MeshBasisFunctions,
)
from pys2sleplet.meshes.harmonic_coefficients.mesh_field import MeshField
from pys2sleplet.meshes.harmonic_coefficients.mesh_noise_field import MeshNoiseField
from pys2sleplet.meshes.mesh_coefficients import MeshCoefficients
from pys2sleplet.meshes.slepian_coefficients.mesh_slepian_field import MeshSlepianField
from pys2sleplet.meshes.slepian_coefficients.mesh_slepian_functions import (
    MeshSlepianFunctions,
)
from pys2sleplet.meshes.slepian_coefficients.mesh_slepian_noise_field import (
    MeshSlepianNoiseField,
)
from pys2sleplet.meshes.slepian_coefficients.mesh_slepian_wavelet_coefficients import (
    MeshSlepianWaveletCoefficients,
)
from pys2sleplet.meshes.slepian_coefficients.mesh_slepian_wavelets import (
    MeshSlepianWavelets,
)

_file_location = Path(__file__).resolve()
_meshes_path = _file_location.parents[1] / "data" / "meshes"
MESHES: list[str] = [
    Path(x.removesuffix(".toml")).stem
    for x in glob(str(_meshes_path / "regions" / "*.toml"))
]

FLM: list[Coefficients] = [
    Africa,
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

FP: list[Coefficients] = [
    Slepian,
    SlepianDiracDelta,
    SlepianIdentity,
    SlepianNoiseSouthAmerica,
    SlepianSouthAmerica,
    SlepianWaveletCoefficientsSouthAmerica,
    SlepianWavelets,
]

COEFFICIENTS: list[Coefficients] = FLM + FP

MAPS_LM: list[Coefficients] = [Earth, SouthAmerica, Wmap]

MESH_HARMONIC: list[MeshCoefficients] = [MeshBasisFunctions, MeshField, MeshNoiseField]

MESH_SLEPIAN: list[MeshCoefficients] = [
    MeshSlepianField,
    MeshSlepianFunctions,
    MeshSlepianNoiseField,
    MeshSlepianWaveletCoefficients,
    MeshSlepianWavelets,
]

MESH_COEFFICIENTS: list[MeshCoefficients] = MESH_HARMONIC + MESH_SLEPIAN
