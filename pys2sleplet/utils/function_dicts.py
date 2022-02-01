from glob import glob
from pathlib import Path

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
from pys2sleplet.meshes.harmonic_coefficients.mesh_basis_functions import (
    MeshBasisFunctions,
)
from pys2sleplet.meshes.harmonic_coefficients.mesh_field import MeshField
from pys2sleplet.meshes.harmonic_coefficients.mesh_noise_field import MeshNoiseField
from pys2sleplet.meshes.mesh_coefficients import MeshCoefficients
from pys2sleplet.meshes.slepian_coefficients.slepian_mesh_field import SlepianMeshField
from pys2sleplet.meshes.slepian_coefficients.slepian_mesh_functions import (
    SlepianMeshFunctions,
)
from pys2sleplet.meshes.slepian_coefficients.slepian_mesh_noise_field import (
    SlepianMeshNoiseField,
)
from pys2sleplet.meshes.slepian_coefficients.slepian_mesh_wavelet_coefficients import (
    SlepianMeshWaveletCoefficients,
)
from pys2sleplet.meshes.slepian_coefficients.slepian_mesh_wavelets import (
    SlepianMeshWavelets,
)

_file_location = Path(__file__).resolve()
_meshes_path = _file_location.parents[1] / "data" / "meshes"
MESHES: set[str] = {
    Path(x.removesuffix(".toml")).stem
    for x in glob(str(_meshes_path / "regions" / "*.toml"))
}

FLM: dict[str, Coefficients] = dict(
    axisymmetric_wavelet_coefficients_earth=AxisymmetricWaveletCoefficientsEarth,
    axisymmetric_wavelets=AxisymmetricWavelets,
    dirac_delta=DiracDelta,
    directional_spin_wavelets=DirectionalSpinWavelets,
    elongated_gaussian=ElongatedGaussian,
    earth=Earth,
    gaussian=Gaussian,
    harmonic_gaussian=HarmonicGaussian,
    identity=Identity,
    noise=NoiseEarth,
    ridgelets=Ridgelets,
    south_america=SouthAmerica,
    spherical_harmonic=SphericalHarmonic,
    squashed_gaussian=SquashedGaussian,
    wmap=Wmap,
)

FP: dict[str, Coefficients] = dict(
    slepian=Slepian,
    slepian_dirac_delta=SlepianDiracDelta,
    slepian_identity=SlepianIdentity,
    slepian_noise=SlepianNoiseSouthAmerica,
    slepian_south_america=SlepianSouthAmerica,
    slepian_wavelet_coefficients=SlepianWaveletCoefficientsSouthAmerica,
    slepian_wavelets=SlepianWavelets,
)

MAPS_LM: dict[str, Coefficients] = dict(
    earth=Earth, south_america=SouthAmerica, wmap=Wmap
)

MAPS_P: dict[str, Coefficients] = dict(slepian_south_america=SlepianSouthAmerica)

COEFFICIENTS: dict[str, Coefficients] = {**FLM, **FP}

MESH_HARMONIC: dict[str, MeshCoefficients] = dict(
    basis=MeshBasisFunctions, field=MeshField, noise=MeshNoiseField
)

MESH_SLEPIAN: dict[str, MeshCoefficients] = dict(
    coefficients=SlepianMeshWaveletCoefficients,
    slepian=SlepianMeshFunctions,
    slepian_field=SlepianMeshField,
    slepian_noise=SlepianMeshNoiseField,
    wavelets=SlepianMeshWavelets,
)

MESH_COEFFICIENTS: dict[str, MeshCoefficients] = {**MESH_HARMONIC, **MESH_SLEPIAN}
