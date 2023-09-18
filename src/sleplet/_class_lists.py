import pathlib

import sleplet.functions.africa
import sleplet.functions.axisymmetric_wavelet_coefficients_africa
import sleplet.functions.axisymmetric_wavelet_coefficients_earth
import sleplet.functions.axisymmetric_wavelet_coefficients_south_america
import sleplet.functions.axisymmetric_wavelets
import sleplet.functions.coefficients
import sleplet.functions.dirac_delta
import sleplet.functions.earth
import sleplet.functions.elongated_gaussian
import sleplet.functions.gaussian
import sleplet.functions.harmonic_gaussian
import sleplet.functions.identity
import sleplet.functions.noise_earth
import sleplet.functions.ridgelets
import sleplet.functions.slepian
import sleplet.functions.slepian_africa
import sleplet.functions.slepian_dirac_delta
import sleplet.functions.slepian_identity
import sleplet.functions.slepian_noise_africa
import sleplet.functions.slepian_noise_south_america
import sleplet.functions.slepian_south_america
import sleplet.functions.slepian_wavelet_coefficients_africa
import sleplet.functions.slepian_wavelet_coefficients_south_america
import sleplet.functions.slepian_wavelets
import sleplet.functions.south_america
import sleplet.functions.spherical_harmonic
import sleplet.functions.squashed_gaussian
import sleplet.functions.wmap
import sleplet.meshes.mesh_basis_functions
import sleplet.meshes.mesh_coefficients
import sleplet.meshes.mesh_field
import sleplet.meshes.mesh_noise_field
import sleplet.meshes.mesh_slepian_field
import sleplet.meshes.mesh_slepian_functions
import sleplet.meshes.mesh_slepian_noise_field
import sleplet.meshes.mesh_slepian_wavelet_coefficients
import sleplet.meshes.mesh_slepian_wavelets

_data_path = pathlib.Path(__file__).resolve().parent / "_data"
MESHES: list[str] = [
    pathlib.Path(x).stem.removeprefix("meshes_regions_")
    for x in pathlib.Path(_data_path).glob("*.toml")
]

FLM: list[type[sleplet.functions.coefficients.Coefficients]] = [
    sleplet.functions.africa.Africa,
    sleplet.functions.axisymmetric_wavelet_coefficients_africa.AxisymmetricWaveletCoefficientsAfrica,
    sleplet.functions.axisymmetric_wavelet_coefficients_earth.AxisymmetricWaveletCoefficientsEarth,
    sleplet.functions.axisymmetric_wavelet_coefficients_south_america.AxisymmetricWaveletCoefficientsSouthAmerica,
    sleplet.functions.axisymmetric_wavelets.AxisymmetricWavelets,
    sleplet.functions.dirac_delta.DiracDelta,
    sleplet.functions.earth.Earth,
    sleplet.functions.elongated_gaussian.ElongatedGaussian,
    sleplet.functions.gaussian.Gaussian,
    sleplet.functions.harmonic_gaussian.HarmonicGaussian,
    sleplet.functions.identity.Identity,
    sleplet.functions.noise_earth.NoiseEarth,
    sleplet.functions.ridgelets.Ridgelets,
    sleplet.functions.south_america.SouthAmerica,
    sleplet.functions.spherical_harmonic.SphericalHarmonic,
    sleplet.functions.squashed_gaussian.SquashedGaussian,
    sleplet.functions.wmap.Wmap,
]

FP: list[type[sleplet.functions.coefficients.Coefficients]] = [
    sleplet.functions.slepian.Slepian,
    sleplet.functions.slepian_africa.SlepianAfrica,
    sleplet.functions.slepian_dirac_delta.SlepianDiracDelta,
    sleplet.functions.slepian_identity.SlepianIdentity,
    sleplet.functions.slepian_noise_africa.SlepianNoiseAfrica,
    sleplet.functions.slepian_noise_south_america.SlepianNoiseSouthAmerica,
    sleplet.functions.slepian_south_america.SlepianSouthAmerica,
    sleplet.functions.slepian_wavelet_coefficients_africa.SlepianWaveletCoefficientsAfrica,
    sleplet.functions.slepian_wavelet_coefficients_south_america.SlepianWaveletCoefficientsSouthAmerica,
    sleplet.functions.slepian_wavelets.SlepianWavelets,
]

COEFFICIENTS: list[type[sleplet.functions.coefficients.Coefficients]] = FLM + FP

MAPS_LM: list[type[sleplet.functions.coefficients.Coefficients]] = [
    sleplet.functions.earth.Earth,
    sleplet.functions.south_america.SouthAmerica,
    sleplet.functions.wmap.Wmap,
]

MESH_HARMONIC: list[type[sleplet.meshes.mesh_coefficients.MeshCoefficients]] = [
    sleplet.meshes.mesh_basis_functions.MeshBasisFunctions,
    sleplet.meshes.mesh_field.MeshField,
    sleplet.meshes.mesh_noise_field.MeshNoiseField,
]

MESH_SLEPIAN: list[type[sleplet.meshes.mesh_coefficients.MeshCoefficients]] = [
    sleplet.meshes.mesh_slepian_field.MeshSlepianField,
    sleplet.meshes.mesh_slepian_functions.MeshSlepianFunctions,
    sleplet.meshes.mesh_slepian_noise_field.MeshSlepianNoiseField,
    sleplet.meshes.mesh_slepian_wavelet_coefficients.MeshSlepianWaveletCoefficients,
    sleplet.meshes.mesh_slepian_wavelets.MeshSlepianWavelets,
]

MESH_COEFFICIENTS: list[type[sleplet.meshes.mesh_coefficients.MeshCoefficients]] = (
    MESH_HARMONIC + MESH_SLEPIAN
)
