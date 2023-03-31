from glob import glob
from pathlib import Path

import sleplet.functions.coefficients
import sleplet.functions.flm.africa
import sleplet.functions.flm.axisymmetric_wavelet_coefficients_africa
import sleplet.functions.flm.axisymmetric_wavelet_coefficients_earth
import sleplet.functions.flm.axisymmetric_wavelet_coefficients_south_america
import sleplet.functions.flm.axisymmetric_wavelets
import sleplet.functions.flm.dirac_delta
import sleplet.functions.flm.directional_spin_wavelets
import sleplet.functions.flm.earth
import sleplet.functions.flm.elongated_gaussian
import sleplet.functions.flm.gaussian
import sleplet.functions.flm.harmonic_gaussian
import sleplet.functions.flm.identity
import sleplet.functions.flm.noise_earth
import sleplet.functions.flm.ridgelets
import sleplet.functions.flm.south_america
import sleplet.functions.flm.spherical_harmonic
import sleplet.functions.flm.squashed_gaussian
import sleplet.functions.flm.wmap
import sleplet.functions.fp.slepian
import sleplet.functions.fp.slepian_africa
import sleplet.functions.fp.slepian_dirac_delta
import sleplet.functions.fp.slepian_identity
import sleplet.functions.fp.slepian_noise_africa
import sleplet.functions.fp.slepian_noise_south_america
import sleplet.functions.fp.slepian_south_america
import sleplet.functions.fp.slepian_wavelet_coefficients_africa
import sleplet.functions.fp.slepian_wavelet_coefficients_south_america
import sleplet.functions.fp.slepian_wavelets
import sleplet.meshes.harmonic_coefficients.mesh_basis_functions
import sleplet.meshes.harmonic_coefficients.mesh_field
import sleplet.meshes.harmonic_coefficients.mesh_noise_field
import sleplet.meshes.mesh_coefficients
import sleplet.meshes.slepian_coefficients.mesh_slepian_field
import sleplet.meshes.slepian_coefficients.mesh_slepian_functions
import sleplet.meshes.slepian_coefficients.mesh_slepian_noise_field
import sleplet.meshes.slepian_coefficients.mesh_slepian_wavelet_coefficients
import sleplet.meshes.slepian_coefficients.mesh_slepian_wavelets

_data_path = Path(__file__).resolve().parent / "_data"

MESHES: list[str] = [
    Path(x).stem.removeprefix("meshes_regions_")
    for x in glob(str(_data_path / "*.toml"))
]

FLM: list[type[sleplet.functions.coefficients.Coefficients]] = [
    sleplet.functions.flm.africa.Africa,
    sleplet.functions.flm.axisymmetric_wavelet_coefficients_africa.AxisymmetricWaveletCoefficientsAfrica,
    sleplet.functions.flm.axisymmetric_wavelet_coefficients_earth.AxisymmetricWaveletCoefficientsEarth,
    sleplet.functions.flm.axisymmetric_wavelet_coefficients_south_america.AxisymmetricWaveletCoefficientsSouthAmerica,
    sleplet.functions.flm.axisymmetric_wavelets.AxisymmetricWavelets,
    sleplet.functions.flm.dirac_delta.DiracDelta,
    sleplet.functions.flm.directional_spin_wavelets.DirectionalSpinWavelets,
    sleplet.functions.flm.earth.Earth,
    sleplet.functions.flm.elongated_gaussian.ElongatedGaussian,
    sleplet.functions.flm.gaussian.Gaussian,
    sleplet.functions.flm.harmonic_gaussian.HarmonicGaussian,
    sleplet.functions.flm.identity.Identity,
    sleplet.functions.flm.noise_earth.NoiseEarth,
    sleplet.functions.flm.ridgelets.Ridgelets,
    sleplet.functions.flm.south_america.SouthAmerica,
    sleplet.functions.flm.spherical_harmonic.SphericalHarmonic,
    sleplet.functions.flm.squashed_gaussian.SquashedGaussian,
    sleplet.functions.flm.wmap.Wmap,
]

FP: list[type[sleplet.functions.coefficients.Coefficients]] = [
    sleplet.functions.fp.slepian.Slepian,
    sleplet.functions.fp.slepian_africa.SlepianAfrica,
    sleplet.functions.fp.slepian_dirac_delta.SlepianDiracDelta,
    sleplet.functions.fp.slepian_identity.SlepianIdentity,
    sleplet.functions.fp.slepian_noise_africa.SlepianNoiseAfrica,
    sleplet.functions.fp.slepian_noise_south_america.SlepianNoiseSouthAmerica,
    sleplet.functions.fp.slepian_south_america.SlepianSouthAmerica,
    sleplet.functions.fp.slepian_wavelet_coefficients_africa.SlepianWaveletCoefficientsAfrica,
    sleplet.functions.fp.slepian_wavelet_coefficients_south_america.SlepianWaveletCoefficientsSouthAmerica,
    sleplet.functions.fp.slepian_wavelets.SlepianWavelets,
]

COEFFICIENTS: list[type[sleplet.functions.coefficients.Coefficients]] = FLM + FP

MAPS_LM: list[type[sleplet.functions.coefficients.Coefficients]] = [
    sleplet.functions.flm.earth.Earth,
    sleplet.functions.flm.south_america.SouthAmerica,
    sleplet.functions.flm.wmap.Wmap,
]

MESH_HARMONIC: list[type[sleplet.meshes.mesh_coefficients.MeshCoefficients]] = [
    sleplet.meshes.harmonic_coefficients.mesh_basis_functions.MeshBasisFunctions,
    sleplet.meshes.harmonic_coefficients.mesh_field.MeshField,
    sleplet.meshes.harmonic_coefficients.mesh_noise_field.MeshNoiseField,
]

MESH_SLEPIAN: list[type[sleplet.meshes.mesh_coefficients.MeshCoefficients]] = [
    sleplet.meshes.slepian_coefficients.mesh_slepian_field.MeshSlepianField,
    sleplet.meshes.slepian_coefficients.mesh_slepian_functions.MeshSlepianFunctions,
    sleplet.meshes.slepian_coefficients.mesh_slepian_noise_field.MeshSlepianNoiseField,
    sleplet.meshes.slepian_coefficients.mesh_slepian_wavelet_coefficients.MeshSlepianWaveletCoefficients,
    sleplet.meshes.slepian_coefficients.mesh_slepian_wavelets.MeshSlepianWavelets,
]

MESH_COEFFICIENTS: list[type[sleplet.meshes.mesh_coefficients.MeshCoefficients]] = (
    MESH_HARMONIC + MESH_SLEPIAN
)
