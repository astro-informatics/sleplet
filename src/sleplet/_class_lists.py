from glob import glob
from pathlib import Path

import sleplet.functions
import sleplet.functions.flm
import sleplet.functions.fp
import sleplet.meshes
import sleplet.meshes.harmonic_coefficients
import sleplet.meshes.slepian_coefficients

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
    sleplet.functions.fp.Slepian,
    sleplet.functions.fp.SlepianAfrica,
    sleplet.functions.fp.SlepianDiracDelta,
    sleplet.functions.fp.SlepianIdentity,
    sleplet.functions.fp.SlepianNoiseAfrica,
    sleplet.functions.fp.SlepianNoiseSouthAmerica,
    sleplet.functions.fp.SlepianSouthAmerica,
    sleplet.functions.fp.SlepianWaveletCoefficientsAfrica,
    sleplet.functions.fp.SlepianWaveletCoefficientsSouthAmerica,
    sleplet.functions.fp.SlepianWavelets,
]

COEFFICIENTS: list[type[sleplet.functions.coefficients.Coefficientss]] = FLM + FP

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
    sleplet.meshes.slepian_coefficientsMeshSlepianField,
    sleplet.meshes.slepian_coefficientsMeshSlepianFunctions,
    sleplet.meshes.slepian_coefficientsMeshSlepianNoiseField,
    sleplet.meshes.slepian_coefficientsMeshSlepianWaveletCoefficients,
    sleplet.meshes.slepian_coefficientsMeshSlepianWavelets,
]

MESH_COEFFICIENTS: list[type[sleplet.meshes.mesh_coefficients.MeshCoefficients]] = (
    MESH_HARMONIC + MESH_SLEPIAN
)
