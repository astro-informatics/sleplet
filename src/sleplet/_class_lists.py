from glob import glob
from pathlib import Path

import sleplet.functions
import sleplet.functions.coefficients
import sleplet.meshes
import sleplet.meshes.mesh_coefficients

_data_path = Path(__file__).resolve().parent / "_data"

MESHES: list[str] = [
    Path(x).stem.removeprefix("meshes_regions_")
    for x in glob(str(_data_path / "*.toml"))
]

FLM: list[type[sleplet.functions.coefficients.Coefficients]] = [
    sleplet.functions.Africa,
    sleplet.functions.AxisymmetricWaveletCoefficientsAfrica,
    sleplet.functions.AxisymmetricWaveletCoefficientsEarth,
    sleplet.functions.AxisymmetricWaveletCoefficientsSouthAmerica,
    sleplet.functions.AxisymmetricWavelets,
    sleplet.functions.DiracDelta,
    sleplet.functions.DirectionalSpinWavelets,
    sleplet.functions.Earth,
    sleplet.functions.ElongatedGaussian,
    sleplet.functions.Gaussian,
    sleplet.functions.HarmonicGaussian,
    sleplet.functions.Identity,
    sleplet.functions.NoiseEarth,
    sleplet.functions.Ridgelets,
    sleplet.functions.SouthAmerica,
    sleplet.functions.SphericalHarmonic,
    sleplet.functions.SquashedGaussian,
    sleplet.functions.Wmap,
]

FP: list[type[sleplet.functions.coefficients.Coefficients]] = [
    sleplet.functions.Slepian,
    sleplet.functions.SlepianAfrica,
    sleplet.functions.SlepianDiracDelta,
    sleplet.functions.SlepianIdentity,
    sleplet.functions.SlepianNoiseAfrica,
    sleplet.functions.SlepianNoiseSouthAmerica,
    sleplet.functions.SlepianSouthAmerica,
    sleplet.functions.SlepianWaveletCoefficientsAfrica,
    sleplet.functions.SlepianWaveletCoefficientsSouthAmerica,
    sleplet.functions.SlepianWavelets,
]

COEFFICIENTS: list[type[sleplet.functions.coefficients.Coefficients]] = FLM + FP

MAPS_LM: list[type[sleplet.functions.coefficients.Coefficients]] = [
    sleplet.functions.Earth,
    sleplet.functions.SouthAmerica,
    sleplet.functions.Wmap,
]

MESH_HARMONIC: list[type[sleplet.meshes.mesh_coefficients.MeshCoefficients]] = [
    sleplet.meshes.MeshBasisFunctions,
    sleplet.meshes.MeshField,
    sleplet.meshes.MeshNoiseField,
]

MESH_SLEPIAN: list[type[sleplet.meshes.mesh_coefficients.MeshCoefficients]] = [
    sleplet.meshes.MeshSlepianField,
    sleplet.meshes.MeshSlepianFunctions,
    sleplet.meshes.MeshSlepianNoiseField,
    sleplet.meshes.MeshSlepianWaveletCoefficients,
    sleplet.meshes.MeshSlepianWavelets,
]

MESH_COEFFICIENTS: list[type[sleplet.meshes.mesh_coefficients.MeshCoefficients]] = (
    MESH_HARMONIC + MESH_SLEPIAN
)
