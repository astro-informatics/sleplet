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

FLM: list[type[sleplet.functions.Coefficients]] = [
    sleplet.functions.flm.Africa,
    sleplet.functions.flm.AxisymmetricWaveletCoefficientsAfrica,
    sleplet.functions.flm.AxisymmetricWaveletCoefficientsEarth,
    sleplet.functions.flm.AxisymmetricWaveletCoefficientsSouthAmerica,
    sleplet.functions.flm.AxisymmetricWavelets,
    sleplet.functions.flm.DiracDelta,
    sleplet.functions.flm.DirectionalSpinWavelets,
    sleplet.functions.flm.Earth,
    sleplet.functions.flm.ElongatedGaussian,
    sleplet.functions.flm.Gaussian,
    sleplet.functions.flm.HarmonicGaussian,
    sleplet.functions.flm.Identity,
    sleplet.functions.flm.NoiseEarth,
    sleplet.functions.flm.Ridgelets,
    sleplet.functions.flm.SouthAmerica,
    sleplet.functions.flm.SphericalHarmonic,
    sleplet.functions.flm.SquashedGaussian,
    sleplet.functions.flm.Wmap,
]

FP: list[type[sleplet.functions.Coefficients]] = [
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

COEFFICIENTS: list[type[sleplet.functions.Coefficients]] = FLM + FP

MAPS_LM: list[type[sleplet.functions.Coefficients]] = [
    sleplet.functions.flm.Earth,
    sleplet.functions.flm.SouthAmerica,
    sleplet.functions.flm.Wmap,
]

MESH_HARMONIC: list[type[sleplet.meshes.MeshCoefficients]] = [
    sleplet.meshes.harmonic_coefficients.MeshBasisFunctions,
    sleplet.meshes.harmonic_coefficients.MeshField,
    sleplet.meshes.harmonic_coefficients.MeshNoiseField,
]

MESH_SLEPIAN: list[type[sleplet.meshes.MeshCoefficients]] = [
    sleplet.meshes.slepian_coefficientsMeshSlepianField,
    sleplet.meshes.slepian_coefficientsMeshSlepianFunctions,
    sleplet.meshes.slepian_coefficientsMeshSlepianNoiseField,
    sleplet.meshes.slepian_coefficientsMeshSlepianWaveletCoefficients,
    sleplet.meshes.slepian_coefficientsMeshSlepianWavelets,
]

MESH_COEFFICIENTS: list[type[sleplet.meshes.MeshCoefficients]] = (
    MESH_HARMONIC + MESH_SLEPIAN
)
