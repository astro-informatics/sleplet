from typing import Dict

from pys2sleplet.flm.functions import Functions
from pys2sleplet.flm.kernels.dirac_delta import DiracDelta
from pys2sleplet.flm.kernels.elongated_gaussian import ElongatedGaussian
from pys2sleplet.flm.kernels.gaussian import Gaussian
from pys2sleplet.flm.kernels.harmonic_gaussian import HarmonicGaussian
from pys2sleplet.flm.kernels.identity import Identity
from pys2sleplet.flm.kernels.slepian import Slepian
from pys2sleplet.flm.kernels.spherical_harmonic import SphericalHarmonic
from pys2sleplet.flm.kernels.squashed_gaussian import SquashedGaussian
from pys2sleplet.flm.maps.earth import Earth
from pys2sleplet.flm.maps.south_america import SouthAmerica
from pys2sleplet.flm.maps.wmap import Wmap

KERNELS: Dict[str, Functions] = dict(
    dirac_delta=DiracDelta,
    elongated_gaussian=ElongatedGaussian,
    gaussian=Gaussian,
    harmonic_gaussian=HarmonicGaussian,
    identity=Identity,
    slepian=Slepian,
    spherical_harmonic=SphericalHarmonic,
    squashed_gaussian=SquashedGaussian,
)

MAPS: Dict[str, Functions] = dict(earth=Earth, south_america=SouthAmerica, wmap=Wmap)

FUNCTIONS: Dict[str, Functions] = {**KERNELS, **MAPS}
