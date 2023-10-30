import dataclasses
import logging

import numpy as np
import numpy.typing as npt
import pydantic
import typing_extensions

import pyssht as ssht

import sleplet._integration_methods
import sleplet._validation
import sleplet._vars
from sleplet.slepian.slepian_functions import SlepianFunctions

_logger = logging.getLogger(__name__)


@pydantic.dataclasses.dataclass(config=sleplet._validation.validation)
class SlepianDecomposition:
    L: int
    slepian: SlepianFunctions
    _: dataclasses.KW_ONLY
    f: npt.NDArray[np.complex_] | None = None
    flm: npt.NDArray[np.complex_ | np.float_] | None = None
    mask: npt.NDArray[np.float_] | None = None
    _method: str = pydantic.Field(default="", init_var=False, repr=False)

    def __post_init__(self: typing_extensions.Self) -> None:
        self._detect_method()

    def decompose(self: typing_extensions.Self, rank: int) -> complex:
        """Decompose the signal into its Slepian coefficients via the given method."""
        self._validate_rank(rank)

        match self._method:
            case "harmonic_sum":
                return self._harmonic_sum(rank)
            case "integrate_sphere":
                return self._integrate_sphere(rank)
            case "integrate_region":
                return self._integrate_region(rank)
            case _:
                msg = f"'{self._method}' is not a valid method"
                raise ValueError(msg)

    def decompose_all(
        self: typing_extensions.Self,
        n_coefficients: int,
    ) -> npt.NDArray[np.complex_]:
        """Decompose all ranks of the Slepian coefficients."""
        coefficients = np.zeros(n_coefficients, dtype=np.complex_)
        for rank in range(n_coefficients):
            coefficients[rank] = self.decompose(rank)
        return coefficients

    def _integrate_region(self: typing_extensions.Self, rank: int) -> complex:
        r"""
        F_{p} =
        \frac{1}{\lambda_{p}}
        \int\limits_{R} \dd{\Omega(\omega)}
        f(\omega) \overline{S_{p}(\omega)}.
        """
        s_p = ssht.inverse(
            self.slepian.eigenvectors[rank],
            self.L,
            Method=sleplet._vars.SAMPLING_SCHEME,
        )
        weight = sleplet._integration_methods.calc_integration_weight(self.L)
        integration = sleplet._integration_methods.integrate_region_sphere(
            self.mask,
            weight,
            self.f,
            s_p.conj(),
        )
        return integration / self.slepian.eigenvalues[rank]

    def _integrate_sphere(self: typing_extensions.Self, rank: int) -> complex:
        r"""
        F_{p} =
        \int\limits_{S^{2}} \dd{\Omega(\omega)}
        f(\omega) \overline{S_{p}(\omega)}.
        """
        s_p = ssht.inverse(
            self.slepian.eigenvectors[rank],
            self.L,
            Method=sleplet._vars.SAMPLING_SCHEME,
        )
        weight = sleplet._integration_methods.calc_integration_weight(self.L)
        return sleplet._integration_methods.integrate_whole_sphere(
            weight,
            self.f,
            s_p.conj(),
        )

    def _harmonic_sum(self: typing_extensions.Self, rank: int) -> complex:
        r"""
        F_{p} =
        \sum\limits_{\ell=0}^{L^{2}}
        \sum\limits_{m=-\ell}^{\ell}
        f_{\ell m} (S_{p})_{\ell m}^{*}.
        """
        return (self.flm * self.slepian.eigenvectors[rank].conj()).sum()

    def _detect_method(self: typing_extensions.Self) -> None:
        """Detect what method is used to perform the decomposition."""
        if self.flm is not None:
            _logger.info("harmonic sum method selected")
            self._method = "harmonic_sum"
        elif self.f is not None:
            if self.mask is None:
                _logger.info("integrating the whole sphere method selected")
                self._method = "integrate_sphere"
            else:
                _logger.info("integrating a region on the sphere method selected")
                self._method = "integrate_region"
        else:
            msg = (
                "need to pass one off harmonic coefficients, real pixels "
                "or real pixels with a mask"
            )
            raise RuntimeError(msg)

    def _validate_rank(self: typing_extensions.Self, rank: int) -> None:
        """Check the requested rank is valid."""
        if not isinstance(rank, int):
            msg = "rank should be an integer"
            raise TypeError(msg)
        if rank < 0:
            msg = "rank cannot be negative"
            raise ValueError(msg)
        if rank >= self.L**2:
            msg = f"rank should be less than {self.L**2}"
            raise ValueError(msg)
