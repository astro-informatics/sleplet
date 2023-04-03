import logging
from dataclasses import KW_ONLY

import numpy as np
import pyssht as ssht
from numpy import typing as npt
from pydantic.dataclasses import dataclass

import sleplet._integration_methods
import sleplet._validation
import sleplet._vars
from sleplet.slepian.slepian_functions import SlepianFunctions

_logger = logging.getLogger(__name__)


@dataclass(config=sleplet._validation.Validation)
class SlepianDecomposition:
    L: int
    slepian: SlepianFunctions
    _: KW_ONLY
    f: npt.NDArray[np.complex_] | None = None
    flm: npt.NDArray[np.complex_ | np.float_] | None = None
    mask: npt.NDArray[np.float_] | None = None

    def __post_init_post_parse__(self) -> None:
        self._detect_method()

    def decompose(self, rank: int) -> complex:
        """Decompose the signal into its Slepian coefficients via the given method."""
        self._validate_rank(rank)

        match self.method:
            case "harmonic_sum":
                return self._harmonic_sum(rank)
            case "integrate_sphere":
                return self._integrate_sphere(rank)
            case "integrate_region":
                return self._integrate_region(rank)
            case _:
                raise ValueError(f"'{self.method}' is not a valid method")

    def decompose_all(self, n_coefficients: int) -> npt.NDArray[np.complex_]:
        """Decompose all ranks of the Slepian coefficients."""
        coefficients = np.zeros(n_coefficients, dtype=np.complex_)
        for rank in range(n_coefficients):
            coefficients[rank] = self.decompose(rank)
        return coefficients

    def _integrate_region(self, rank: int) -> complex:
        r"""
        F_{p} =
        \frac{1}{\lambda_{p}}
        \int\limits_{R} \dd{\Omega(\omega)}
        f(\omega) \overline{S_{p}(\omega)}.
        """
        assert isinstance(self.mask, np.ndarray)  # noqa: S101
        assert isinstance(self.f, np.ndarray)  # noqa: S101
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

    def _integrate_sphere(self, rank: int) -> complex:
        r"""
        F_{p} =
        \int\limits_{S^{2}} \dd{\Omega(\omega)}
        f(\omega) \overline{S_{p}(\omega)}.
        """
        assert isinstance(self.f, np.ndarray)  # noqa: S101
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

    def _harmonic_sum(self, rank: int) -> complex:
        r"""
        F_{p} =
        \sum\limits_{\ell=0}^{L^{2}}
        \sum\limits_{m=-\ell}^{\ell}
        f_{\ell m} (S_{p})_{\ell m}^{*}.
        """
        return (self.flm * self.slepian.eigenvectors[rank].conj()).sum()

    def _detect_method(self) -> None:
        """Detects what method is used to perform the decomposition."""
        if isinstance(self.flm, np.ndarray):
            _logger.info("harmonic sum method selected")
            self.method = "harmonic_sum"
        elif isinstance(self.f, np.ndarray) and not isinstance(self.mask, np.ndarray):
            _logger.info("integrating the whole sphere method selected")
            self.method = "integrate_sphere"
        elif isinstance(self.f, np.ndarray):
            _logger.info("integrating a region on the sphere method selected")
            self.method = "integrate_region"
        else:
            raise RuntimeError(
                "need to pass one off harmonic coefficients, real pixels "
                "or real pixels with a mask",
            )

    def _validate_rank(self, rank: int) -> None:
        """Checks the requested rank is valid."""
        if not isinstance(rank, int):
            raise TypeError("rank should be an integer")
        if rank < 0:
            raise ValueError("rank cannot be negative")
        if rank >= self.L**2:
            raise ValueError(f"rank should be less than {self.L**2}")
