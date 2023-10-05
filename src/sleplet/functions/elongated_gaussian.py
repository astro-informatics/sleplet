"""Contains the `ElongatedGaussian` class."""
import numpy as np
import numpy.typing as npt
import pydantic.v1 as pydantic

import sleplet._string_methods
import sleplet._validation
import sleplet._vars
import sleplet.harmonic_methods
from sleplet.functions.flm import Flm


@pydantic.dataclasses.dataclass(config=sleplet._validation.Validation, kw_only=True)
class ElongatedGaussian(Flm):
    r"""
    Creates an elongated Gaussian
    \(\exp(-(\frac{{(\theta-\overline{\theta})}^{2}}{2\sigma_{\theta}^{2}}
    + \frac{{(\phi-\overline{\phi})}^{2}}{2\sigma_{\phi}^{2}}))\).
    """

    p_sigma: float = 0.1
    r"""Sets the \(\sigma_{\phi}\) value."""
    t_sigma: float = 1
    r"""Sets the \(\sigma_{\theta}\) value."""

    def __post_init_post_parse__(self) -> None:
        super().__post_init_post_parse__()

    def _create_coefficients(self) -> npt.NDArray[np.complex_ | np.float_]:
        return sleplet.harmonic_methods._ensure_f_bandlimited(
            self._grid_fun,
            self.L,
            Reality=self.reality,
            Spin=self.spin,
        )

    def _create_name(self) -> str:
        return (
            f"{sleplet._string_methods._convert_camel_case_to_snake_case(self.__class__.__name__)}"
            f"{sleplet._string_methods.filename_args(self.t_sigma, 'tsig')}"
            f"{sleplet._string_methods.filename_args(self.p_sigma, 'psig')}"
        )

    def _set_reality(self) -> bool:
        return True

    def _set_spin(self) -> int:
        return 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 2
            if len(self.extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.t_sigma, self.p_sigma = (
                np.float_power(10, x) for x in self.extra_args
            )

    def _grid_fun(
        self,
        theta: npt.NDArray[np.float_],
        phi: npt.NDArray[np.float_],
    ) -> npt.NDArray[np.float_]:
        """Function on the grid."""
        return np.exp(
            -(
                ((theta - sleplet._vars.THETA_0) / self.t_sigma) ** 2
                + ((phi - sleplet._vars.PHI_0) / self.p_sigma) ** 2
            )
            / 2,
        )
