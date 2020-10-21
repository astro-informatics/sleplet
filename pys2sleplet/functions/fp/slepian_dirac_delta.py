from dataclasses import dataclass, field

from pys2sleplet.functions.f_p import F_P
from pys2sleplet.utils.plot_methods import calc_nearest_grid_point
from pys2sleplet.utils.slepian_methods import compute_s_p_omega_prime
from pys2sleplet.utils.vars import ALPHA_DEFAULT, BETA_DEFAULT


@dataclass
class SlepianDiracDelta(F_P):
    alpha: float
    beta: float
    _alpha: float = field(default=ALPHA_DEFAULT, init=False, repr=False)
    _beta: float = field(default=BETA_DEFAULT, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_annotations(self) -> None:
        self.annotations = self.slepian.annotations

    def _create_coefficients(self) -> None:
        alpha_point, beta_point = calc_nearest_grid_point(self.L, self.alpha, self.beta)
        self.coefficients = compute_s_p_omega_prime(
            self.L, alpha_point, beta_point, self.slepian
        ).conj()

    def _create_name(self) -> None:
        self.name = f"slepian_dirac_delta_{self.slepian.region.name_ending}"

    def _set_reality(self) -> None:
        self.reality = False

    def _set_spin(self) -> None:
        self.spin = 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            raise AttributeError(
                f"{self.__class__.__name__} does not support extra arguments"
            )

    @property  # type: ignore
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        if isinstance(alpha, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            alpha = SlepianDiracDelta._alpha
        self._alpha = alpha

    @property  # type: ignore
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, beta: float) -> None:
        if isinstance(beta, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            beta = SlepianDiracDelta._beta
        self._beta = beta
