from argparse import Namespace
from typing import List, Tuple

import numpy as np

from pys2sleplet.flm.functions import Functions
from pys2sleplet.utils.plot_methods import ensure_f_bandlimited
from pys2sleplet.utils.string_methods import filename_args


class SquashedGaussian(Functions):
    def __init__(self, L: int, args: Namespace = Namespace(extra_args=[-2, -1])):
        self.t_sig, self.freq = self.validate_args(args)
        name = f"squashed_gaussian{filename_args(self.t_sig, 'tsig')}{filename_args(self.freq, 'freq')}"
        reality = True
        super().__init__(name, L, reality)

    @staticmethod
    def read_args(args: List[int]) -> Tuple[int, int]:
        # args
        try:
            t_sig, freq = args.pop(0), args.pop(0)
        except IndexError:
            raise ValueError("function requires exactly two extra args")
        return t_sig, freq

    def validate_args(self, args: Namespace) -> Tuple[int, int]:
        extra_args = args.extra_args
        t_sig, freq = self.validate_args(extra_args)

        # validation
        if not float(t_sig).is_integer():
            raise ValueError("theta sigma should be an integer")
        if not float(freq).is_integer():
            raise ValueError("sine frequency should be an integer")
        t_sig, freq = 10 ** t_sig, 10 ** freq
        return t_sig, freq

    def grid_fun(
        self, theta: np.ndarray, phi: np.ndarray, theta_0: float = 0
    ) -> np.ndarray:
        """
        function on the grid
        """
        f = np.exp(-(((theta - theta_0) / self.t_sig) ** 2) / 2) * np.sin(
            self.freq * phi
        )
        return f

    def create_flm(self) -> np.ndarray:
        flm = ensure_f_bandlimited(self.grid_fun, self.L, self.reality)
        return flm
