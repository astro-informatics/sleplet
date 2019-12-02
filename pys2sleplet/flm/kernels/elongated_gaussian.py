from argparse import Namespace
from typing import List, Tuple

import numpy as np

from pys2sleplet.flm.functions import Functions
from pys2sleplet.utils.plot_methods import ensure_f_bandlimited
from pys2sleplet.utils.string_methods import filename_args


class ElongatedGaussian(Functions):
    def __init__(self, L: int, args: Namespace = Namespace(extra_args=[0, -3])):
        self.t_sig, self.p_sig = self.validate_args(args)
        name = f"elongated_gaussian{filename_args(self.t_sig, 'tsig')}{filename_args(self.p_sig, 'psig')}"
        self.reality = True
        super().__init__(name, L)

    @staticmethod
    def read_args(args: List[int]) -> Tuple[int, int]:
        # args
        try:
            t_sig, p_sig = int(args.pop(0)), int(args.pop(0))
        except IndexError:
            raise ValueError("function requires exactly two extra args")
        return t_sig, p_sig

    def validate_args(self, args: Namespace) -> Tuple[int, int]:
        extra_args = args.extra_args
        t_sig, p_sig = self.read_args(extra_args)

        # validation
        if not float(t_sig).is_integer():
            raise ValueError("theta sigma should be an integer")
        if not float(p_sig).is_integer():
            raise ValueError("phi sigma should be an integer")
        t_sig, p_sig = 10 ** t_sig, 10 ** p_sig
        return t_sig, p_sig

    def grid_fun(
        self,
        theta: np.ndarray,
        phi: np.ndarray,
        theta_0: float = 0,
        phi_0: float = np.pi,
    ) -> np.ndarray:
        """
        function on the grid
        """
        f = np.exp(
            -(((theta - theta_0) / self.t_sig) ** 2 + ((phi - phi_0) / self.p_sig) ** 2)
            / 2
        )
        return f

    def create_flm(self) -> np.ndarray:
        flm = ensure_f_bandlimited(self.grid_fun, self.L, self.reality)
        return flm
