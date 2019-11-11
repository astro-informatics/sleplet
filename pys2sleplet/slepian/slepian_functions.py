from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class SlepianFunctions(ABC):
    def __init__(self, L: int) -> None:
        self.L = L
        self.eigenvalues, self.eigenvectors = self.eigenproblem()

    @abstractmethod
    def eigenproblem(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def annotations(self) -> List[dict]:
        raise NotImplementedError

    # def eigenproblem(self) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #     calculates the eigenvalues and eigenvectors of the D matrix
    #     """
    #     self.filename = ""
    #     if self.is_polar_cap:
    #         # polar cap/gap name addition
    #         if self.is_polar_gap:
    #             self.filename += "_gap"
    #         self.filename += f"_m{self.order}"
    #         # numpy binary filename
    #         binary = (
    #             Path(__file__).resolve().parent
    #             / "data"
    #             / "slepian"
    #             / "polar"
    #             / f"D{self.filename_angle()}{self.filename}_L{self.L}"
    #         )
    #         spc = SlepianPolarCap(
    #             self.L,
    #             self.theta_max,
    #             binary,
    #             self.is_polar_gap,
    #             self.ncpu,
    #             self.save_matrices,
    #         )
    #         eigenvalues, eigenvectors = spc.eigenproblem(self.order)
    #     else:
    #         # numpy binary filename
    #         binary = (
    #             Path(__file__).resolve().parent
    #             / "data"
    #             / "slepian"
    #             / "lat_long"
    #             / f"K{self.filename_angle()}{self.filename}_L{self.L}"
    #         )
    #         slll = SlepianLimitLatLong(
    #             self.L,
    #             self.phi_min,
    #             self.phi_max,
    #             self.theta_min,
    #             self.theta_max,
    #             binary,
    #             self.ncpu,
    #             self.save_matrices,
    #         )
    #         eigenvalues, eigenvectors = slll.eigenproblem()

    #     sa = SlepianArbitrary(
    #         self.L,
    #         self.phi_min,
    #         self.phi_max,
    #         self.theta_min,
    #         self.theta_max,
    #         self.ncpu,
    #     )
    #     eigenvalues, eigenvectors = sa.eigenproblem()
    #     return eigenvalues, eigenvectors
