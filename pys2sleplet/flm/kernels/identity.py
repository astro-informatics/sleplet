import numpy as np

from pys2sleplet.flm.functions import Functions


class Identity(Functions):
    def __init__(self, L: int):
        name = "identity"
        self.reality = False
        super().__init__(name, L)

    def create_flm(self) -> np.ndarray:
        flm = np.ones((self.L * self.L)) + 1j * np.zeros((self.L * self.L))
        return flm
