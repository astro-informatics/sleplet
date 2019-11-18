import numpy as np

from ..functions import Functions


class Identity(Functions):
    def __init__(self, L: int):
        name = "identity"
        super().__init__(name, L)

    def create_flm(self):
        self.flm = np.ones((self.L * self.L)) + 1j * np.zeros((self.L * self.L))
