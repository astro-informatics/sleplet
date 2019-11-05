import numpy as np

from pys2sleplet.flm.functions import Functions


class Identity(Functions):
    def __init__(self):
        super().__init__("identity")

    def create_flm(self):
        self.flm = np.ones((self.L * self.L)) + 1j * np.zeros((self.L * self.L))
