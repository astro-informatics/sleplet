from pys2sleplet.functions import Functions


class DiracDelta(Functions):
    def __init__(self):
        super().__init__("dirac_delta", reality=True)
