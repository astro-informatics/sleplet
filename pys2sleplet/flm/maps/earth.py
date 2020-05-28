from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyssht as ssht
from scipy import io as sio

from pys2sleplet.flm.functions import Functions

_file_location = Path(__file__).resolve()


@dataclass
class Earth(Functions):
    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_annotations(self) -> None:
        pass

    def _create_flm(self) -> None:
        # load in data
        self.flm = self._load_flm()

        # fill in negative m components so as to
        # avoid confusion with zero values
        for ell in range(1, self.L):
            for m in range(1, ell + 1):
                ind_pm = ssht.elm2ind(ell, m)
                ind_nm = ssht.elm2ind(ell, -m)
                self.flm[ind_nm] = (-1) ** m * self.flm[ind_pm].conj()

        # don't take the full L
        # invert dataset as Earth backwards
        self.flm = self.flm[: self.L * self.L].conj()

    def _create_name(self) -> None:
        self.name = "earth"

    def _set_reality(self) -> None:
        self.reality = True

    def _setup_args(self) -> None:
        if self.extra_args is not None:
            raise AttributeError(f"Does not support extra arguments")

    @staticmethod
    def _load_flm() -> np.ndarray:
        """
        load coefficients from file
        """
        matfile = str(
            _file_location.parents[2]
            / "data"
            / "maps"
            / "earth"
            / "EGM2008_Topography_flms_L2190.mat"
        )
        mat_contents = sio.loadmat(matfile)
        flm = np.ascontiguousarray(mat_contents["flm"][:, 0])
        return flm
