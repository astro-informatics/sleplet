from dataclasses import dataclass

from pys2sleplet.flm.functions import Functions
from pys2sleplet.utils.pys2let import s2let


@dataclass
class Wavelet(Functions):
    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_annotations(self) -> None:
        pass

    def _create_flm(self) -> None:
        B = 3
        J_min = 1
        scal, wav = s2let.axisym_wav_l(B, self.L, J_min)
        self.multipole = scal.astype(complex)

    def _create_name(self) -> None:
        self.name = "wavelet"

    def _set_reality(self) -> None:
        self.reality = False

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            raise AttributeError(
                f"{self.__class__.__name__} does not support extra arguments"
            )
