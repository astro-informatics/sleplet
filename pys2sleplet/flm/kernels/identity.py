from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from pys2sleplet.flm.functions import Functions


@dataclass
class Identity(Functions):
    L: int
    extra_args: Optional[List[int]] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.reality = True

    def _setup_args(self) -> None:
        pass

    def _create_flm(self) -> np.ndarray:
        flm = np.ones((self.L * self.L)) + 1j * np.zeros((self.L * self.L))
        return flm

    def _create_name(self) -> str:
        name = "identity"
        return name

    def _create_annotations(self) -> List[Dict]:
        pass
