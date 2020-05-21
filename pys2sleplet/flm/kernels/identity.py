from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from pys2sleplet.flm.functions import Functions


@dataclass
class Identity(Functions):
    extra_args: Optional[List[int]] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()

    def _setup_args(self) -> None:
        pass

    def _set_reality(self) -> bool:
        return True

    def _create_flm(self) -> np.ndarray:
        flm = np.ones((self.L * self.L)) + 1j * np.zeros((self.L * self.L))
        return flm

    def _create_name(self) -> str:
        name = "identity"
        return name

    def _create_annotations(self) -> List[Dict]:
        pass
