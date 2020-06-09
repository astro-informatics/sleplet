from typing import Dict, Union

import numpy as np

L: int = 18
ORDERS: int = 5
ORDER_RANK: Dict[int, int] = {0: 4, 1: 3, 2: 3, 3: 2, 4: 2, 5: 2, 6: 1, 7: 1}
RANKS: int = 4
TEXT_BOX: Dict[str, Union[str, float]] = dict(
    boxstyle="round", facecolor="wheat", alpha=0.5
)
THETA_MAX: float = 2 * np.pi / 9
