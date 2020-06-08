from typing import Dict, List

import numpy as np

L: int = 18
ORDER_DICT: Dict[int, List[int]] = {
    0: list(range(4)),
    1: list(range(3)),
    2: list(range(3)),
    3: list(range(2)),
    4: list(range(2)),
    5: list(range(2)),
    6: list(range(1)),
    7: list(range(1)),
}
THETA_MAX: float = 2 * np.pi / 9
