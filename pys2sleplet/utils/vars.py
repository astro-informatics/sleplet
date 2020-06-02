from typing import Dict

import numpy as np

ANNOTATION_DOTS: int = 12
ARROW_STYLE: Dict[str, int] = dict(arrowhead=6, ax=5, ay=5)
ORDER_DEFAULT: int = 0
PHI_0: float = np.pi
PHI_MIN_DEFAULT: float = 0.0
PHI_MAX_DEFAULT: float = 2 * np.pi
SAMPLING_SCHEME: str = "MWSS"
THETA_0: float = 0.0
THETA_MIN_DEFAULT: float = 0
THETA_MAX_DEFAULT: float = np.pi
ZOOM_DEFAULT: float = 7.88
