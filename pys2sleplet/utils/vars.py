from typing import Dict

import numpy as np

ANNOTATION_COLOUR: str = "salmon"
ANNOTATION_SECOND_COLOUR: str = "gold"
ANNOTATION_DOTS: int = 12
ARROW_STYLE: Dict = dict(
    arrowhead=0,
    arrowside="start",
    ax=4,
    ay=4,
    opacity=0.4,
    startarrowsize=1.5,
    startarrowhead=6,
)
DECOMPOSITION_DEFAULT: str = "harmonic_sum"
EARTH_ALPHA: float = np.deg2rad(54)
EARTH_BETA: float = np.deg2rad(108)
EARTH_GAMMA: float = np.deg2rad(63)
GAP_DEFAULT: bool = False
PHI_0: float = np.pi
PHI_MIN_DEFAULT: float = 0.0
PHI_MAX_DEFAULT: float = 2 * np.pi
SAMPLING_SCHEME: str = "MWSS"
SOUTH_AMERICA_RANGE: int = np.deg2rad(40)
THETA_0: float = 0.0
THETA_MIN_DEFAULT: float = 0
THETA_MAX_DEFAULT: float = np.pi
ZOOM_DEFAULT: float = 7.88
