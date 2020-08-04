from typing import Dict

import numpy as np

ANNOTATION_COLOUR: str = "salmon"
ANNOTATION_SECOND_COLOUR: str = "gold"
ANNOTATION_DOTS: int = 12
ARROW_STYLE: Dict = dict(arrowhead=7, ax=5, ay=5, opacity=0.5)
CONTINENT_ALPHA: float = np.deg2rad(116)
CONTINENT_BETA: float = np.deg2rad(252)
CONTINENT_GAMMA: float = np.deg2rad(38)
DECOMPOSITION_DEFAULT: str = "harmonic_sum"
EARTH_ALPHA: float = np.deg2rad(54)
EARTH_BETA: float = np.deg2rad(108)
EARTH_GAMMA: float = np.deg2rad(63)
GAP_DEFAULT: bool = False
MID_COLOURBAR: float = 0.5
PHI_0: float = np.pi
PHI_MIN_DEFAULT: float = 0.0
PHI_MAX_DEFAULT: float = 2 * np.pi
SAMPLING_SCHEME: str = "MWSS"
SOUTH_AMERICA_RANGE: int = np.deg2rad(40)
THETA_0: float = 0.0
THETA_MIN_DEFAULT: float = 0
THETA_MAX_DEFAULT: float = np.pi
ZOOM_DEFAULT: float = 7.88
