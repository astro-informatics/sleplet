import numpy as np

AFRICA_ALPHA: float = np.deg2rad(54)
AFRICA_BETA: float = np.deg2rad(108)
AFRICA_GAMMA: float = np.deg2rad(63)
AFRICA_RANGE: int = np.deg2rad(40)
ALPHA_DEFAULT: float = 0.75
ANNOTATION_COLOUR: str = "gold"
ARROW_STYLE: dict = dict(
    arrowhead=0, arrowside="start", ax=4, ay=4, startarrowsize=0.5, startarrowhead=6
)
BETA_DEFAULT: float = 0.125
SOUTH_AMERICA_ALPHA: float = np.deg2rad(54)
SOUTH_AMERICA_BETA: float = np.deg2rad(108)
SOUTH_AMERICA_GAMMA: float = np.deg2rad(63)
GAUSSIAN_KERNEL_KNN_DEFAULT: int = 5
GAUSSIAN_KERNEL_THETA_DEFAULT: float = 1
L_MAX_DEFAULT: int = -1
L_MIN_DEFAULT: int = 0
PHI_0: float = np.pi
PHI_MIN_DEFAULT: float = 0.0
PHI_MAX_DEFAULT: float = 2 * np.pi
RANDOM_SEED: int = 30
SAMPLING_SCHEME: str = "MWSS"
SMOOTHING: int = 2
SOUTH_AMERICA_RANGE: int = np.deg2rad(40)
THETA_0: float = 0.0
THETA_MIN_DEFAULT: float = 0
THETA_MAX_DEFAULT: float = np.pi
UNSEEN: float = -1.56e30
