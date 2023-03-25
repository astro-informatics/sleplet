import numpy as np

AFRICA_ALPHA: float = np.deg2rad(44)
AFRICA_BETA: float = np.deg2rad(87)
AFRICA_GAMMA: float = np.deg2rad(341)
AFRICA_RANGE: int = np.deg2rad(41)
ALPHA_DEFAULT: float = 0.75
ANNOTATION_COLOUR: str = "gold"
ARROW_STYLE: dict = {
    "arrowhead": 0,
    "arrowside": "start",
    "ax": 4,
    "ay": 4,
    "startarrowsize": 0.5,
    "startarrowhead": 6,
}
BETA_DEFAULT: float = 0.125
SOUTH_AMERICA_ALPHA: float = np.deg2rad(54)
SOUTH_AMERICA_BETA: float = np.deg2rad(108)
SOUTH_AMERICA_GAMMA: float = np.deg2rad(63)
L_MAX_DEFAULT: int = -1
L_MIN_DEFAULT: int = 0
MESH_CBAR_FONT_SIZE: int = 32
MESH_CBAR_LEN: float = 0.95
MESH_UNSEEN: float = -1e5  # kaleido bug
PHI_0: float = np.pi
PHI_MIN_DEFAULT: float = 0.0
PHI_MAX_DEFAULT: float = 2 * np.pi
RANDOM_SEED: int = 30
SAMPLING_SCHEME: str = "MWSS"
SMOOTHING: int = 2
SOUTH_AMERICA_RANGE: int = np.deg2rad(40)
SPHERE_UNSEEN: float = -1.56e30
THETA_0: float = 0.0
THETA_MIN_DEFAULT: float = 0
THETA_MAX_DEFAULT: float = np.pi
ZENODO_DATA_DOI: str = "10.5281/zenodo.7767698"
