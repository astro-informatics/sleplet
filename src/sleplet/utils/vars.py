import os

import numpy as np

AFRICA_ALPHA = np.deg2rad(44)
AFRICA_BETA = np.deg2rad(87)
AFRICA_GAMMA = np.deg2rad(341)
AFRICA_RANGE = np.deg2rad(41)
ALPHA_DEFAULT = 0.75
ANNOTATION_COLOUR = "gold"
ARROW_STYLE = {
    "arrowhead": 0,
    "arrowside": "start",
    "ax": 4,
    "ay": 4,
    "startarrowsize": 0.5,
    "startarrowhead": 6,
}
BETA_DEFAULT = 0.125
MESH_CBAR_FONT_SIZE = 32
MESH_CBAR_LEN = 0.95
MESH_UNSEEN = -1e5  # kaleido bug
NCPU = int(os.getenv("NCPU", "4"))
PHI_0 = np.pi
PHI_MAX_DEFAULT = 2 * np.pi
PHI_MAX = int(os.getenv("PHI_MAX", "360"))
PHI_MIN_DEFAULT = 0.0
PHI_MIN = int(os.getenv("PHI_MIN", "0"))
POLAR_GAP = os.getenv("POLAR_GAP", "False").lower() == "true"
RANDOM_SEED = 30
SAMPLES = 2
SAMPLING_SCHEME = "MWSS"
SLEPIAN_MASK = os.getenv("SLEPIAN_MASK", "south_america")
SMOOTHING = 2
SOUTH_AMERICA_ALPHA = np.deg2rad(54)
SOUTH_AMERICA_BETA = np.deg2rad(108)
SOUTH_AMERICA_GAMMA = np.deg2rad(63)
SOUTH_AMERICA_RANGE = np.deg2rad(40)
SPHERE_UNSEEN = -1.56e30
THETA_0 = 0.0
THETA_MAX_DEFAULT = np.pi
THETA_MAX = int(os.getenv("THETA_MAX", "180"))
THETA_MIN_DEFAULT = 0
THETA_MIN = int(os.getenv("THETA_MIN", "0"))
ZENODO_DATA_DOI = "10.5281/zenodo.7767698"
