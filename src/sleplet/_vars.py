from pathlib import Path

import numpy as np

DATA_PATH = Path(__file__).resolve().parent / "_data"
FIG_PATH = Path(__file__).resolve().parent / "_figures"
PHI_0 = np.pi
PHI_MAX_DEFAULT = 2 * np.pi
PHI_MIN_DEFAULT = 0.0
RANDOM_SEED = 30
SAMPLING_SCHEME = "MWSS"
SPHERE_UNSEEN = -1.56e30
THETA_0 = 0.0
THETA_MAX_DEFAULT = np.pi
THETA_MIN_DEFAULT = 0
