from typing import List

import numpy as np

from pys2sleplet.utils.vars import L_MIN_DEFAULT


def split_L_into_chunks(
    L_max: int, ncpu: int, L_min: int = L_MIN_DEFAULT
) -> List[np.ndarray]:
    """
    split L into a list of arrays for parallelism
    """
    arr = np.arange(L_min, L_max)
    size = len(arr)
    arr[size // 2 : size] = arr[size // 2 : size][::-1]
    return [np.sort(arr[i::ncpu]) for i in range(ncpu)]
