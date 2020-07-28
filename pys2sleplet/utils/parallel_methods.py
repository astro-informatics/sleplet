from typing import List

import numpy as np


def split_L_into_chunks(L: int, ncpu: int) -> List[np.ndarray]:
    """
    split L into a list of arrays for parallelism
    """
    arr = np.arange(L)
    size = len(arr)
    arr[size // 2 : size] = arr[size // 2 : size][::-1]
    return [np.sort(arr[i::ncpu]) for i in range(ncpu)]
