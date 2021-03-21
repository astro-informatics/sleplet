import numpy as np
from multiprocess.shared_memory import SharedMemory

from pys2sleplet.utils.vars import L_MIN_DEFAULT


def split_L_into_chunks(
    L_max: int, ncpu: int, L_min: int = L_MIN_DEFAULT
) -> list[np.ndarray]:
    """
    split L into a list of arrays for parallelism
    """
    arr = np.arange(L_min, L_max)
    size = len(arr)
    arr[size // 2 : size] = arr[size // 2 : size][::-1]
    return [np.sort(arr[i::ncpu]) for i in range(ncpu)]


def create_shared_memory_array(array: np.ndarray) -> tuple[np.ndarray, SharedMemory]:
    """
    creates a shared memory array to be used in a parallel function
    """
    ext_shared_memory = SharedMemory(create=True, size=array.nbytes)
    array_ext = np.ndarray(array.shape, dtype=array.dtype, buffer=ext_shared_memory.buf)
    return array_ext, ext_shared_memory


def attach_to_shared_memory_block(
    array: np.ndarray, ext_shared_memory: SharedMemory
) -> tuple[np.ndarray, SharedMemory]:
    """
    used within the parallel function to attach an array to the shared memory
    """
    int_shared_memory = SharedMemory(name=ext_shared_memory.name)
    array_int = np.ndarray(array.shape, dtype=array.dtype, buffer=int_shared_memory.buf)
    return array_int, int_shared_memory


def free_shared_memory(*shared_memory: SharedMemory) -> None:
    """
    closes the shared memory object
    """
    for shm in shared_memory:
        shm.close()


def release_shared_memory(*shared_memory: SharedMemory) -> None:
    """
    releases the shared memory object
    """
    for shm in shared_memory:
        shm.unlink()
