import multiprocess.shared_memory
import numpy as np
import numpy.typing as npt


def split_arr_into_chunks(arr_max: int, ncpu: int) -> list[npt.NDArray[np.int_]]:
    """Split L into a list of arrays for parallelism."""
    arr = np.arange(arr_max)
    size = len(arr)
    arr[size // 2 : size] = arr[size // 2 : size][::-1]
    return [np.sort(arr[i::ncpu]) for i in range(ncpu)]


def create_shared_memory_array(
    array: npt.NDArray[np.float_],
) -> tuple[npt.NDArray[np.float_], multiprocess.shared_memory.SharedMemory]:
    """Create a shared memory array to be used in a parallel function."""
    ext_shared_memory = multiprocess.shared_memory.SharedMemory(
        create=True,
        size=array.nbytes,
    )
    array_ext: npt.NDArray[np.float_] = np.ndarray(
        array.shape,
        dtype=array.dtype,
        buffer=ext_shared_memory.buf,
    )
    return array_ext, ext_shared_memory


def attach_to_shared_memory_block(
    array: npt.NDArray[np.float_],
    ext_shared_memory: multiprocess.shared_memory.SharedMemory,
) -> tuple[npt.NDArray[np.float_], multiprocess.shared_memory.SharedMemory]:
    """Attach an array to the shared memory within the parallel function."""
    int_shared_memory = multiprocess.shared_memory.SharedMemory(
        name=ext_shared_memory.name,
    )
    array_int: npt.NDArray[np.float_] = np.ndarray(
        array.shape,
        dtype=array.dtype,
        buffer=int_shared_memory.buf,
    )
    return array_int, int_shared_memory


def free_shared_memory(*shared_memory: multiprocess.shared_memory.SharedMemory) -> None:
    """Close the shared memory object."""
    for shm in shared_memory:
        shm.close()


def release_shared_memory(
    *shared_memory: multiprocess.shared_memory.SharedMemory,
) -> None:
    """Releases the shared memory object."""
    for shm in shared_memory:
        shm.unlink()
