import numpy as np
import numpy.typing as npt


def sifting_convolution(
    f_coefficient: npt.NDArray[np.complex128 | np.float64],
    g_coefficient: npt.NDArray[np.complex128 | np.float64],
    *,
    shannon: int | None = None,
) -> npt.NDArray[np.complex128 | np.float64]:
    """Compute the sifting convolution between two multipoles."""
    n = shannon if shannon is not None else np.newaxis
    # change shape if the sizes don't match
    g_reshape = (
        g_coefficient[np.newaxis]
        if len(g_coefficient.shape) < len(f_coefficient.shape)
        else g_coefficient
    )
    return (f_coefficient.T[:n] * g_reshape.conj().T[:n]).T  # type: ignore[return-value]
