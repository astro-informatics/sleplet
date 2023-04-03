import numpy as np
import pyssht as ssht
from numpy import typing as npt
from numpy.random import default_rng
from numpy.testing import assert_equal

from sleplet.functions import AxisymmetricWavelets
from sleplet.harmonic_methods import compute_random_signal
from sleplet.wavelet_methods import axisymmetric_wavelet_forward

B = 3
J_MIN = 2
L = 128
RANDOM_SEED = 30
SAMPLING_SCHEME = "MWSS"


def _compute_wavelet_covariance(
    wavelets: npt.NDArray[np.complex_],
    *,
    var_signal: float,
) -> npt.NDArray[np.float_]:
    """Computes the theoretical covariance of the wavelet coefficients."""
    covar_theory = (np.abs(wavelets) ** 2).sum(axis=1)
    return covar_theory * var_signal


def _is_ergodic(j_min: int, *, j: int = 0) -> bool:
    """
    Computes whether the function follows ergodicity.

    ergodicity fails for J_min = 0, because the scaling function will only
    cover f00. Hence <flm flm*> will be 0 in that case and the scaling
    coefficients will all be the same. So, if we do have J_min=0, we take the
    variance over all realisations instead (of course, we then won't have a
    standard deviation to compare it to the theoretical variance).
    """
    return j_min != 0 or j != 0


def axisymmetric_wavelet_covariance(
    L: int,
    B: int,
    j_min: int,
    *,
    runs: int = 100,
    var_flm: float = 1,
) -> None:
    """
    Compute theoretical covariance of wavelet coefficients.

    the covariance <Wj(omega)Wj*(omega)> is given by the following expression:
    sigma^2 Sum(l,0) |Psi^j_l0|^2

    where sigma^2 is the variance of the harmonic coefficients and Psi^j_l0
    are the harmonic coefficients of the j-th wavelet

    a similar expression applies for the scaling function coefficients

    should we use the actual variance of each realisation instead?
    """
    print(f"L={L}, B={B}, j_min={j_min}")

    # compute wavelets
    aw = AxisymmetricWavelets(L, B=B, j_min=j_min)

    # theoretical covariance
    covar_theory = _compute_wavelet_covariance(aw.wavelets, var_signal=var_flm)
    assert_equal(aw.wavelets.shape[0], covar_theory.shape[0])

    # initialise matrix
    covar_runs_shape = (runs, *covar_theory.shape)
    covar_data = np.zeros(covar_runs_shape, dtype=np.complex_)

    # set seed
    rng = default_rng(RANDOM_SEED)

    for i in range(runs):
        print(f"start run: {i+1}/{runs}")

        # Generate normally distributed random complex signal
        flm = compute_random_signal(L, rng, var_signal=var_flm)

        # compute wavelet coefficients
        wlm = axisymmetric_wavelet_forward(L, flm, aw.wavelets)

        # compute covariance from data
        for j, coefficient in enumerate(wlm):
            f_wav_j = ssht.inverse(coefficient, L, Method=SAMPLING_SCHEME)
            covar_data[i, j] = (
                f_wav_j.var() if _is_ergodic(j_min, j=j) else f_wav_j[0, 0]
            )

    # compute mean and variance
    mean_covar_data = covar_data.mean(axis=0)
    std_covar_data = covar_data.std(axis=0)

    # override for scaling function
    if not _is_ergodic(j_min):
        mean_covar_data[0] = covar_data[0].var()

    # ensure reality
    mean_covar_data = np.abs(mean_covar_data)

    # compute errors
    error_absolute = np.abs(mean_covar_data - covar_theory)
    error_in_std = error_absolute / std_covar_data

    # report errors
    for j in range(len(aw.wavelets)):
        message = (
            f"error in std: {error_in_std[j]:e}"
            if _is_ergodic(j_min, j=j)
            else f"absolute error: {error_absolute[j]:e}"
        )
        print(f"axisymmetric wavelet covariance {j}: '{message}'")


if __name__ == "__main__":
    axisymmetric_wavelet_covariance(L, B, J_MIN)
