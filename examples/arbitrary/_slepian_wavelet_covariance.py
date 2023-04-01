import numpy as np
import pyssht as ssht
from numpy import typing as npt
from numpy.testing import assert_equal

from sleplet.functions import SlepianWavelets
from sleplet.slepian_methods import compute_s_p_omega

SAMPLING_SCHEME = "MWSS"


def compute_slepian_wavelet_covariance(
    L: int,
    slepian_wavelets: SlepianWavelets,
    *,
    var_signal: float,
) -> npt.NDArray[np.float_]:
    """Computes the theoretical covariance of the wavelet coefficients."""
    s_p = compute_s_p_omega(L, slepian_wavelets.slepian)
    wavelets_reshape = slepian_wavelets.wavelets[
        :,
        : slepian_wavelets.slepian.N,
        np.newaxis,
        np.newaxis,
    ]
    covar_theory = (np.abs(wavelets_reshape) ** 2 * np.abs(s_p) ** 2).sum(axis=1)
    covariance = covar_theory * var_signal
    assert_equal(slepian_wavelets.wavelets.shape[0], covariance.shape[0])
    assert_equal(
        ssht.sample_shape(slepian_wavelets.L, Method=SAMPLING_SCHEME),
        covariance.shape[1:],
    )
    return covariance
