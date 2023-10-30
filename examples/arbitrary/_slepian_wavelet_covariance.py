import numpy as np
import numpy.typing as npt

import pyssht as ssht

import sleplet

SAMPLING_SCHEME = "MWSS"


def compute_slepian_wavelet_covariance(
    L: int,
    slepian_wavelets: sleplet.functions.SlepianWavelets,
    *,
    var_signal: float,
) -> npt.NDArray[np.float_]:
    """Compute the theoretical covariance of the wavelet coefficients."""
    s_p = sleplet.slepian_methods.compute_s_p_omega(L, slepian_wavelets.slepian)
    wavelets_reshape = slepian_wavelets.wavelets[
        :,
        : slepian_wavelets.slepian.N,
        np.newaxis,
        np.newaxis,
    ]
    covar_theory = (np.abs(wavelets_reshape) ** 2 * np.abs(s_p) ** 2).sum(axis=1)
    covariance = covar_theory * var_signal
    np.testing.assert_equal(
        slepian_wavelets.wavelets.shape[0],
        covariance.shape[0],
    )
    np.testing.assert_equal(
        ssht.sample_shape(slepian_wavelets.L, Method=SAMPLING_SCHEME),
        covariance.shape[1:],
    )
    return covariance
