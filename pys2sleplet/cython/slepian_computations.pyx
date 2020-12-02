import numpy as np

from libc.stdio cimport printf
from libc.stdlib cimport calloc, malloc


cdef extern from "ssht/ssht.h" nogil:
    ctypedef enum ssht_dl_method_t:
        SSHT_DL_RISBO, SSHT_DL_TRAPANI

    void ssht_core_mw_inverse_sov_sym_ss(
        double complex *f,
        const double complex *flm,
        int L,
        int spin,
        ssht_dl_method_t dl_method,
        int verbosity
    )


def create_arbitrary_D_matrix(
    int L,
    int resolution,
    double[:,::1] weight,
    unsigned char[:,::1] mask,
):
    """
    solves the eigenproblem for a given L, resolution & arbitrary region
    """
    cdef int ell
    cdef double complex[:,::1] D
    D = np.zeros((L * L, L * L), dtype=np.complex_)

    for ell in range(L * L):
        print(f"ell: {ell}")
        _fill_D_matrix(D, L, resolution, weight, mask, ell)

    return D.base

cdef void _fill_D_matrix(
     double complex[:,::1] D,
     int L,
     int resolution,
     double[:,::1] weight,
     unsigned char[:,::1] mask,
     int i
) nogil:
    """
    fills all the indices for the given ell
    """
    cdef int m_i, j, ell_j, m_j, k

    # fill in diagonal components
    D[i][i] = _compute_integral(L, resolution, weight, mask, i, i)
    _, m_i = _ssht_ind2elm(i)

    for j in range(i + 1, L * L):
        ell_j, m_j = _ssht_ind2elm(j)
        # if possible to use previous calculations
        if m_i == 0 and m_j != 0 and ell_j < L:
            # if positive m then use conjugate relation
            if m_j > 0:
                D[j][i] = _compute_integral(L, resolution, weight, mask, j, i)
                k = _ssht_elm2ind(ell_j, -m_j)
                D[k][i] = (-1) ** m_j * D[j][i]
        else:
            D[j][i] = _compute_integral(L, resolution, weight, mask, j, i)


cdef double complex _compute_integral(
    int L,
    int resolution,
    double[:,::1] weight,
    unsigned char[:,::1] mask,
    int i,
    int j
) nogil:
    """
    calculates the D integral between two spherical harmonics
    """
    return _integrate_region_sphere(
        resolution,
        _ssht_inverse(
            _create_spherical_harmonic(resolution, i), resolution
        ),
        _ssht_inverse(
            _create_spherical_harmonic(resolution, j), resolution
        ),
        weight,
        mask,
    )


cdef double complex * _create_spherical_harmonic(
    int L,
    int ind
) nogil:
    """
    create a spherical harmonic in harmonic space for the given index
    """
    cdef double complex *flm = NULL
    flm = <double complex *> calloc(L * L, sizeof(double complex))
    flm[ind] = 1
    return flm


cdef double complex _integrate_region_sphere(
    int L,
    double complex *f,
    double complex *g,
    double[:,::1] weight,
    unsigned char[:,::1] mask
) nogil:
    """
    computes the integration for a region of the sphere
    """
    cdef int i, j
    cdef double complex integrand = 0
    for i in range(L + 1):
        for j in range(2 * L):
            integrand += (
                f[(2 * L) * i + j] *
                g[(2 * L) * i + j].conjugate() *
                weight[i][j] * mask[i][j]
            )
    return integrand

cdef double complex * _ssht_inverse(
    double complex *flm,
    int L
) nogil:
    """
    reimplements the complex mwss sampling from ssht
    """
    cdef ssht_dl_method_t dl_method = SSHT_DL_RISBO
    cdef double complex *f = NULL
    f = <double complex *> malloc((L + 1) * (2 * L) * sizeof(double complex))
    ssht_core_mw_inverse_sov_sym_ss(f, flm, L, 0, dl_method, 0)
    return f

cdef inline int _ssht_elm2ind(
    int ell,
    int m
) nogil:
    """
    reimplements the elm2ind method from ssht
    """
    return ell * ell + ell + m

cdef inline (int, int) _ssht_ind2elm(
    int ind
) nogil:
    """
    reimplements the ind2elm method from ssht
    """
    cdef int ell, m
    ell = _sqrt(ind)
    m = ind - ell * ell - ell
    return ell, m

cdef inline int _sqrt(
    int n
) nogil:
    """
    computes the square root floored
    """
    cdef int square = 1, delta = 3
    while square <= n:
        square += delta
        delta  += 2
    return delta // 2 - 1
