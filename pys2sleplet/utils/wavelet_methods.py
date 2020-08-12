import numpy as np
import pyssht as ssht

from pys2sleplet.utils.pys2let import s2let


def kappas_slepian_space(L: int, B: int, j_min: int) -> np.ndarray:
    """
    create tiling functions in Slepian space
    """
    kappa0, kappa = s2let.axisym_wav_l(B, L, j_min)
    kappas = np.zeros((kappa.shape[1] + 1, L ** 2))
    for ell in range(L):
        ind = ssht.elm2ind(ell, 0)
        kappas[0, ind] = kappa0[ell]
        for j in range(kappa.shape[1]):
            kappas[j + 1, ind] = kappa[ell, j]
    return kappas
