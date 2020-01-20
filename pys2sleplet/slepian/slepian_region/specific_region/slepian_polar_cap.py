import multiprocessing.sharedctypes as sct
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pyssht as ssht
from dynaconf import settings
from scipy.special import factorial as fact

from pys2sleplet.utils.bool_methods import is_polar_gap, is_small_polar_cap
from pys2sleplet.utils.vars import PHI_MAX_DEFAULT, PHI_MIN_DEFAULT, THETA_MIN_DEFAULT

from ..slepian_specific import SlepianSpecific


class SlepianPolarCap(SlepianSpecific):
    def __init__(self, L: int, theta_max: float, order: int = 0):
        self.order = order
        self._ndots = 12
        self._name_ending = f"slepian_polar-{settings.THETA_MAX}_m-{self.order}"
        if is_polar_gap:
            self._name_ending += "_gap"
        super().__init__(
            L,
            np.deg2rad(PHI_MIN_DEFAULT),
            np.deg2rad(PHI_MAX_DEFAULT),
            np.deg2rad(THETA_MIN_DEFAULT),
            theta_max,
        )

    def _create_annotations(self) -> List[Dict]:
        annotation = []  # type: List[Dict]
        config = dict(arrowhead=6, ax=5, ay=5)

        if is_small_polar_cap(self.theta_max):
            theta_top = np.array(self.theta_max)
            for i in range(self._ndots):
                self._add_to_annotation(annotation, config, theta_top, i)

                if is_polar_gap:
                    theta_bottom = np.array(np.pi - self.theta_max)
                    for j in range(self._ndots):
                        self._add_to_annotation(
                            annotation, config, theta_bottom, j, colour="white"
                        )
        return annotation

    def _create_fn_name(self) -> str:
        name = f"slepian{self._name_ending}"
        return name

    def _create_matrix_location(self) -> Path:
        location = (
            Path(__file__).resolve().parents[3]
            / "data"
            / "polar"
            / f"D_L-{self.L}{self._name_ending}"
        )
        return location

    def _solve_eigenproblem(self) -> Tuple[np.ndarray, np.ndarray]:
        emm = self._create_emm_vec()

        Dm = self._load_Dm_matrix(emm)

        # solve eigenproblem for order 'm'
        eigenvalues, gl = np.linalg.eigh(Dm)

        eigenvalues, eigenvectors = self._clean_evals_and_evecs(eigenvalues, gl, emm)

        return eigenvalues, eigenvectors

    def _clean_evals_and_evecs(
        self, eigenvalues: np.ndarray, gl: np.ndarray, emm: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        need eigenvalues and eigenvectors to be in a certain format
        """
        # eigenvalues should be real
        eigenvalues = eigenvalues.real

        # Sort eigenvalues and eigenvectors in descending order of eigenvalues
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        gl = np.conj(gl[:, idx])

        # put back in full D space for harmonic transform
        emm = emm[: self.L * self.L]
        ind = np.tile(emm == self.order, (self.L - abs(self.order), 1))
        eigenvectors = np.zeros(
            (self.L - abs(self.order), self.L * self.L), dtype=complex
        )
        eigenvectors[ind] = gl.T.flatten()

        # ensure first element of each eigenvector is positive
        eigenvectors *= np.where(eigenvectors[:, 0] < 0, -1, 1)[:, np.newaxis]

        # if -ve 'm' find orthogonal eigenvectors to +ve 'm' eigenvectors
        if self.order < 0:
            eigenvectors *= 1j

        return eigenvalues, eigenvectors

    def _load_Dm_matrix(self, emm: np.ndarray) -> np.ndarray:
        """
        if the Dm matrix already exists load it
        otherwise create it and save the result
        """
        # check if matrix already exists
        if Path(self.matrix_location).exists():
            Dm = np.load(self.matrix_location)
        else:
            # create Legendre polynomials table
            Plm = ssht.create_ylm(self.theta_max, 0, 2 * self.L).real.reshape(-1)
            ind = emm == 0
            l = np.arange(2 * self.L).reshape(1, -1)
            Pl = np.sqrt((4 * np.pi) / (2 * l + 1)) * Plm[ind]
            P = np.concatenate((Pl, l))

            # Computing order 'm' Slepian matrix
            if settings.NCPU == 1:
                Dm = self.Dm_matrix_serial(abs(self.order), P)
            else:
                Dm = self.Dm_matrix_parallel(abs(self.order), P, settings.NCPU)

            # save to speed up for future
            if settings.SAVE_MATRICES:
                np.save(self.matrix_location, Dm)

        return Dm

    def _create_emm_vec(self) -> np.ndarray:
        """
        create emm vector for eigenproblem
        """
        emm = np.zeros(2 * self.L * 2 * self.L)
        k = 0

        for l in range(2 * self.L):
            M = 2 * l + 1
            emm[k : k + M] = np.arange(-l, l + 1)
            k = k + M

        return emm

    @property
    def order(self) -> int:
        return self.__order

    @order.setter
    def order(self, var: int) -> None:
        # check order is in correct range
        if abs(var) >= self.L:
            raise ValueError(
                f"Slepian polar cap order magnitude should be less than {self.L}"
            )
        self.__order = var

    @staticmethod
    def Wigner3j(l1: int, l2: int, l3: int, m1: int, m2: int, m3: int) -> float:
        """
        Syntax:
        s = Wigner3j (l1, l2, l3, m1, m2, m3)

        Input:
        l1  =  first degree in Wigner 3j symbol
        l2  =  second degree in Wigner 3j symbol
        l3  =  third degree in Wigner 3j symbol
        m1  =  first order in Wigner 3j symbol
        m2  =  second order in Wigner 3j symbol
        m3  =  third order in Wigner 3j symbol

        Output:
        s  =  Wigner 3j symbol for l1,m1; l2,m2; l3,m3

        Description:
        Computes Wigner 3j symbol using Racah formula
        """
        if (
            2 * l1 != np.floor(2 * l1)
            or 2 * l2 != np.floor(2 * l2)
            or 2 * l3 != np.floor(2 * l3)
            or 2 * m1 != np.floor(2 * m1)
            or 2 * m2 != np.floor(2 * m2)
            or 2 * m3 != np.floor(2 * m3)
        ):
            raise Exception("Arguments must either be integer or half-integer!")

        if (
            m1 + m2 + m3 != 0
            or l3 < abs(l1 - l2)
            or l3 > (l1 + l2)
            or abs(m1) > abs(l1)
            or abs(m2) > abs(l2)
            or abs(m3) > abs(l3)
            or l1 + l2 + l3 != np.floor(l1 + l2 + l3)
        ):
            s = 0
        else:
            t1 = l2 - l3 - m1
            t2 = l1 - l3 + m2
            t3 = l1 + l2 - l3
            t4 = l1 - m1
            t5 = l2 + m2

            tmin = max(0, max(t1, t2))
            tmax = min(t3, min(t4, t5))

            s = 0
            # sum is over all those t for which the following factorials have
            # non-zero arguments.
            for t in range(tmin, tmax + 1):
                s += (-1) ** t / (
                    fact(t, exact=False)
                    * fact(t - t1, exact=False)
                    * fact(t - t2, exact=False)
                    * fact(t3 - t, exact=False)
                    * fact(t4 - t, exact=False)
                    * fact(t5 - t, exact=False)
                )

            triangle_coefficient = (
                fact(l1 + l2 - l3, exact=False)
                * fact(l1 - l2 + l3, exact=False)
                * fact(-l1 + l2 + l3, exact=False)
                / fact(l1 + l2 + l3 + 1, exact=False)
            )

            s *= (
                np.float_power(-1, l1 - l2 - m3)
                * np.sqrt(triangle_coefficient)
                * np.sqrt(
                    fact(l1 + m1, exact=False)
                    * fact(l1 - m1, exact=False)
                    * fact(l2 + m2, exact=False)
                    * fact(l2 - m2, exact=False)
                    * fact(l3 + m3, exact=False)
                    * fact(l3 - m3, exact=False)
                )
            )

        return s

    def Dm_matrix_serial(self, m: int, P: np.ndarray) -> np.ndarray:
        """
        Syntax:
        Dm = Dm_matrix_serial(m, P)

        Input:
        m  =  order
        P(:,1)  =  Pl = Legendre Polynomials column vector for l = 0 : L-1
        P(:,2)  =  ell values vector

        Output:
        Dm = (L - m) square Slepian matrix for order m

        Description:
        This piece of code computes the Slepian matrix, Dm, for order m and all
        degrees, using the formulation given in "Spatiospectral Concentration on
        a Sphere" by F.J. Simons, F.A. Dahlen and M.A. Wieczorek.
        """
        Dm = np.zeros((self.L - m, self.L - m))
        Pl, ell = P
        lvec = np.arange(m, self.L)

        for i in range(self.L - m):
            l = lvec[i]
            for j in range(i, self.L - m):
                p = lvec[j]
                c = 0
                for n in range(abs(l - p), l + p + 1):
                    if n - 1 == -1:
                        A = 1
                    else:
                        A = Pl[ell == n - 1]
                    c += (
                        self.Wigner3j(l, n, p, 0, 0, 0)
                        * self.Wigner3j(l, n, p, m, 0, -m)
                        * (A - Pl[ell == n + 1])
                    )
                Dm[i, j] = (
                    self.polar_gap_modification(l, p)
                    * np.sqrt((2 * l + 1) * (2 * p + 1))
                    * c
                )
                Dm[j, i] = Dm[i, j]

        Dm *= (-1) ** m / 2

        return Dm

    def Dm_matrix_parallel(self, m: int, P: np.ndarray, ncpu: int) -> np.ndarray:
        """
        Syntax:
        Dm = Dm_matrix_parallel(m, P, ncpu)

        Input:
        m  =  order
        P(:,1)  =  Pl = Legendre Polynomials column vector for l = 0 : L-1
        P(:,2)  =  ell values vector

        Output:
        Dm = (L - m) square Slepian matrix for order m

        Description:
        This piece of code computes the Slepian matrix, Dm, for order m and all
        degrees, using the formulation given in "Spatiospectral Concentration on
        a Sphere" by F.J. Simons, F.A. Dahlen and M.A. Wieczorek.
        """
        Dm = np.zeros((self.L - m, self.L - m))
        Pl, ell = P
        lvec = np.arange(m, self.L)

        # create arrays to store final and intermediate steps
        result = np.ctypeslib.as_ctypes(Dm)
        shared_array = sct.RawArray(result._type_, result)

        def func(chunk: List[int]) -> None:
            """
            calculate D matrix components for each chunk
            """
            # temporary store
            tmp = np.ctypeslib.as_array(shared_array)

            # deal with chunk
            for i in chunk:
                l = lvec[i]
                for j in range(i, self.L - m):
                    p = lvec[j]
                    c = 0
                    for n in range(abs(l - p), l + p + 1):
                        if n - 1 == -1:
                            A = 1
                        else:
                            A = Pl[ell == n - 1]
                        c += (
                            self.Wigner3j(l, n, p, 0, 0, 0)
                            * self.Wigner3j(l, n, p, m, 0, -m)
                            * (A - Pl[ell == n + 1])
                        )
                    tmp[i, j] = (
                        self.polar_gap_modification(l, p)
                        * np.sqrt((2 * l + 1) * (2 * p + 1))
                        * c
                    )
                    tmp[j, i] = tmp[i, j]

        # split up L range to maximise effiency
        arr = np.arange(self.L - m)
        size = len(arr)
        arr[size // 2 : size] = arr[size // 2 : size][::-1]
        chunks = [np.sort(arr[i::ncpu]) for i in range(ncpu)]

        # initialise pool and apply function
        with Pool(processes=ncpu) as p:
            p.map(func, chunks)

        # retrieve from parallel function
        Dm = np.ctypeslib.as_array(shared_array) * (-1) ** m / 2

        return Dm

    @staticmethod
    def polar_gap_modification(ell1: int, ell2: int) -> int:
        factor = 1 + is_polar_gap * (-1) ** (ell1 + ell2)
        return factor

    def _add_to_annotation(
        self,
        annotation: Union[List[Dict], List],
        config: Dict[str, int],
        theta: np.ndarray,
        i: int,
        colour: str = "black",
    ) -> None:
        """
        add to annotation list for given theta
        """
        phi = np.array(2 * np.pi / self._ndots * (i + 1))
        x, y, z = ssht.s2_to_cart(theta, phi)
        annotation.append({**dict(x=x, y=y, z=z, arrowcolor=colour), **config})
