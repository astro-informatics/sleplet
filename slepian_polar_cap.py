from multiprocessing import Pool
import multiprocessing.sharedctypes as sct
import numpy as np
import os
from scipy.special import factorial
import sys
from typing import List

sys.path.append(os.path.join(os.environ["SSHT"], "src", "python"))
import pyssht as ssht


class SlepianPolarCap:
    def __init__(
        self, L, theta_max: int, binary: str, ncpu: int = 1, polar_gap: bool = False
    ) -> None:
        self.binary = binary
        self.L = L
        self.ncpu = ncpu
        self.theta_max = np.deg2rad(theta_max)
        self.polar_gap = polar_gap

    @staticmethod
    def Wigner3j(l1, l2, l3, m1, m2, m3):
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
                    factorial(t, exact=False)
                    * factorial(t - t1, exact=False)
                    * factorial(t - t2, exact=False)
                    * factorial(t3 - t, exact=False)
                    * factorial(t4 - t, exact=False)
                    * factorial(t5 - t, exact=False)
                )

            triangle_coefficient = (
                factorial(l1 + l2 - l3, exact=False)
                * factorial(l1 - l2 + l3, exact=False)
                * factorial(-l1 + l2 + l3, exact=False)
                / factorial(l1 + l2 + l3 + 1, exact=False)
            )

            s *= (
                np.float_power(-1, l1 - l2 - m3)
                * np.sqrt(triangle_coefficient)
                * np.sqrt(
                    factorial(l1 + m1, exact=False)
                    * factorial(l1 - m1, exact=False)
                    * factorial(l2 + m2, exact=False)
                    * factorial(l2 - m2, exact=False)
                    * factorial(l3 + m3, exact=False)
                    * factorial(l3 - m3, exact=False)
                )
            )

        return s

    def Dm_matrix_serial(self, m, P):
        """
        Syntax:
        Dm = Dm_matrix (m, P)

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

    def Dm_matrix_parallel(self, m, P):
        """
        Syntax:
        Dm = Dm_matrix (m, P)

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

        # ensure function declared before multiprocessing pool
        global func

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
        chunks = [np.sort(arr[i :: self.ncpu]) for i in range(self.ncpu)]

        # initialise pool and apply function
        with Pool(processes=self.ncpu) as p:
            p.map(func, chunks)

        # retrieve from parallel function
        Dm = np.ctypeslib.as_array(shared_array) * (-1) ** m / 2

        return Dm

    def polar_gap_modification(self, ell1, ell2):
        return 1 + self.polar_gap * (-1) ** (ell1 + ell2)

    def eigenproblem(self, m):
        """
        """
        # create emm vector
        emm = np.zeros(2 * self.L * 2 * self.L)
        k = 0
        for l in range(2 * self.L):
            M = 2 * l + 1
            emm[k : k + M] = np.arange(-l, l + 1)
            k = k + M

        # check if matrix already exists
        if os.path.exists(self.binary):
            Dm = np.load(self.binary)
        else:
            # create Legendre polynomials table
            Plm = ssht.create_ylm(self.theta_max, 0, 2 * self.L).real.reshape(-1)
            ind = emm == 0
            l = np.arange(2 * self.L).reshape(1, -1)
            Pl = np.sqrt((4 * np.pi) / (2 * l + 1)) * Plm[ind]
            P = np.concatenate((Pl, l))

            # Computing order 'm' Slepian matrix
            if self.ncpu == 1:
                Dm = self.Dm_matrix_serial(abs(m), P)
            else:
                Dm = self.Dm_matrix_parallel(abs(m), P)

            # save to speed up for future
            np.save(self.binary, Dm)

        # solve eigenproblem for order 'm'
        eigenvalues, gl = np.linalg.eig(Dm)

        # eigenvalues should be real
        eigenvalues = eigenvalues.real

        # Sort eigenvalues and eigenvectors in descending order of eigenvalues
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        gl = np.conj(gl[:, idx])

        # put back in full D space for harmonic transform
        emm = emm[: self.L * self.L]
        ind = np.tile(emm == m, (self.L - abs(m), 1))
        glm = np.zeros((self.L - abs(m), self.L * self.L), dtype=complex)
        glm[ind] = gl.T.flatten()

        # if -ve 'm' find orthogonal eigenvectors to +ve 'm' eigenvectors
        if m < 0:
            eigenvectors = []
            for flm in glm:
                eigenvectors.append(ssht.rotate_flms(flm, -np.pi / 2, 0, 0, self.L))
            eigenvectors = np.array(eigenvectors)
        else:
            eigenvectors = glm

        return eigenvalues, eigenvectors
