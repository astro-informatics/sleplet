from helper import calc_samples
import multiprocessing as mp
import multiprocessing.sharedctypes as sct
import numpy as np
import pyssht as ssht
from typing import List, Tuple


class SlepianArbitrary:
    def __init__(
        self,
        L: int,
        phi_min: int,
        phi_max: int,
        theta_min: int,
        theta_max: int,
        ncpu: int = 1,
    ) -> None:
        samples = calc_samples(L)
        theta, phi = ssht.sample_positions(samples, Method="MWSS")
        thetas, phis = ssht.sample_positions(samples, Grid=True, Method="MWSS")
        phi_mask = np.where(
            (phi >= np.deg2rad(phi_min)) & (phi <= np.deg2rad(phi_max))
        )[0]
        theta_mask = np.where(
            (theta >= np.deg2rad(theta_min)) & (theta <= np.deg2rad(theta_max))
        )[0]
        ylm = ssht.create_ylm(thetas, phis, L)
        self.delta_phi = np.mean(np.ediff1d(phi))
        self.delta_theta = np.mean(np.ediff1d(theta))
        self.L = L
        self.N = L * L
        self.ncpu = ncpu
        self.thetas = thetas[theta_mask[:, np.newaxis], phi_mask]
        self.ylm = ylm[:, theta_mask[:, np.newaxis], phi_mask]

    def f(self, i: int, j: int) -> np.ndarray:
        f = self.ylm[i] * np.conj(self.ylm[j])
        return f

    def w(self) -> np.ndarray:
        w = np.sin(self.thetas) * self.delta_theta * self.delta_phi
        return w

    def integral(self, i: int, j: int) -> complex:
        F = np.sum(self.f(i, j) * self.w())
        return F

    def matrix_serial(self) -> np.ndarray:
        # initialise
        D = np.zeros((self.N, self.N), dtype=complex)

        for i in range(self.N):
            # fill in diagonal components
            D[i][i] = self.integral(i, i)
            _, m_i = ssht.ind2elm(i)
            for j in range(i + 1, self.N):
                ell_j, m_j = ssht.ind2elm(j)
                # if possible to use previous calculations
                if m_i == 0 and m_j != 0 and ell_j < self.L:
                    # if positive m then use conjugate relation
                    if m_j > 0:
                        D[i][j] = self.integral(i, j)
                        D[j][i] = np.conj(D[i][j])
                        k = ssht.elm2ind(ell_j, -m_j)
                        D[i][k] = (-1) ** m_j * np.conj(D[i][j])
                        D[k][i] = np.conj(D[i][k])
                else:
                    D[i][j] = self.integral(i, j)
                    D[j][i] = np.conj(D[i][j])
        return D

    def matrix_parallel(self):
        # initialise
        real = np.zeros((self.N, self.N))
        imag = np.zeros((self.N, self.N))

        # create arrays to store final and intermediate steps
        result_r = np.ctypeslib.as_ctypes(real)
        result_i = np.ctypeslib.as_ctypes(imag)
        shared_array_r = sct.RawArray(result_r._type_, result_r)
        shared_array_i = sct.RawArray(result_i._type_, result_i)

        # ensure function declared before multiprocessing pool
        global func

        def func(chunk: List[int]) -> None:
            """
            calculate D matrix components for each chunk
            """
            # store real and imag parts separately
            tmp_r = np.ctypeslib.as_array(shared_array_r)
            tmp_i = np.ctypeslib.as_array(shared_array_i)

            # deal with chunk
            for i in chunk:
                # fill in diagonal components
                integral = self.integral(i, i)
                tmp_r[i][i] = integral.real
                tmp_i[i][i] = integral.imag
                _, m_i = ssht.ind2elm(i)

                for j in range(i + 1, self.N):
                    ell_j, m_j = ssht.ind2elm(j)
                    # if possible to use previous calculations
                    if m_i == 0 and m_j != 0 and ell_j < self.L:
                        # if positive m then use conjugate relation
                        if m_j > 0:
                            integral = self.integral(i, j)
                            tmp_r[i][j] = integral.real
                            tmp_i[i][j] = integral.imag
                            tmp_r[j][i] = tmp_r[i][j]
                            tmp_i[j][i] = -tmp_i[i][j]
                            k = ssht.elm2ind(ell_j, -m_j)
                            tmp_r[i][k] = (-1) ** m_j * tmp_r[i][j]
                            tmp_i[i][k] = (-1) ** (m_j + 1) * tmp_i[i][j]
                            tmp_r[k][i] = tmp_r[i][k]
                            tmp_i[k][i] = -tmp_i[i][k]
                    else:
                        integral = self.integral(i, j)
                        tmp_r[i][j] = integral.real
                        tmp_i[i][j] = integral.imag
                        tmp_r[j][i] = tmp_r[i][j]
                        tmp_i[j][i] = -tmp_i[i][j]

        # split up L range to maximise effiency
        arr = np.arange(self.N)
        size = len(arr)
        arr[size // 2 : size] = arr[size // 2 : size][::-1]
        chunks = [np.sort(arr[i :: self.ncpu]) for i in range(self.ncpu)]

        # initialise pool and apply function
        with mp.Pool(processes=self.ncpu) as p:
            p.map(func, chunks)

        # retrieve real and imag components
        result_r = np.ctypeslib.as_array(shared_array_r)
        result_i = np.ctypeslib.as_array(shared_array_i)

        return result_r + 1j * result_i

    def eigenproblem(self) -> Tuple[np.ndarray, np.ndarray]:
        # Compute Slepian matrix
        if self.ncpu == 1:
            D = self.matrix_serial()
        else:
            D = self.matrix_parallel()

        # solve eigenproblem
        eigenvalues, eigenvectors = np.linalg.eigh(D)

        # eigenvalues should be real
        eigenvalues = eigenvalues.real

        # Sort eigenvalues and eigenvectors in descending order of eigenvalues
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = np.conj(eigenvectors[:, idx]).T

        # ensure first element of each eigenvector is positive
        eigenvectors *= np.where(eigenvectors[:, 0] < 0, -1, 1)[:, np.newaxis]

        # find repeating eigenvalues and ensure orthorgonality
        pairs = np.where(np.abs(np.diff(eigenvalues)) < 1e-14)[0] + 1
        eigenvectors[pairs] *= 1j

        return eigenvalues, eigenvectors
