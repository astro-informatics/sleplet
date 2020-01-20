import multiprocessing.sharedctypes as sct
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyssht as ssht
from dynaconf import settings

from ..slepian_specific import SlepianSpecific


class SlepianLimitLatLong(SlepianSpecific):
    def __init__(
        self, L: int, theta_min: float, theta_max: float, phi_min: float, phi_max: float
    ) -> None:
        self._name_ending = (
            f"_theta-{settings.THETA_MIN}-{settings.THETA_MAX}"
            f"_phi-{settings.PHI_MIN}-{settings.PHI_MAX}"
        )
        super().__init__(L, phi_min, phi_max, theta_min, theta_max)

    def _create_annotations(self) -> List[Dict]:
        annotation = []
        config = dict(arrowhead=6, ax=5, ay=5)
        p1, p2, t1, t2 = (
            np.array(self.phi_min),
            np.array(self.phi_max),
            np.array(self.theta_min),
            np.array(self.theta_max),
        )
        p3, p4, t3, t4 = (
            (p1 + 2 * p2) / 3,
            (2 * p1 + p2) / 3,
            (t1 + 2 * t2) / 3,
            (2 * t1 + t2) / 3,
        )
        for t in [t1, t2, t3, t4]:
            for p in [p1, p2, p3, p4]:
                if not ((t == t3 or t == t4) and (p == p3 or p == p4)):
                    x, y, z = ssht.s2_to_cart(t, p)
                    annotation.append(
                        {**dict(x=x, y=y, z=z, arrowcolor="black"), **config}
                    )
        return annotation

    def _create_fn_name(self) -> str:
        name = f"slepian{self._name_ending}"
        return name

    def _create_matrix_location(self) -> Path:
        location = (
            Path(__file__).resolve().parents[3]
            / "data"
            / "lat_lon"
            / f"D_L-{self.L}{self._name_ending}"
        )
        return location

    def _solve_eigenproblem(self) -> Tuple[np.ndarray, np.ndarray]:
        K = self._load_K_matrix()

        eigenvalues, eigenvectors = np.linalg.eigh(K)

        eigenvalues, eigenvectors = self._clean_evals_and_evecs(
            eigenvalues, eigenvectors
        )

        return eigenvalues, eigenvectors

    @staticmethod
    def _clean_evals_and_evecs(
        eigenvalues: np.ndarray, eigenvectors: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        need eigenvalues and eigenvectors to be in a certain format
        """
        # eigenvalues should be real
        eigenvalues = eigenvalues.real

        # Sort eigenvalues and eigenvectors in descending order of eigenvalues
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = np.conj(eigenvectors[:, idx]).T

        # ensure first element of each eigenvector is positive
        eigenvectors *= np.where(eigenvectors[:, 0] < 0, -1, 1)[:, np.newaxis]

        return eigenvalues, eigenvectors

    def _load_K_matrix(self) -> np.ndarray:
        """
        if the K matrix already exists load it
        otherwise create it and save the result
        """
        # check if matrix already exists
        if Path(self.matrix_location).exists():
            K = np.load(self.matrix_location)
        else:
            # Compute sub-integral matrix
            G = self.slepian_integral()

            # Compute Slepian matrix
            if settings.NCPU == 1:
                K = self.slepian_matrix_serial(G)
            else:
                K = self.slepian_matrix_parallel(G, settings.NCPU)

            # save to speed up for future
            if settings.SAVE_MATRICES:
                np.save(self.matrix_location, K)

        return K

    def slepian_integral(self) -> np.ndarray:
        """
        Syntax:
        G = slepian_integral()

        Output:
        G  =  Sub-integral matrix (obtained after the use of Wigner-D and
        Wigner-d functions in computing the Slepian integral) for all orders

        Description:
        This piece of code computes the sub-integral matrix (obtained after the
        use of Wigner-D and Wigner-d functions in computing the Slepian integral)
        for all orders using the formulation given in "Slepian spatialspectral
        concentration problem on the sphere: Analytical formulation for limited
        colatitude-longitude spatial region" by A. P. Bates, Z. Khalid and R. A.
        Kennedy.
        """
        G = np.zeros((4 * self.L - 3, 4 * self.L - 3), dtype=complex)

        # Using conjugate symmetry property to reduce the number of iterations
        def _helper(row, col, S):
            """
            """
            try:
                Q = (1 / (col * col - 1)) * (
                    np.exp(1j * col * self.theta_min)
                    * (1j * col * np.sin(self.theta_min) - np.cos(self.theta_min))
                    + np.exp(1j * col * self.theta_max)
                    * (np.cos(self.theta_max) - 1j * col * np.sin(self.theta_max))
                )
            except ZeroDivisionError:
                Q = 0.25 * (
                    2 * 1j * col * (self.theta_max - self.theta_min)
                    + np.exp(2 * 1j * col * self.theta_min)
                    - np.exp(2 * 1j * col * self.theta_max)
                )

            G[2 * (self.L - 1) + row, 2 * (self.L - 1) + col] = Q * S
            G[2 * (self.L - 1) - row, 2 * (self.L - 1) - col] = np.conj(
                G[2 * (self.L - 1) + row, 2 * (self.L - 1) + col]
            )

        # row = 0
        S = self.phi_max - self.phi_min
        for col in range(-2 * (self.L - 1), 1):
            _helper(0, col, S)

        # row != 0
        for row in range(-2 * (self.L - 1), 0):
            S = (1j / row) * (
                np.exp(1j * row * self.phi_min) - np.exp(1j * row * self.phi_max)
            )
            for col in range(-2 * (self.L - 1), 2 * (self.L - 1) + 1):
                _helper(row, col, S)

        return G

    def slepian_matrix_serial(self, G: np.ndarray) -> np.ndarray:
        """
        Syntax:
        K = slepian_matrix_serial(G)

        Input:
        G  =  Sub-integral matrix (obtained after the use of Wigner-D and
        Wigner-d functions in computing the Slepian integral) for all orders

        Output:
        K  =  Slepian matrix

        Description:
        This piece of code computes the Slepian matrix using the formulation
        given in "Slepian spatialspectral concentration problem on the sphere:
        Analytical formulation for limited colatitude-longitude spatial region"
        by A. P. Bates, Z. Khalid and R. A. Kennedy.
        """
        dl_array = ssht.generate_dl(np.pi / 2, self.L)
        K = np.zeros((self.L * self.L, self.L * self.L), dtype=complex)

        for l in range(self.L):
            dl = dl_array[l]

            for p in range(l + 1):
                dp = dl_array[p]
                C1 = np.sqrt((2 * l + 1) * (2 * p + 1)) / (4 * np.pi)

                for m in range(-l, l + 1):
                    for q in range(-p, p + 1):

                        row = m - q
                        C2 = (-1j) ** row
                        ind_r = 2 * (self.L - 1) + row

                        for mp in range(-l, l + 1):
                            C3 = (
                                dl[self.L - 1 + mp, self.L - 1 + m]
                                * dl[self.L - 1 + mp, self.L - 1]
                            )
                            S1 = 0

                            for qp in range(-p, p + 1):
                                col = mp - qp
                                C4 = (
                                    dp[self.L - 1 + qp, self.L - 1 + q]
                                    * dp[self.L - 1 + qp, self.L - 1]
                                )
                                ind_c = 2 * (self.L - 1) + col
                                S1 += C4 * G[ind_r, ind_c]

                            K[l * (l + 1) + m, p * (p + 1) + q] += C3 * S1

                        K[l * (l + 1) + m, p * (p + 1) + q] *= C1 * C2

        i_upper = np.triu_indices(K.shape[0])
        K[i_upper] = np.conj(K.T[i_upper])

        return K

    def slepian_matrix_parallel(self, G: np.ndarray, ncpu: int):
        """
        Syntax:
        K = slepian_matrix_parallel(G, ncpu)

        Input:
        G  =  Sub-integral matrix (obtained after the use of Wigner-D and
        Wigner-d functions in computing the Slepian integral) for all orders

        Output:
        K  =  Slepian matrix

        Description:
        This piece of code computes the Slepian matrix using the formulation
        given in "Slepian spatialspectral concentration problem on the sphere:
        Analytical formulation for limited colatitude-longitude spatial region"
        by A. P. Bates, Z. Khalid and R. A. Kennedy.
        """
        dl_array = ssht.generate_dl(np.pi / 2, self.L)

        # initialise
        real = np.zeros((self.L * self.L, self.L * self.L))
        imag = np.zeros((self.L * self.L, self.L * self.L))

        # create arrays to store final and intermediate steps
        result_r = np.ctypeslib.as_ctypes(real)
        result_i = np.ctypeslib.as_ctypes(imag)
        shared_array_r = sct.RawArray(result_r._type_, result_r)
        shared_array_i = sct.RawArray(result_i._type_, result_i)

        def func(chunk: List[int]) -> None:
            """
            calculate K matrix components for each chunk
            """
            # store real and imag parts separately
            tmp_r = np.ctypeslib.as_array(shared_array_r)
            tmp_i = np.ctypeslib.as_array(shared_array_i)

            # deal with chunk
            for l in chunk:
                dl = dl_array[l]

                for p in range(l + 1):
                    dp = dl_array[p]
                    C1 = np.sqrt((2 * l + 1) * (2 * p + 1)) / (4 * np.pi)

                    for m in range(-l, l + 1):
                        for q in range(-p, p + 1):

                            row = m - q
                            C2 = (-1j) ** row
                            ind_r = 2 * (self.L - 1) + row

                            for mp in range(-l, l + 1):
                                C3 = (
                                    dl[self.L - 1 + mp, self.L - 1 + m]
                                    * dl[self.L - 1 + mp, self.L - 1]
                                )
                                S1 = 0

                                for qp in range(-p, p + 1):
                                    col = mp - qp
                                    C4 = (
                                        dp[self.L - 1 + qp, self.L - 1 + q]
                                        * dp[self.L - 1 + qp, self.L - 1]
                                    )
                                    ind_c = 2 * (self.L - 1) + col
                                    S1 += C4 * G[ind_r, ind_c]

                                idx = (l * (l + 1) + m, p * (p + 1) + q)
                                tmp_r[idx] += np.real(C3 * S1)
                                tmp_i[idx] += np.imag(C3 * S1)

                            idx = (l * (l + 1) + m, p * (p + 1) + q)
                            real, imag = tmp_r[idx], tmp_i[idx]
                            tmp_r[idx] = real * np.real(C1 * C2) - imag * np.imag(
                                C1 * C2
                            )
                            tmp_i[idx] = real * np.imag(C1 * C2) + imag * np.real(
                                C1 * C2
                            )

        # split up L range to maximise effiency
        arr = np.arange(self.L)
        size = len(arr)
        arr[size // 2 : size] = arr[size // 2 : size][::-1]
        chunks = [np.sort(arr[i::ncpu]) for i in range(ncpu)]

        # initialise pool and apply function
        with Pool(processes=ncpu) as p:
            p.map(func, chunks)

        # retrieve real and imag components
        result_r = np.ctypeslib.as_array(shared_array_r)
        result_i = np.ctypeslib.as_array(shared_array_i)
        K = result_r + 1j * result_i

        # fill in remaining triangle section
        i_upper = np.triu_indices(K.shape[0])
        K[i_upper] = np.conj(K.T[i_upper])
        K[i_upper] = np.conj(K.T[i_upper])

        return K
