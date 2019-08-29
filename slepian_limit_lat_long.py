import numpy as np
import os
import sys

sys.path.append(os.path.join(os.environ["SSHT"], "src", "python"))
import pyssht as ssht


class SlepianLimitLatLong:
    def __init__(self, L, phi_min, phi_max, theta_min, theta_max):
        self.L = L
        self.phi_min = np.deg2rad(phi_min)
        self.phi_max = np.deg2rad(phi_max)
        self.theta_min = np.deg2rad(theta_min)
        self.theta_max = np.deg2rad(theta_max)

    def slepian_integral(self):
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
        for col in range(-2 * (self.L - 1), 1):
            S = self.phi_max - self.phi_min
            _helper(0, col, S)

        # row != 0
        for row in range(-2 * (self.L - 1), 0):
            for col in range(-2 * (self.L - 1), 2 * (self.L - 1) + 1):
                S = (1j / row) * (
                    np.exp(1j * row * self.phi_min) - np.exp(1j * row * self.phi_max)
                )
                _helper(row, col, S)

        return G

    def slepian_matrix(self, G):
        """
        Syntax:
        K = slepain_matrix(G)

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
        K = np.zeros((self.L * self.L, self.L * self.L), dtype=complex)
        dl_array = ssht.generate_dl(np.pi / 2, self.L)

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

                            K[l * (l + 1) + m, p * (p + 1) + q] = (
                                K[l * (l + 1) + m, p * (p + 1) + q] + C3 * S1
                            )

                        K[l * (l + 1) + m, p * (p + 1) + q] = (
                            C1 * C2 * K[l * (l + 1) + m, p * (p + 1) + q]
                        )

        i_upper = np.triu_indices(K.shape[0])
        K[i_upper] = np.conj(K.T[i_upper])

        return K

    def eigenproblem(self):
        """
        """
        # Compute sub-integral matrix
        G = self.slepian_integral()

        # Compute Slepian matrix
        K = self.slepian_matrix(G)

        # solve eigenproblem
        eigenvalues, eigenvectors = np.linalg.eig(K)

        # eigenvalues should be real
        eigenvalues = eigenvalues.real

        # Sort eigenvalues and eigenvectors in descending order of eigenvalues
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        return eigenvalues, eigenvectors
