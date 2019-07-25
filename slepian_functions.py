from plotting import Plotting
import matplotlib
import multiprocessing as mp
import multiprocessing.sharedctypes as sct
import numpy as np
import os
import scipy.io as sio
import sys
from typing import List, Tuple

sys.path.append(os.path.join(os.environ["SSHT"], "src", "python"))
import pyssht as ssht


class SlepianFunctions:
    def __init__(
        self, phi_min: int, phi_max: int, theta_min: int, theta_max: int, config: dict
    ):
        samples = 1 * config["L"]
        theta, phi = ssht.sample_positions(samples, Method="MWSS")
        thetas, phis = ssht.sample_positions(samples, Grid=True, Method="MWSS")
        phi_mask = np.where(
            (phi >= np.deg2rad(phi_min)) & (phi <= np.deg2rad(phi_max))
        )[0]
        theta_mask = np.where(
            (theta >= np.deg2rad(theta_min)) & (theta <= np.deg2rad(theta_max))
        )[0]
        ylm = ssht.create_ylm(thetas, phis, config["L"])
        self.delta_phi = np.mean(np.ediff1d(phi))
        self.delta_theta = np.mean(np.ediff1d(theta))
        self.L = config["L"]
        self.location = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__))
        )
        self.N = config["L"] * config["L"]
        self.ncpu = config["ncpu"]
        self.phi_max = phi_max
        self.phi_max_is_default = phi_max == 360
        self.phi_min = phi_min
        self.phi_min_is_default = phi_min == 0
        self.plotting = Plotting(
            auto_open=config["auto_open"], save_fig=config["save_fig"]
        )
        self.resolution = self.plotting.calc_resolution(config["L"])
        self.theta_max = theta_max
        self.theta_max_is_default = theta_max == 180
        self.theta_min = theta_min
        self.theta_min_is_default = theta_min == 0
        self.thetas = thetas[theta_mask[:, np.newaxis], phi_mask]
        self.ylm = ylm[:, theta_mask[:, np.newaxis], phi_mask]
        self.plotting.missing_key(config, "annotation", True)
        self.plotting.missing_key(config, "type", None)
        self.eigen_values, self.eigen_vectors = self.eigen_problem()

    # ----------------------------------------
    # ---------- D matrix functions ----------
    # ----------------------------------------

    def f(self, i: int, j: int) -> np.ndarray:
        """
        function to integrate using quadrature
        """
        f = self.ylm[i] * np.conj(self.ylm[j])
        return f

    def w(self) -> np.ndarray:
        """
        weight for the summation
        """
        w = np.sin(self.thetas) * self.delta_theta * self.delta_phi
        return w

    def D_integral(self, i: int, j: int) -> complex:
        """
        function which performs a summation to calculate the integral
        """
        F = np.sum(self.f(i, j) * self.w())
        return F

    def D_matrix(self) -> np.ndarray:
        """
        calculates the D matrix for the eigenproblem
        serial method easier to debug and more efficient for one core
        """
        if self.ncpu == 1:
            D = self.D_matrix_serial()
        else:
            D = self.D_matrix_parallel()
        return D

    def D_matrix_serial(self) -> np.ndarray:
        """
        serial method to calculate D matrix
        """
        # initialise
        D = np.zeros((self.N, self.N), dtype=complex)

        for i in range(self.N):
            # fill in diagonal components
            D[i][i] = self.D_integral(i, i)
            _, m_i = ssht.ind2elm(i)
            for j in range(i + 1, self.N):
                ell_j, m_j = ssht.ind2elm(j)
                # if possible to use previous calculations
                if m_i == 0 and m_j != 0 and ell_j < self.L:
                    # if positive m then use conjugate relation
                    if m_j > 0:
                        D[i][j] = self.D_integral(i, j)
                        D[j][i] = np.conj(D[i][j])
                        k = ssht.elm2ind(ell_j, -m_j)
                        D[i][k] = (-1) ** m_j * np.conj(D[i][j])
                        D[k][i] = np.conj(D[i][k])
                else:
                    D[i][j] = self.D_integral(i, j)
                    D[j][i] = np.conj(D[i][j])
        return D

    def D_matrix_parallel(self) -> complex:
        """
        parallel method to calculate D matrix
        """
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
                integral = self.D_integral(i, i)
                tmp_r[i][i] = integral.real
                tmp_i[i][i] = integral.imag
                _, m_i = ssht.ind2elm(i)

                for j in range(i + 1, self.N):
                    ell_j, m_j = ssht.ind2elm(j)
                    # if possible to use previous calculations
                    if m_i == 0 and m_j != 0 and ell_j < self.L:
                        # if positive m then use conjugate relation
                        if m_j > 0:
                            integral = self.D_integral(i, j)
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
                        integral = self.D_integral(i, j)
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

    def eigen_problem(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        calculates the eigenvalues and eigenvectors of the D matrix
        """
        # numpy binary filename
        filename = os.path.join(
            self.location,
            "npy",
            "d_matrix",
            (f"D_L-{self.L}{self.filename_angle()}.npy"),
        )

        # check if file of D matrix already exists
        if os.path.exists(filename):
            D = np.load(filename)
        else:
            D = self.D_matrix()
            # save to speed up for future
            np.save(filename, D)

        A = (np.deg2rad(self.phi_max) - np.deg2rad(self.phi_min)) * (
            np.cos(np.deg2rad(self.theta_min)) - np.cos(np.deg2rad(self.theta_max))
        )
        N = self.L * self.L * A / (4 * np.pi)
        print(N, np.trace(D))
        # np.testing.assert_allclose(N, np.trace(D), rtol=1e-2)

        # order eigenvalues in descending order
        eigen_values, eigen_vectors = np.linalg.eigh(D)
        eigen_values = eigen_values[::-1]
        eigen_vectors = eigen_vectors[::-1]
        return eigen_values, eigen_vectors

    # ----------------------------------------
    # ---------- plotting function -----------
    # ----------------------------------------

    def plot(self, rank: int) -> None:
        """
        master plotting method
        """
        # setup
        if self.eigen_values.max() != 0:
            e_val = np.abs(self.eigen_values[rank] / (self.eigen_values.max()))
        else:
            e_val = 1
        print(f"Eigenvalue {rank + 1}: {e_val:e}")
        filename = f"slepian-{rank + 1}_L-{self.L}{self.filename_angle()}_res-{self.resolution}_"
        flm = self.eigen_vectors[rank]

        # boost resolution
        if self.resolution != self.L:
            flm = self.plotting.resolution_boost(flm, self.L, self.resolution)

        # inverse & plot
        f = ssht.inverse(flm, self.resolution, Method="MWSS")

        # check for plotting type
        if self.plotting.type == "real":
            f = f.real
        elif self.plotting.type == "imag":
            f = f.imag
        elif self.plotting.type == "abs":
            f = np.abs(f)
        elif self.plotting.type == "sum":
            f = f.real + f.imag

        # do plot
        filename += self.plotting.type
        self.plotting.plotly_plot(
            f, self.resolution, filename, self.annotations(), cmap=self.colourmap()
        )

    # -----------------------------------------------
    # ---------- plotting helper functions ----------
    # -----------------------------------------------

    def annotations(self) -> List[dict]:
        """
        annotations for the plotly plot
        """
        if self.plotting.annotation:
            annotation = []
            # check if dealing with polar cap
            if (
                self.phi_min_is_default
                and self.phi_max_is_default
                and self.theta_min_is_default
                and self.theta_max <= 45
            ):
                config = dict(arrowcolor="black", arrowhead=6, ax=5, ay=5)
                ndots = 12
                theta = np.array(np.deg2rad(self.theta_max))
                for i in range(ndots):
                    phi = np.array(2 * np.pi / ndots * (i + 1))
                    x, y, z = ssht.s2_to_cart(theta, phi)
                    annotation.append({**dict(x=x, y=y, z=z), **config})
        else:
            annotation = []
        return annotation

    def filename_angle(self) -> str:
        """
        middle part of filename
        """
        # initialise filename
        filename = ""

        # if phi min is not default
        if not self.phi_min_is_default:
            filename += f"_pmin-{self.phi_min}"

        # if phi max is not default
        if not self.phi_max_is_default:
            filename += f"_pmax-{self.phi_max}"

        # if theta min is not default
        if not self.theta_min_is_default:
            filename += f"_tmin-{self.theta_min}"

        # if theta max is not default
        if not self.theta_max_is_default:
            filename += f"_tmax-{self.theta_max}"
        return filename

    def colourmap(self):
        matfile = os.path.join(self.location, "data", "kelim")
        mat_contents = sio.loadmat(matfile)
        cmap = matplotlib.colors.ListedColormap(mat_contents["kelim"])
        return cmap
