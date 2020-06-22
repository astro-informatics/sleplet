from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numexpr as ne
import numpy as np
import pyssht as ssht
from multiprocess import Pool
from multiprocess.shared_memory import SharedMemory

from pys2sleplet.slepian.slepian_functions import SlepianFunctions
from pys2sleplet.utils.array_methods import fill_upper_triangle_of_hermitian_matrix
from pys2sleplet.utils.config import config
from pys2sleplet.utils.mask_methods import create_mask_region
from pys2sleplet.utils.parallel_methods import split_L_into_chunks
from pys2sleplet.utils.region import Region
from pys2sleplet.utils.vars import (
    ANNOTATION_COLOUR,
    ARROW_STYLE,
    PHI_MAX_DEFAULT,
    PHI_MIN_DEFAULT,
    THETA_MAX_DEFAULT,
    THETA_MIN_DEFAULT,
)

_file_location = Path(__file__).resolve()


@dataclass
class SlepianLimitLatLon(SlepianFunctions):
    theta_min: float
    theta_max: float
    phi_min: float
    phi_max: float
    ncpu: int
    _name_ending: str = field(init=False, repr=False)
    _ncpu: int = field(default=config.NCPU, init=False, repr=False)
    _phi_max: float = field(default=PHI_MAX_DEFAULT, init=False, repr=False)
    _phi_min: float = field(default=PHI_MIN_DEFAULT, init=False, repr=False)
    _region: Region = field(init=False, repr=False)
    _theta_max: float = field(default=THETA_MAX_DEFAULT, init=False, repr=False)
    _theta_min: float = field(default=THETA_MIN_DEFAULT, init=False, repr=False)

    def __post_init__(self) -> None:
        self.N = self.L - 1
        self.region = Region(
            theta_min=self.theta_min,
            theta_max=self.theta_max,
            phi_min=self.phi_min,
            phi_max=self.phi_max,
        )
        self.name_ending = self.region.name_ending
        super().__post_init__()

    def _create_annotations(self) -> None:
        p1, p2, t1, t2 = (
            np.array([self.phi_min]),
            np.array([self.phi_max]),
            np.array([self.theta_min]),
            np.array([self.theta_max]),
        )
        p3, p4, t3, t4 = (
            ne.evaluate("(p1 + 2 * p2) / 3"),
            ne.evaluate("(2 * p1 + p2) / 3"),
            ne.evaluate("(t1 + 2 * t2) / 3"),
            ne.evaluate("(2 * t1 + t2) / 3"),
        )
        for t in [t1, t2, t3, t4]:
            t_condition = (t == [t3, t4]).any()
            for p in [p1, p2, p3, p4]:
                p_condition = (p == [p3, p4]).any()
                if not (t_condition and p_condition):
                    x, y, z = ssht.s2_to_cart(t, p)
                    self.annotations.append(
                        {
                            **dict(
                                x=x[0], y=y[0], z=z[0], arrowcolor=ANNOTATION_COLOUR
                            ),
                            **ARROW_STYLE,
                        }
                    )

    def _create_fn_name(self) -> None:
        self.name = f"slepian{self.name_ending}"

    def _create_mask(self) -> None:
        self.mask = create_mask_region(self.L, self.region)

    def _create_matrix_location(self) -> None:
        self.matrix_location = (
            _file_location.parents[2]
            / "data"
            / "slepian"
            / "lat_lon"
            / f"D{self.name_ending}_L{self.L}.npy"
        )

    def _solve_eigenproblem(self) -> None:
        K = self._load_K_matrix()
        eigenvalues, eigenvectors = np.linalg.eigh(K)
        self.eigenvalues, self.eigenvectors = self._clean_evals_and_evecs(
            eigenvalues, eigenvectors
        )

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
            G = self._slepian_integral()

            # Compute Slepian matrix
            if self.ncpu == 1:
                K = self._slepian_matrix_serial(G)
            else:
                K = self._slepian_matrix_parallel(G)

            # save to speed up for future
            if config.SAVE_MATRICES:
                np.save(self.matrix_location, K)

        return K

    def _slepian_integral(self) -> np.ndarray:
        """
        Syntax:
        G = _slepian_integral()

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

        def helper(row: int, col: int, S: float) -> None:
            """
            Using conjugate symmetry property to reduce the number of iterations
            """
            try:
                Q = ne.evaluate(
                    f"(1/({col}*{col}-1))*(exp(1j*col*{self.theta_min})*"
                    f"(1j*col*sin({self.theta_min})-cos({self.theta_min}))"
                    f"+exp(1j*col*{self.theta_max})"
                    f"*(cos({self.theta_max})-1j*col*sin({self.theta_max})))"
                )
            except ZeroDivisionError:
                Q = ne.evaluate(
                    f"(2*1j*col*({self.theta_max}-{self.theta_min})+exp("
                    f"2*1j*col*{self.theta_min})-exp(2*1j*col*{self.theta_max}))/4"
                )

            G[2 * self.N + row, 2 * self.N + col] = ne.evaluate(f"{Q}*S")
            G[2 * self.N - row, 2 * self.N - col] = G[
                2 * self.N + row, 2 * self.N + col
            ].conj()

        # row = 0
        S = ne.evaluate(f"{self.phi_max}-{self.phi_min}")
        for col in range(-2 * self.N, 1):
            helper(0, col, S)

        # row != 0
        for row in range(-2 * self.N, 0):
            S = ne.evaluate(
                f"(1j/row)*(exp(1j*row*{self.phi_min})-exp(1j*row*{self.phi_max}))"
            )
            for col in range(-2 * self.N, 2 * self.N + 1):
                helper(row, col, S)

        return G

    def _slepian_matrix_serial(self, G: np.ndarray) -> np.ndarray:
        """
        Syntax:
        K = _slepian_matrix_serial(G)

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

        # initialise real and imaginary matrices
        K_r = np.zeros((self.L * self.L, self.L * self.L))
        K_i = np.zeros((self.L * self.L, self.L * self.L))

        for l in range(self.L):
            self._slepian_matrix_helper(K_r, K_i, l, dl_array, G)

        # combine real and imaginary parts
        K = ne.evaluate("K_r+1j*K_i")

        # fill in remaining triangle section
        fill_upper_triangle_of_hermitian_matrix(K)

        return K

    def _slepian_matrix_parallel(self, G: np.ndarray) -> np.ndarray:
        """
        Syntax:
        K = _slepian_matrix_parallel(G)

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

        # initialise real and imaginary matrices
        K_r = np.zeros((self.L * self.L, self.L * self.L))
        K_i = np.zeros((self.L * self.L, self.L * self.L))

        # create shared memory block
        shm_r = SharedMemory(create=True, size=K_r.nbytes)
        shm_i = SharedMemory(create=True, size=K_i.nbytes)
        # create a array backed by shared memory
        K_r_ext = np.ndarray(K_r.shape, dtype=K_r.dtype, buffer=shm_r.buf)  # noqa: F841
        K_i_ext = np.ndarray(K_i.shape, dtype=K_i.dtype, buffer=shm_i.buf)  # noqa: F841

        def func(chunk: List[int]) -> None:
            """
            calculate K matrix components for each chunk
            """
            # attach to the existing shared memory block
            ex_shm_r = SharedMemory(name=shm_r.name)
            ex_shm_i = SharedMemory(name=shm_i.name)
            K_r_int = np.ndarray(K_r.shape, dtype=K_r.dtype, buffer=ex_shm_r.buf)
            K_i_int = np.ndarray(K_i.shape, dtype=K_i.dtype, buffer=ex_shm_i.buf)

            # deal with chunk
            for l in chunk:
                self._slepian_matrix_helper(K_r_int, K_i_int, l, dl_array, G)

            # clean up shared memory
            ex_shm_r.close()
            ex_shm_i.close()

        # split up L range to maximise effiency
        chunks = split_L_into_chunks(self.L, self.ncpu)

        # initialise pool and apply function
        with Pool(processes=self.ncpu) as p:
            p.map(func, chunks)

        # retrieve from parallel function
        K = ne.evaluate(f"K_r_ext+1j*K_i_ext")

        # Free and release the shared memory block at the very end
        shm_r.close()
        shm_r.unlink()
        shm_i.close()
        shm_i.unlink()

        # fill in remaining triangle section
        fill_upper_triangle_of_hermitian_matrix(K)

        return K

    def _slepian_matrix_helper(
        self,
        K_r: np.ndarray,
        K_i: np.ndarray,
        l: int,
        dl_array: np.ndarray,
        G: np.ndarray,
    ) -> None:
        """
        used in both serial and parallel calculations

        the hack with splitting into real and imaginary parts
        is not required for the serial case but here for ease
        """
        dl = dl_array[l]

        for p in range(l + 1):
            dp = dl_array[p]
            C1 = ne.evaluate(f"sqrt((2*l+1)*(2*p+1))/(4*{np.pi})")

            for m in range(-l, l + 1):
                for q in range(-p, p + 1):

                    row = m - q
                    C2 = ne.evaluate(f"(-1j)**{row}")
                    ind_r = ne.evaluate(f"2*{self.N}+{row}")

                    for mp in range(-l, l + 1):
                        C3 = dl[self.N + mp, self.N + m] * dl[self.N + mp, self.N]
                        S1 = 0

                        for qp in range(-p, p + 1):
                            col = ne.evaluate("mp-qp")
                            C4 = dp[self.N + qp, self.N + q] * dp[self.N + qp, self.N]
                            ind_c = ne.evaluate(f"2*{self.N}+{col}")
                            S1 += C4 * G[ind_r, ind_c]

                        idx = (l * (l + 1) + m, p * (p + 1) + q)
                        K_r[idx] += ne.evaluate(f"({C3}*S1).real")
                        K_i[idx] += ne.evaluate(f"({C3}*S1).imag")

                    idx = (ne.evaluate("l*(l+1)+m"), ne.evaluate("p*(p+1)+q"))
                    real, imag = K_r[idx], K_i[idx]
                    K_r[idx] = ne.evaluate(
                        f"{real}*({C1}*{C2}).real - {imag}*({C1}*{C2}).imag"
                    )
                    K_i[idx] = ne.evaluate(
                        f"{real}*({C1}*{C2}).imag + {imag}*({C1}*{C2}).real"
                    )

    @staticmethod
    def _clean_evals_and_evecs(
        eigenvalues: np.ndarray, eigenvectors: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        need eigenvalues and eigenvectors to be in a certain format
        """
        # eigenvalues should be real
        eigenvalues = ne.evaluate("eigenvalues.real")

        # Sort eigenvalues and eigenvectors in descending order of eigenvalues
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx].conj().T

        # ensure first element of each eigenvector is positive
        eigenvectors *= np.where(eigenvectors[:, 0] < 0, -1, 1)[:, np.newaxis]

        return eigenvalues, eigenvectors

    @property
    def N(self) -> int:
        return self._N

    @N.setter
    def N(self, N: int) -> None:
        self._N = N

    @property
    def name_ending(self) -> str:
        return self._name_ending

    @name_ending.setter
    def name_ending(self, name_ending: str) -> None:
        self._name_ending = name_ending

    @property  # type: ignore
    def ncpu(self) -> int:
        return self._ncpu

    @ncpu.setter
    def ncpu(self, ncpu: int) -> None:
        if isinstance(ncpu, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            ncpu = SlepianLimitLatLon._ncpu
        self._ncpu = ncpu

    @property  # type:ignore
    def phi_max(self) -> float:
        return self._phi_max

    @phi_max.setter
    def phi_max(self, phi_max: float) -> None:
        if isinstance(phi_max, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            phi_max = SlepianLimitLatLon._phi_max
        self._phi_max = phi_max

    @property  # type:ignore
    def phi_min(self) -> float:
        return self._phi_min

    @phi_min.setter
    def phi_min(self, phi_min: float) -> None:
        if isinstance(phi_min, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            phi_min = SlepianLimitLatLon._phi_min
        self._phi_min = phi_min

    @property
    def region(self) -> Region:
        return self._region

    @region.setter
    def region(self, region: Region) -> None:
        self._region = region

    @property  # type:ignore
    def theta_max(self) -> float:
        return self._theta_max

    @theta_max.setter
    def theta_max(self, theta_max: float) -> None:
        if isinstance(theta_max, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            theta_max = SlepianLimitLatLon._theta_max
        self._theta_max = theta_max

    @property  # type: ignore
    def theta_min(self) -> float:
        return self._theta_min

    @theta_min.setter
    def theta_min(self, theta_min: float) -> None:
        if isinstance(theta_min, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            theta_min = SlepianLimitLatLon._theta_min
        self._theta_min = theta_min
