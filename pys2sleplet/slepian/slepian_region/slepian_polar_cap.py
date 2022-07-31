from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import gmpy2 as gp
import numpy as np
import pyssht as ssht
from multiprocess import Pool
from numpy import linalg as LA

from pys2sleplet.slepian.slepian_functions import SlepianFunctions
from pys2sleplet.utils.config import settings
from pys2sleplet.utils.harmonic_methods import create_emm_vector
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.mask_methods import create_mask_region
from pys2sleplet.utils.parallel_methods import (
    attach_to_shared_memory_block,
    create_shared_memory_array,
    free_shared_memory,
    release_shared_memory,
    split_arr_into_chunks,
)
from pys2sleplet.utils.region import Region

L_SAVE_ALL = 16

_file_location = Path(__file__).resolve()
_eigen_path = _file_location.parents[2] / "data" / "slepian" / "eigensolutions"


@dataclass
class SlepianPolarCap(SlepianFunctions):
    theta_max: float
    order: Optional[Union[int, np.ndarray]]
    gap: bool
    _gap: bool = field(default=False, init=False, repr=False)
    _order: Optional[Union[int, np.ndarray]] = field(
        default=None, init=False, repr=False
    )
    _region: Region = field(init=False, repr=False)
    _theta_max: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.region = Region(gap=self.gap, theta_max=self.theta_max)
        super().__post_init__()

    def _create_fn_name(self) -> None:
        self.name = f"slepian_{self.region.name_ending}"

    def _create_mask(self) -> None:
        self.mask = create_mask_region(self.L, self.region)

    def _calculate_area(self) -> None:
        self.area = 2 * np.pi * (1 - np.cos(self.theta_max))

    def _create_matrix_location(self) -> None:
        self.matrix_location = (
            _eigen_path / f"D_{self.region.name_ending}_L{self.L}_N{self.N}"
        )

    def _solve_eigenproblem(self) -> None:
        eval_loc = self.matrix_location / "eigenvalues.npy"
        evec_loc = self.matrix_location / "eigenvectors.npy"
        order_loc = self.matrix_location / "orders.npy"
        if eval_loc.exists() and evec_loc.exists() and order_loc.exists():
            logger.info("binaries found - loading...")
            self._solve_eigenproblem_from_files(eval_loc, evec_loc, order_loc)
        else:
            self._solve_eigenproblem_from_scratch(eval_loc, evec_loc, order_loc)

    def _solve_eigenproblem_from_files(
        self, eval_loc: Path, evec_loc: Path, order_loc: Path
    ) -> None:
        """
        solves eigenproblem with files already saved
        """
        eigenvalues = np.load(eval_loc)
        eigenvectors = np.load(evec_loc)
        orders = np.load(order_loc)

        if self.order is not None:
            idx = np.where(orders == self.order)
            self.eigenvalues = eigenvalues[idx]
            self.eigenvectors = eigenvectors[idx]
        else:
            self.eigenvalues = eigenvalues
            self.eigenvectors = eigenvectors
            self.order = orders

    def _solve_eigenproblem_from_scratch(
        self, eval_loc: Path, evec_loc: Path, order_loc: Path
    ) -> None:
        """
        sovles eigenproblem from scratch and then saves the files
        """
        if isinstance(self.order, int):
            self.eigenvalues, self.eigenvectors = self._solve_eigenproblem_order(
                self.order
            )
        else:
            evals_all = np.empty(0)
            evecs_all = np.empty((0, self.L**2), dtype=np.complex_)
            emm = np.empty(0, dtype=int)
            for m in range(-(self.L - 1), self.L):
                evals_m, evecs_m = self._solve_eigenproblem_order(m)
                evals_all = np.append(evals_all, evals_m)
                evecs_all = np.concatenate((evecs_all, evecs_m))
                emm = np.append(emm, [m] * len(evals_m))
            (
                self.eigenvalues,
                self.eigenvectors,
                self.order,
            ) = self._sort_all_evals_and_evecs(evals_all, evecs_all, emm)
            if settings.SAVE_MATRICES:
                limit = self.N if self.L > L_SAVE_ALL else None
                np.save(eval_loc, self.eigenvalues)
                np.save(evec_loc, self.eigenvectors[:limit])
                np.save(order_loc, self.order)

    def _solve_eigenproblem_order(self, m: int) -> tuple[np.ndarray, np.ndarray]:
        """
        solves the eigenproblem for a given order 'm;
        """
        emm = create_emm_vector(self.L)
        Dm = self._create_Dm_matrix(abs(m), emm)
        eigenvalues, gl = LA.eigh(Dm)
        eigenvalues, eigenvectors = self._clean_evals_and_evecs(eigenvalues, gl, emm, m)
        return eigenvalues, eigenvectors

    def _sort_all_evals_and_evecs(
        self, eigenvalues: np.ndarray, eigenvectors: np.ndarray, orders: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        sorts all eigenvalues and eigenvectors for all orders
        """
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[idx]
        orders = orders[idx]
        return eigenvalues, eigenvectors, orders

    def _create_Dm_matrix(self, m: int, emm: np.ndarray) -> np.ndarray:
        """
        Syntax:
        Dm = _create_Dm_matrix(m, P)

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
        Pl, ell = self._create_legendre_polynomials_table(emm)
        Dm = np.zeros((self.L - m, self.L - m))
        lvec = np.arange(m, self.L)

        Dm_ext, shm_ext = create_shared_memory_array(Dm)

        def func(chunk: list[int]) -> None:
            """
            calculate D matrix components for each chunk
            """
            Dm_int, shm_int = attach_to_shared_memory_block(Dm, shm_ext)

            # deal with chunk
            for i in chunk:
                logger.info(f"start ell: {i}")
                self._dm_matrix_helper(Dm_int, i, m, lvec, Pl, ell)
                logger.info(f"finish ell: {i}")

            free_shared_memory(shm_int)

        # split up L range to maximise effiency
        chunks = split_arr_into_chunks(self.L - m, settings.NCPU)

        # initialise pool and apply function
        with Pool(processes=settings.NCPU) as p:
            p.map(func, chunks)

        # retrieve from parallel function
        Dm = Dm_ext * (-1) ** m / 2

        # Free and release the shared memory block at the very end
        free_shared_memory(shm_ext)
        release_shared_memory(shm_ext)
        return Dm

    def _create_legendre_polynomials_table(
        self, emm: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        create Legendre polynomials table for matrix calculation
        """
        Plm = ssht.create_ylm(self.theta_max, 0, 2 * self.L).real.reshape(-1)
        ind = emm == 0
        l = np.arange(2 * self.L)[np.newaxis]
        Pl = np.sqrt((4 * np.pi) / (2 * l + 1)) * Plm[ind]
        return Pl, l

    def _dm_matrix_helper(
        self,
        Dm: np.ndarray,
        i: int,
        m: int,
        lvec: np.ndarray,
        Pl: np.ndarray,
        ell: np.ndarray,
    ) -> None:
        """
        used in both serial and parallel calculations
        """
        l = int(lvec[i])
        for j in range(i, self.L - m):
            p = int(lvec[j])
            c = 0
            for n in range(abs(l - p), l + p + 1):
                A = Pl[ell == n - 1] if n != 0 else 1
                c += (
                    self._wigner3j(l, n, p, 0, 0, 0)
                    * self._wigner3j(l, n, p, m, 0, -m)
                    * (A - Pl[ell == n + 1])
                )
            Dm[i, j] = (
                self._polar_gap_modification(l, p)
                * np.sqrt((2 * l + 1) * (2 * p + 1))
                * c
            )
            Dm[j, i] = Dm[i, j]

    @staticmethod
    def _wigner3j(l1: int, l2: int, l3: int, m1: int, m2: int, m3: int) -> float:
        """
        Syntax:
        s = _wigner3j (l1, l2, l3, m1, m2, m3)

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
            raise ValueError("Arguments must either be integer or half-integer!")

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

            # sum is over all those t for which the
            # following factorials have non-zero arguments
            s = sum(
                (-1) ** t
                / (
                    gp.factorial(t)
                    * gp.factorial(t - t1)
                    * gp.factorial(t - t2)
                    * gp.factorial(t3 - t)
                    * gp.factorial(t4 - t)
                    * gp.factorial(t5 - t)
                )
                for t in range(tmin, tmax + 1)
            )
            triangle_coefficient = (
                gp.factorial(t3)
                * gp.factorial(l1 - l2 + l3)
                * gp.factorial(-l1 + l2 + l3)
                / gp.factorial(l1 + l2 + l3 + 1)
            )

            s *= (
                np.float_power(-1, l1 - l2 - m3)
                * gp.sqrt(triangle_coefficient)
                * gp.sqrt(
                    gp.factorial(l1 + m1)
                    * gp.factorial(t4)
                    * gp.factorial(t5)
                    * gp.factorial(l2 - m2)
                    * gp.factorial(l3 + m3)
                    * gp.factorial(l3 - m3)
                )
            )
        return s

    def _polar_gap_modification(self, ell1: int, ell2: int) -> int:
        """
        eq 67 - Spherical Slepian functions and the polar gap in geodesy
        multiply by 1 + (-1)*(ell+ell')
        """
        return 1 + self.gap * (-1) ** (ell1 + ell2)

    def _clean_evals_and_evecs(
        self, eigenvalues: np.ndarray, gl: np.ndarray, emm: np.ndarray, m: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        need eigenvalues and eigenvectors to be in a certain format
        """
        # eigenvalues should be real
        eigenvalues = eigenvalues.real

        # Sort eigenvalues and eigenvectors in descending order of eigenvalues
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        gl = gl[:, idx].conj()

        # put back in full D space for harmonic transform
        emm = emm[: self.L**2]
        ind = np.tile(emm == m, (self.L - abs(m), 1))
        eigenvectors = np.zeros((self.L - abs(m), self.L**2), dtype=np.complex_)
        eigenvectors[ind] = gl.T.flatten()

        # ensure first element of each eigenvector is positive
        eigenvectors *= np.where(eigenvectors[:, 0] < 0, -1, 1)[:, np.newaxis]

        # if -ve 'm' find orthogonal eigenvectors to +ve 'm' eigenvectors
        if m < 0:
            eigenvectors *= 1j

        return eigenvalues, eigenvectors

    @property  # type:ignore
    def gap(self) -> bool:
        return self._gap

    @gap.setter
    def gap(self, gap: bool) -> None:
        if isinstance(gap, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            gap = SlepianPolarCap._gap
        self._gap = gap

    @property  # type:ignore
    def order(self) -> Optional[Union[int, np.ndarray]]:
        return self._order

    @order.setter
    def order(self, order: Optional[Union[int, np.ndarray]]) -> None:
        if isinstance(order, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            order = SlepianPolarCap._order
        if order is not None and (np.abs(order) >= self.L).any():
            raise ValueError(f"Order magnitude should be less than {self.L}")
        self._order = order

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
        if theta_max == 0:
            raise ValueError("theta_max cannot be zero")
        self._theta_max = theta_max
