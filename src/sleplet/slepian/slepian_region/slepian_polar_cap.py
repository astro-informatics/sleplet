from concurrent.futures import ThreadPoolExecutor
from dataclasses import KW_ONLY
from pathlib import Path

import gmpy2 as gp
import numpy as np
import pyssht as ssht
from numpy import linalg as LA
from numpy import typing as npt
from pydantic import validator
from pydantic.dataclasses import dataclass

from sleplet.slepian.slepian_functions import SlepianFunctions
from sleplet.utils.config import settings
from sleplet.utils.harmonic_methods import create_emm_vector
from sleplet.utils.logger import logger
from sleplet.utils.mask_methods import create_mask_region
from sleplet.utils.parallel_methods import (
    attach_to_shared_memory_block,
    create_shared_memory_array,
    free_shared_memory,
    release_shared_memory,
    split_arr_into_chunks,
)
from sleplet.utils.region import Region
from sleplet.utils.validation import Validation

L_SAVE_ALL = 16

_file_location = Path(__file__).resolve()
_eigen_path = _file_location.parents[2] / "data" / "slepian" / "eigensolutions"


@dataclass(config=Validation)
class SlepianPolarCap(SlepianFunctions):
    theta_max: float
    _: KW_ONLY
    gap: bool = False
    order: int | npt.NDArray[np.int_] | None = None

    def __post_init_post_parse__(self) -> None:
        super().__post_init_post_parse__()

    def _create_fn_name(self) -> str:
        return f"slepian_{self.region.name_ending}"

    def _create_region(self) -> Region:
        return Region(gap=self.gap, theta_max=self.theta_max)

    def _create_mask(self) -> npt.NDArray[np.float_]:
        return create_mask_region(self.L, self.region)

    def _calculate_area(self) -> float:
        return 2 * np.pi * (1 - np.cos(self.theta_max))

    def _create_matrix_location(self) -> Path:
        return _eigen_path / f"D_{self.region.name_ending}_L{self.L}_N{self.N}"

    def _solve_eigenproblem(
        self,
    ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.complex_]]:
        eval_loc = self.matrix_location / "eigenvalues.npy"
        evec_loc = self.matrix_location / "eigenvectors.npy"
        order_loc = self.matrix_location / "orders.npy"
        if eval_loc.exists() and evec_loc.exists() and order_loc.exists():
            logger.info("binaries found - loading...")
            return self._solve_eigenproblem_from_files(eval_loc, evec_loc, order_loc)
        else:
            return self._solve_eigenproblem_from_scratch(eval_loc, evec_loc, order_loc)

    def _solve_eigenproblem_from_files(
        self, eval_loc: Path, evec_loc: Path, order_loc: Path
    ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.complex_]]:
        """
        solves eigenproblem with files already saved
        """
        eigenvalues = np.load(eval_loc)
        eigenvectors = np.load(evec_loc)
        orders = np.load(order_loc)

        if self.order is not None:
            idx = np.where(orders == self.order)
            return eigenvalues[idx], eigenvectors[idx]
        else:
            self.order = orders
            return eigenvalues, eigenvectors

    def _solve_eigenproblem_from_scratch(
        self, eval_loc: Path, evec_loc: Path, order_loc: Path
    ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.complex_]]:
        """
        sovles eigenproblem from scratch and then saves the files
        """
        if isinstance(self.order, int):
            return self._solve_eigenproblem_order(self.order)

        evals_all = np.empty(0)
        evecs_all = np.empty((0, self.L**2), dtype=np.complex_)
        emm = np.empty(0, dtype=int)
        for m in range(-(self.L - 1), self.L):
            evals_m, evecs_m = self._solve_eigenproblem_order(m)
            evals_all = np.append(evals_all, evals_m)
            evecs_all = np.concatenate((evecs_all, evecs_m))
            emm = np.append(emm, [m] * len(evals_m))
        (
            eigenvalues,
            eigenvectors,
            self.order,
        ) = self._sort_all_evals_and_evecs(evals_all, evecs_all, emm)
        if settings["SAVE_MATRICES"]:
            limit = self.N if self.L > L_SAVE_ALL else None
            np.save(eval_loc, eigenvalues)
            np.save(evec_loc, eigenvectors[:limit])
            np.save(order_loc, self.order)
        return eigenvalues, eigenvectors

    def _solve_eigenproblem_order(
        self, m: int
    ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.complex_]]:
        """
        solves the eigenproblem for a given order 'm;
        """
        emm = create_emm_vector(self.L)
        Dm = self._create_Dm_matrix(abs(m), emm)
        eigenvalues, gl = LA.eigh(Dm)
        eigenvalues, eigenvectors = self._clean_evals_and_evecs(eigenvalues, gl, emm, m)
        return eigenvalues, eigenvectors

    def _sort_all_evals_and_evecs(
        self,
        eigenvalues: npt.NDArray[np.float_],
        eigenvectors: npt.NDArray[np.complex_],
        orders: npt.NDArray[np.int_],
    ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.complex_], npt.NDArray[np.int_]]:
        """
        sorts all eigenvalues and eigenvectors for all orders
        """
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[idx]
        orders = orders[idx]
        return eigenvalues, eigenvectors, orders

    def _create_Dm_matrix(
        self, m: int, emm: npt.NDArray[np.float_]
    ) -> npt.NDArray[np.float_]:
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
        chunks = split_arr_into_chunks(self.L - m, settings["NCPU"])

        # initialise pool and apply function
        with ThreadPoolExecutor(max_workers=settings["NCPU"]) as e:
            e.map(func, chunks)

        # retrieve from parallel function
        Dm = Dm_ext * (-1) ** m / 2

        # Free and release the shared memory block at the very end
        free_shared_memory(shm_ext)
        release_shared_memory(shm_ext)
        return Dm

    def _create_legendre_polynomials_table(
        self, emm: npt.NDArray[np.float_]
    ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.int_]]:
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
        Dm: npt.NDArray[np.float_],
        i: int,
        m: int,
        lvec: npt.NDArray[np.int_],
        Pl: npt.NDArray[np.float_],
        ell: npt.NDArray[np.int_],
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
        self,
        eigenvalues: npt.NDArray[np.float_],
        gl: npt.NDArray[np.float_],
        emm: npt.NDArray[np.float_],
        m: int,
    ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.complex_]]:
        """
        need eigenvalues and eigenvectors to be in a certain format
        """
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

    @validator("order")
    def check_order(cls, v, values):
        if v is not None and (np.abs(v) >= values["L"]).any():
            raise ValueError(f"Order magnitude should be less than {values['L']}")
        return v

    @validator("theta_max")
    def check_theta_max(cls, v):
        if v == 0:
            raise ValueError("theta_max cannot be zero")
        return v
