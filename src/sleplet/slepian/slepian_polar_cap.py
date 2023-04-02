from concurrent.futures import ThreadPoolExecutor
from dataclasses import KW_ONLY
from pathlib import Path

import gmpy2 as gp
import numpy as np
import pyssht as ssht
from numpy import linalg as LA  # noqa: N812
from numpy import typing as npt
from pydantic import validator
from pydantic.dataclasses import dataclass

import sleplet
import sleplet._data.setup_pooch
import sleplet._mask_methods
import sleplet._parallel_methods
import sleplet._validation
import sleplet.harmonic_methods
import sleplet.slepian.region
from sleplet.slepian.slepian_functions import SlepianFunctions

_data_path = Path(__file__).resolve().parents[1] / "_data"

L_SAVE_ALL = 16


@dataclass(config=sleplet._validation.Validation)
class SlepianPolarCap(SlepianFunctions):
    """Class to create a polar cap Slepian region on the sphere."""

    theta_max: float
    """TODO"""
    _: KW_ONLY
    gap: bool = False
    """TODO"""
    order: int | npt.NDArray[np.int_] | None = None
    """TODO"""

    def __post_init_post_parse__(self) -> None:
        super().__post_init_post_parse__()

    def _create_fn_name(self) -> str:
        return f"slepian_{self.region.name_ending}"

    def _create_region(self) -> "sleplet.slepian.region.Region":
        return sleplet.slepian.region.Region(gap=self.gap, theta_max=self.theta_max)

    def _create_mask(self) -> npt.NDArray[np.float_]:
        return sleplet._mask_methods.create_mask_region(self.L, self.region)

    def _calculate_area(self) -> float:
        return 2 * np.pi * (1 - np.cos(self.theta_max))

    def _create_matrix_location(self) -> str:
        return f"slepian_eigensolutions_D_{self.region.name_ending}_L{self.L}_N{self.N}"

    def _solve_eigenproblem(
        self,
    ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.complex_]]:
        eval_loc = f"{self.matrix_location}_eigenvalues.npy"
        evec_loc = f"{self.matrix_location}_eigenvectors.npy"
        order_loc = f"{self.matrix_location}_orders.npy"
        try:
            return self._solve_eigenproblem_from_files(eval_loc, evec_loc, order_loc)
        except TypeError:
            return self._solve_eigenproblem_from_scratch(eval_loc, evec_loc, order_loc)

    def _solve_eigenproblem_from_files(
        self,
        eval_loc: str,
        evec_loc: str,
        order_loc: str,
    ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.complex_]]:
        """Solves eigenproblem with files already saved."""
        eigenvalues = np.load(
            sleplet._data.setup_pooch.find_on_pooch_then_local(eval_loc),
        )
        eigenvectors = np.load(
            sleplet._data.setup_pooch.find_on_pooch_then_local(evec_loc),
        )
        orders = np.load(sleplet._data.setup_pooch.find_on_pooch_then_local(order_loc))

        if self.order is not None:
            idx = np.where(orders == self.order)
            return eigenvalues[idx], eigenvectors[idx]
        self.order = orders
        return eigenvalues, eigenvectors

    def _solve_eigenproblem_from_scratch(
        self,
        eval_loc: str,
        evec_loc: str,
        order_loc: str,
    ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.complex_]]:
        """Sovles eigenproblem from scratch and then saves the files."""
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
        limit = self.N if self.L > L_SAVE_ALL else None
        np.save(_data_path / eval_loc, eigenvalues)
        np.save(_data_path / evec_loc, eigenvectors[:limit])
        np.save(_data_path / order_loc, self.order)
        return eigenvalues, eigenvectors

    def _solve_eigenproblem_order(
        self,
        m: int,
    ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.complex_]]:
        """Solves the eigenproblem for a given order 'm;"""
        emm = sleplet.harmonic_methods._create_emm_vector(self.L)
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
        """Sorts all eigenvalues and eigenvectors for all orders."""
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[idx]
        orders = orders[idx]
        return eigenvalues, eigenvectors, orders

    def _create_Dm_matrix(  # noqa: N802
        self,
        m: int,
        emm: npt.NDArray[np.float_],
    ) -> npt.NDArray[np.float_]:
        """Syntax:
        Dm = _create_Dm_matrix(m, P).

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

        Dm_ext, shm_ext = sleplet._parallel_methods.create_shared_memory_array(Dm)

        def func(chunk: list[int]) -> None:
            """Calculate D matrix components for each chunk."""
            Dm_int, shm_int = sleplet._parallel_methods.attach_to_shared_memory_block(
                Dm,
                shm_ext,
            )

            # deal with chunk
            for i in chunk:
                sleplet.logger.info(f"start ell: {i}")
                self._dm_matrix_helper(Dm_int, i, m, lvec, Pl, ell)
                sleplet.logger.info(f"finish ell: {i}")

            sleplet._parallel_methods.free_shared_memory(shm_int)

        # split up L range to maximise effiency
        chunks = sleplet._parallel_methods.split_arr_into_chunks(
            self.L - m,
            sleplet.NCPU,
        )

        # initialise pool and apply function
        with ThreadPoolExecutor(max_workers=sleplet.NCPU) as e:
            e.map(func, chunks)

        # retrieve from parallel function
        Dm = Dm_ext * (-1) ** m / 2

        # Free and release the shared memory block at the very end
        sleplet._parallel_methods.free_shared_memory(shm_ext)
        sleplet._parallel_methods.release_shared_memory(shm_ext)
        return Dm

    def _create_legendre_polynomials_table(
        self,
        emm: npt.NDArray[np.float_],
    ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.int_]]:
        """Create Legendre polynomials table for matrix calculation."""
        Plm = ssht.create_ylm(self.theta_max, 0, 2 * self.L).real.reshape(-1)
        ind = emm == 0
        ell = np.arange(2 * self.L)[np.newaxis]
        Pl = np.sqrt((4 * np.pi) / (2 * ell + 1)) * Plm[ind]
        return Pl, ell

    def _dm_matrix_helper(  # noqa: PLR0913
        self,
        Dm: npt.NDArray[np.float_],
        i: int,
        m: int,
        lvec: npt.NDArray[np.int_],
        Pl: npt.NDArray[np.float_],
        ell: npt.NDArray[np.int_],
    ) -> None:
        """Used in both serial and parallel calculations."""
        el = int(lvec[i])
        for j in range(i, self.L - m):
            p = int(lvec[j])
            c = 0
            for n in range(abs(el - p), el + p + 1):
                A = Pl[ell == n - 1] if n != 0 else 1
                c += (
                    self._wigner3j(el, n, p, 0, 0, 0)
                    * self._wigner3j(el, n, p, m, 0, -m)
                    * (A - Pl[ell == n + 1])
                )
            Dm[i, j] = (
                self._polar_gap_modification(el, p)
                * np.sqrt((2 * el + 1) * (2 * p + 1))
                * c
            )
            Dm[j, i] = Dm[i, j]

    @staticmethod
    def _wigner3j(  # noqa: PLR0913
        l1: int,
        l2: int,
        l3: int,
        m1: int,
        m2: int,
        m3: int,
    ) -> float:
        """Syntax:
        s = _wigner3j (l1, l2, l3, m1, m2, m3).

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
                    * gp.factorial(l3 - m3),
                )
            )
        return s

    def _polar_gap_modification(self, ell1: int, ell2: int) -> int:
        """Eq 67 - Spherical Slepian functions and the polar gap in geodesy
        multiply by 1 + (-1)*(ell+ell').
        """
        return 1 + self.gap * (-1) ** (ell1 + ell2)

    def _clean_evals_and_evecs(
        self,
        eigenvalues: npt.NDArray[np.float_],
        gl: npt.NDArray[np.float_],
        emm: npt.NDArray[np.float_],
        m: int,
    ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.complex_]]:
        """Need eigenvalues and eigenvectors to be in a certain format."""
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
    def _check_order(cls, v, values):
        if v is not None and (np.abs(v) >= values["L"]).any():
            raise ValueError(f"Order magnitude should be less than {values['L']}")
        return v

    @validator("theta_max")
    def _check_theta_max(cls, v):
        if v == 0:
            raise ValueError("theta_max cannot be zero")
        return v
