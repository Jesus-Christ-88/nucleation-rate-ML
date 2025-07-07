# ising3d.py

import numpy as np


class Ising3D:
    """A simple 3‑D Ising model on an L×L×L cubic lattice with
    periodic boundary conditions.

    Parameters
    ----------
    L : int
        Linear lattice size (total spins = L³).
    J : float, optional
        Ferromagnetic coupling constant (>0).  Default 1.0.
    h : float, optional
        Uniform external field (negative → favours spin −1).  Default −0.1.
    T : float, optional
        Temperature in k_B = 1 units.  Default 2.0.
    """

    # ------------------------------------------------------------------
    #                          construction
    # ------------------------------------------------------------------
    def __init__(self, L: int, J: float = 1.0, h: float = -0.1, T: float = 2.0):
        self.L, self.J, self.h, self.T = L, J, h, T
        self.beta = 1.0 / T

        # all spins start +1 (metastable for h < 0)
        self.s = np.ones((L, L, L), dtype=np.int8)

    # ------------------------------------------------------------------
    #                    periodic‑boundary helpers
    # ------------------------------------------------------------------
    def _mod(self, i: int) -> int:
        """Fast positive modulo for small ±ints (avoids Python % overhead)."""
        return i % self.L

    def neighbours(self, i: int, j: int, k: int):
        """Yield the 6 nearest‑neighbour indices with periodic boundaries."""
        L = self.L
        return (
            (self._mod(i + 1), j, k),
            (self._mod(i - 1), j, k),
            (i, self._mod(j + 1), k),
            (i, self._mod(j - 1), k),
            (i, j, self._mod(k + 1)),
            (i, j, self._mod(k - 1)),
        )

    # ------------------------------------------------------------------
    #                           observables
    # ------------------------------------------------------------------
    def magnetisation(self) -> float:
        """Average spin m = (1/N) Σ_i s_i  ;  ranges in [‑1, +1]."""
        return self.s.mean()

    def _dE_single_flip(self, i: int, j: int, k: int) -> float:
        """Energy change ΔE if the spin at (i,j,k) were flipped.

        Formula (in units where k_B = 1):
            ΔE = 2 s_i ( J Σ_nb s_nb + h )
        where the sum is over the 6 nearest neighbours of site (i,j,k).
        """
        sijk = self.s[i, j, k]

        # sum over the 6 neighbour spins
        nb_sum = 0
        for p, q, r in self.neighbours(i, j, k):
            nb_sum += self.s[p, q, r]

        return 2 * sijk * (self.J * nb_sum + self.h)

    def total_energy(self) -> float:
        """Total Hamiltonian
            H = -J Σ_<ij> s_i s_j  - h Σ_i s_i
        Bonds are counted once per pair.
        """
        # nearest‑neighbour interaction term via array rolls
        bonds = (
            self.s * np.roll(self.s, 1, axis=0) +
            self.s * np.roll(self.s, 1, axis=1) +
            self.s * np.roll(self.s, 1, axis=2)
        ).sum()  # each bond counted exactly once
        return -self.J * bonds - self.h * self.s.sum()

    # ------------------------------------------------------------------
    #                           string helpers
    # ------------------------------------------------------------------
    def __repr__(self):
        return (
            f"Ising3D(L={self.L}, J={self.J}, h={self.h}, "
            f"T={self.T}, m={self.magnetisation():.3f})"
        )
    # ------------------------------------------------------------------
    #                    MC dynamics
    # ------------------------------------------------------------------
    def sweep(self):
        """
        One Metropolis sweep: L^3 attempted single-spin flips.
        """
        L = self.L
        for _ in range(L**3):
            i = np.random.randint(0, L)
            j = np.random.randint(0, L)
            k = np.random.randint(0, L)

            dE = self._dE_single_flip(i, j, k)
            if dE <= 0 or np.random.rand() < np.exp(-self.beta * dE):
                self.s[i, j, k] *= -1  # accept the flip


