"""
Game model shared by every problem in the tutorial.

Notation (matches the main README):
    N        number of core levels, L in 0..N-1 in code (1..N in the docs)
    M        number of item star tiers, j in 0..M-1 in code (1..M stars)
    P[L, j]  probability that a hit at level L yields a (j+1)-star item
    stars[j] = j+1, the star rating of tier j
    r[L]     expected stars per hit at level L  = sum_j stars[j] P[L, j]
    h        cost of one hit
    C[L]     cost of attempting upgrade L -> L+1        (len N-1)
    U[L]     success probability of upgrade L -> L+1    (len N-1)
    D[L]     prob. of dropping to L-1 when upgrade from L fails (len N-1, demotion mechanic)

Currency is *coins*; *stars* only ever mean item quality.

Words: utility u = what is maximised (baseline: total stars); value V = best expected utility.

Every axis of the tag (I, U, B, E, O, K) is a *property of the problem*, not of the
numbers. Mechanics (demotion via D, pity, hit cap) are rule tweaks, not axes. The helpers at the bottom produce the variant of the running example that a
given problem folder needs (e.g. `sure_upgrades` for the Ud axis value).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import reduce
from math import gcd

import numpy as np


@dataclass(frozen=True)
class Game:
    h: int
    C: tuple[int, ...]
    U: tuple[float, ...]
    P: np.ndarray                       # (N, M)
    D: tuple[float, ...] = field(default=())
    items_deterministic: bool = False   # axis I: Id if True, Ir if False

    def __post_init__(self):
        N, M = self.P.shape
        assert len(self.C) == len(self.U) == N - 1
        assert np.allclose(self.P.sum(1), 1)
        if not self.D:
            object.__setattr__(self, "D", (0.0,) * (N - 1))

    # ---- derived quantities ----
    @property
    def N(self) -> int:
        return self.P.shape[0]

    @property
    def M(self) -> int:
        return self.P.shape[1]

    @property
    def stars(self) -> np.ndarray:
        """Star rating of each tier: 1..M."""
        return np.arange(1, self.M + 1, dtype=float)

    @property
    def r(self) -> np.ndarray:
        """Expected stars per hit at each level."""
        return self.P @ self.stars

    @property
    def unit(self) -> int:
        """Delta = gcd of all costs: budgets are discretised on this grid."""
        return reduce(gcd, [self.h, *self.C])

    @property
    def expected_upgrade_cost(self) -> np.ndarray:
        """C_L / U_L: expected coins to clear level L (geometric number of tries)."""
        return np.array(self.C) / np.array(self.U)

    @property
    def tag_IU(self) -> str:
        i = "Id" if self.items_deterministic else "Ir"
        u = "Ud" if all(p == 1 for p in self.U) else "Ur"
        return f"{i}-{u}"


# --------------------------------------------------------------------------------------
# Running example. A *given* table (axis K: known). It happens to come from a parametric
# family (see `parametric_item_table`), rounded to 2 decimals so it reads like data.
# --------------------------------------------------------------------------------------
EXAMPLE_P = np.array([
    # 1*    2*    3*    4*    5*
    [0.67, 0.30, 0.03, 0.00, 0.00],   # level 1
    [0.37, 0.49, 0.13, 0.01, 0.00],   # level 2
    [0.13, 0.46, 0.35, 0.06, 0.00],   # level 3
    [0.02, 0.23, 0.50, 0.23, 0.02],   # level 4
    [0.00, 0.06, 0.35, 0.46, 0.13],   # level 5
    [0.00, 0.01, 0.13, 0.49, 0.37],   # level 6
    [0.00, 0.00, 0.03, 0.30, 0.67],   # level 7
])

EXAMPLE = Game(
    h=200,
    C=(400, 600, 900, 1300, 1800, 2500),
    U=(0.70, 0.60, 0.50, 0.40, 0.30, 0.25),
    P=EXAMPLE_P,
)


def parametric_item_table(N: int, M: int, sigma: float = 0.8) -> np.ndarray:
    """P[L, j] proportional to a Gaussian bump centred at mu_L, mu rising 1 -> M.

    A *parametric* description: 2 numbers (sigma, and the linear mu schedule) instead of
    N*M table entries. For solving, it is identical to a given table; it matters for
    closed forms and for learning (axis K).
    """
    mu = 1 + (M - 1) * np.arange(N) / (N - 1)
    j = np.arange(1, M + 1)
    P = np.exp(-((j[None] - mu[:, None]) ** 2) / (2 * sigma**2))
    return P / P.sum(1, keepdims=True)


# ---- axis variants of a game ---------------------------------------------------------
def deterministic_items(g: Game) -> Game:
    """Axis I: Ir -> Id. Every hit at level L yields exactly r_L."""
    return replace(g, items_deterministic=True)


def sure_upgrades(g: Game) -> Game:
    """Axis U: Ur -> Ud. Every upgrade succeeds at the listed cost."""
    return replace(g, U=(1.0,) * len(g.U), D=())


def certainty_equivalent(g: Game) -> Game:
    """Ur -> Ud by charging the expected cost C_L/U_L for a sure upgrade.

    Costs are rounded to the game's unit so budgets stay on the grid.
    """
    u = g.unit
    C = tuple(int(round(c / p / u)) * u for c, p in zip(g.C, g.U))
    return replace(g, C=C, U=(1.0,) * len(g.U), D=(), items_deterministic=True)
