"""
Monte Carlo simulator: the ground truth every solver is checked against.

DP-style solvers *find* a policy and *predict* its value. The simulator only *plays*
a policy many times and reports what actually happened. If the two disagree, the solver
(or its model) is wrong.

A policy is any callable  policy(obs) -> "hit" | "up" | "stop", where obs is an `Obs`.
For hidden-budget problems (axis Bh) the simulator passes b=None: the policy only sees
what it has spent. `Obs.W` and `Obs.top` summarise the items so far, for utilities that
are not a plain sum (axis Ou, problem 04).

By default a game scores total stars. Pass `utility(items)` (items = list of star ratings)
to score anything else.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .model import Game


@dataclass(frozen=True)
class Obs:
    b: int | None   # coins remaining (None if the budget is hidden)
    s: int          # coins spent so far
    L: int          # core level, 0-indexed
    W: float = 0.0  # total stars so far
    top: int = 0    # best item so far (stars; 0 = no items yet)


Policy = Callable[[Obs], str]


def play_once(game: Game, policy: Policy, budget: int, rng: np.random.Generator,
              hidden_budget: bool = False, max_steps: int = 10**6, utility=None) -> float:
    """Play one game; return total stars collected, or utility(items) if given."""
    b, s, L, value, top = budget, 0, 0, 0.0, 0
    items = []
    r = game.r
    for _ in range(max_steps):
        a = policy(Obs(None if hidden_budget else b, s, L, value, top))
        if a == "stop":
            break
        cost = game.h if a == "hit" else game.C[L]
        if a == "up" and L >= game.N - 1:
            raise ValueError("policy tried to upgrade past the top level")
        if cost > b:            # cannot afford: the game ends (hidden budget learns it here)
            break
        b, s = b - cost, s + cost
        if a == "hit":
            if game.items_deterministic:
                value += r[L]
            else:
                j = int(game.stars[rng.choice(game.M, p=game.P[L])])
                value += j
                top = max(top, j)
                items.append(j)
        else:
            if rng.random() < game.U[L]:
                L += 1
            elif L > 0 and rng.random() < game.D[L]:
                L -= 1
    return value if utility is None else utility(items)


def simulate(game: Game, policy: Policy, budget, n: int = 2000, seed: int = 0,
             hidden_budget: bool = False, utility=None) -> np.ndarray:
    """Play n games. `budget` is an int, or a callable rng -> int (random budget)."""
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        B = budget(rng) if callable(budget) else budget
        out[i] = play_once(game, policy, B, rng, hidden_budget, utility=utility)
    return out


def summary(x: np.ndarray) -> str:
    se = x.std(ddof=1) / np.sqrt(len(x))
    return f"{x.mean():.2f} ± {1.96 * se:.2f} (95% CI, n={len(x)})"


# ---- a few policies reused across problems --------------------------------------------
def target_level_policy(game: Game, k: int) -> Policy:
    """Fixed-target plan: try to upgrade until level k (0-indexed), then hit until broke.

    If the next upgrade is unaffordable before reaching k (possible when upgrades fail),
    it hits with what's left instead. With a hidden budget it just keeps trying.
    """
    def pi(o: Obs) -> str:
        if o.L < k and (o.b is None or o.b >= game.C[o.L]):
            return "up"
        return "hit"
    return pi


def table_policy(A: np.ndarray, unit: int) -> Policy:
    """Closed-loop policy read from a DP action table A[b // unit, L]."""
    def pi(o: Obs) -> str:
        return str(A[o.b // unit, o.L])
    return pi
