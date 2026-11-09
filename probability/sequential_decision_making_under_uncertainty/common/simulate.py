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
    t: int = 0      # tick (problems with time, axis Bt); 0 otherwise


Policy = Callable[[Obs], str]


def play_once(game: Game, policy: Policy, budget: int, rng: np.random.Generator,
              hidden_budget: bool = False, max_steps: int = 10**6, utility=None,
              income: int = 0, ticks: int | None = None) -> float:
    """Play one game; return total stars collected, or utility(items) if given.

    With `ticks` (axis Bt): the game lasts that many ticks; each tick `income` coins
    arrive, then the policy takes ONE action ("hit", "up", or "wait").
    """
    b, s, L, value, top = budget, 0, 0, 0.0, 0
    items = []
    r = game.r
    for t in range(max_steps if ticks is None else ticks):
        b += income
        a = policy(Obs(None if hidden_budget else b, s, L, value, top, t))
        if a == "wait":
            continue
        if a == "stop":
            break
        cost = game.h if a == "hit" else game.C[L]
        if a == "up" and L >= game.N - 1:
            raise ValueError("policy tried to upgrade past the top level")
        if cost > b:            # cannot afford: the game ends (hidden budget learns it here)
            if ticks is not None:
                continue        # with income, more coins may come: the action is just refused
            break
        b, s = b - cost, s + cost
        if a == "hit":
            if game.items_deterministic:
                value += r[L]
            else:
                jt = rng.choice(game.M, p=game.P[L])
                j = int(game.stars[jt])
                b, s = b + game.refund[jt], s - game.refund[jt]   # Er: refund (s = net spent)
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
             hidden_budget: bool = False, utility=None, income: int = 0,
             ticks: int | None = None) -> np.ndarray:
    """Play n games. `budget` is an int, or a callable rng -> int (random budget)."""
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        B = budget(rng) if callable(budget) else budget
        out[i] = play_once(game, policy, B, rng, hidden_budget, utility=utility,
                           income=income, ticks=ticks)
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
