"""
Problem 06 (Ir-Ur-Bh-Ec-Oe-Kk), method: survival analysis.

The hidden budget B is a "lifetime" measured in coins. Two summaries of its distribution
drive the decisions:

    hazard        lambda_c(s) = P(B < s + c | B >= s) = 1 - S(s + c) / S(s)
                  chance that an action costing c is refused, given you got to s
    mean residual m(s)        = E[B - s | B >= s]
                  expected coins left, given you got to s

Heuristic ("plug in the mean"): at each step pretend the budget left is exactly m(s) and
play 03's optimal policy for that budget. Compared with:
  - the exact optimum (belief MDP, ../belief_mdp/solve.py);
  - a naive plug-in that never updates: pretend the budget left is E[B] - s.

All values are exact (policy evaluation on (s, L)).

Run:  python solve.py
"""
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common").is_dir())
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from common.loader import load  # noqa: E402
from common.model import EXAMPLE  # noqa: E402

BM = "problems/06_Ir-Ur-Bh-Ec-Oe-Kk/belief_mdp/solve.py"


def hazard(S: np.ndarray, c: int) -> np.ndarray:
    """1 - S(s+c)/S(s) for every grid point s with S(s) > 0 (nan elsewhere); c in grid units."""
    S = np.concatenate([S, np.zeros(c)])
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1 - S[c:] / S[:-c]


def mean_residual(pmf: np.ndarray) -> np.ndarray:
    """m(s) = E[B - s | B >= s] in grid units (nan where S(s) = 0)."""
    k = np.arange(len(pmf))
    S = np.cumsum(pmf[::-1])[::-1]
    tail = np.cumsum((pmf * k)[::-1])[::-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        return tail / S - k


def plug_in_policy(A03: np.ndarray, b_hat: np.ndarray) -> np.ndarray:
    """(s, L) action table: 03's action at budget b_hat(s). 'stop' -> 'hit' (with a hidden
    budget, stopping early never helps: leftover coins are worth nothing)."""
    bh = np.clip(np.floor(np.nan_to_num(b_hat) + 1e-9).astype(int), 0, len(A03) - 1)
    pol = A03[bh].copy()
    pol[pol == "stop"] = "hit"
    return pol


def main():
    from common.plotting import ACCENT, plt, save

    bm = load(BM)
    p05 = load(bm.P05)
    game = EXAMPLE
    du = game.unit
    V03, A03 = load(bm.BI).solve(game, p05.B_MAX)
    dists = p05.budget_distributions(du)
    mean_idx = p05.MEAN // du

    print(f"{'prior':>26} {'seen (05)':>9} {'optimal':>8} {'plug-in m(s)':>12} {'plug-in E[B]-s':>14}")
    curves = {}
    for name, p in dists.items():
        k = np.arange(len(p))
        V, _ = bm.solve(game, p)
        m = mean_residual(p)
        v_m = bm.solve(game, p, plug_in_policy(A03, m))[0][0, 0]
        v_0 = bm.solve(game, p, plug_in_policy(A03, np.maximum(mean_idx - k, 0)))[0][0, 0]
        seen = (p * V03[:, 0]).sum()
        print(f"{name:>26} {seen:>9.2f} {V[0, 0]:>8.2f} {v_m:>12.2f} {v_0:>14.2f}")
        S = np.cumsum(p[::-1])[::-1]
        curves[name] = (hazard(S, game.h // du), m * du)

    # ---- figure: hazard and mean residual budget ----------------------------------------
    s = np.arange(len(next(iter(dists.values())))) * du
    keep = s <= 20_000
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    colors = ["black", "#2171b5", ACCENT, "#6a3d9a"]
    for (name, (lam, m)), col in zip(curves.items(), colors):
        ax.plot(s[keep], lam[keep], color=col, lw=1.4, label=name)
        ax2.plot(s[keep], m[keep], color=col, lw=1.4, label=name)
    ax.set_xlabel("coins spent s")
    ax.set_ylabel("P(next hit refused | got to s)")
    ax.set_title("(a) hazard for one hit (cost 200)")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1.5)
    ax2.set_xlabel("coins spent s")
    ax2.set_ylabel("coins")
    ax2.set_title("(b) mean residual budget m(s) = E[B − s | B ≥ s]")
    ax2.legend(fontsize=8, loc="upper right")
    for a in (ax, ax2):
        a.set_xticks(range(0, 20_001, 5_000))
        a.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    save(fig, __file__, "hazard_residual")


if __name__ == "__main__":
    main()
