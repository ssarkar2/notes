"""
Problem 03 (Ir-Ur-Bk-Ec-Oe-Kk), method: optimal stopping (structure of the DP policy).

Question: does the optimal policy from backward induction have a simple shape? The
candidate is a *threshold rule*, one number per level:

    at level L:  upgrade if b >= b*_L, otherwise hit.

If the thresholds rise with L, play looks like "keep trying to upgrade while rich
enough for your current level; once below the threshold, stop upgrading and harvest",
i.e. an optimal stopping problem (when to stop investing).

This script reads the thresholds off the DP action table, measures how far the exact
policy deviates from a clean threshold (the "flicker" band), and evaluates the
threshold rule exactly against the DP optimum.

Run:  python solve.py
"""
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common").is_dir())
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from common.loader import load  # noqa: E402
from common.model import EXAMPLE, Game  # noqa: E402

B_MAX = 200_000   # wider than elsewhere: the level-6 threshold sits near 100k
BI = "problems/03_Ir-Ur-Bk-Ec-Oe-Kk/backward_induction/solve.py"


def thresholds(game: Game, A: np.ndarray):
    """Per level L < N-1: (first budget where 'up' appears, budget from which 'up' always holds).

    Budgets in coins; None if 'up' never appears / never settles within the table.
    """
    du = game.unit
    out = []
    for L in range(game.N - 1):
        up = A[:, L] == "up"
        if not up.any():
            out.append((None, None))
            continue
        first = int(np.argmax(up))
        not_up = np.where(~up)[0]
        settled = int(not_up[-1] + 1) if len(not_up) else 0
        out.append((first * du, settled * du if settled < len(up) else None))
    return out


def threshold_policy(game: Game, b_star):
    def pi(b, L):
        if L < game.N - 1 and b_star[L] is not None and b >= b_star[L] and b >= game.C[L]:
            return "up"
        return "hit" if b >= game.h else "stop"
    return pi


def main():
    from common.plotting import ACCENT, plt, save

    bi = load(BI)
    game = EXAMPLE
    du = game.unit
    V, A = bi.solve(game, B_MAX)
    th = thresholds(game, A)

    print(f"{'level':>5} {'first up':>9} {'always up from':>15} {'flicker cells':>13}")
    for L, (first, settled) in enumerate(th):
        flick = 0 if first is None or settled is None else \
            int(np.sum(A[first // du:settled // du, L] != "up"))
        print(f"{L + 1:>5} {first!s:>9} {settled!s:>15} {flick:>13}")

    b_star = [s for _, s in th]
    Vt = bi.evaluate(game, threshold_policy(game, b_star), B_MAX)
    gap = V[:, 0] - Vt[:, 0]
    i = int(np.argmax(gap))
    print(f"\nthreshold rule vs DP over 0..{B_MAX:,}: max gap {gap[i]:.3f} stars at B={i * du:,} "
          f"(DP {V[i, 0]:.2f}); max relative gap {np.max(gap[V[:, 0] > 0] / V[V[:, 0] > 0, 0]):.2%}")
    for B in (5_000, 20_000):
        print(f"B={B}: DP {V[B // du, 0]:.2f}, threshold rule {Vt[B // du, 0]:.2f}")

    # where does adapting beat the best fixed plan? compare with the fixed plan's switch points
    Vfix, kfix = bi.best_fixed_target(game, 60_000)
    gain = V[: len(Vfix), 0] - Vfix
    switches = [int(b) * du for b in np.where(np.diff(kfix) > 0)[0] + 1]
    first_sw = sorted({s for s in switches if all(abs(s - t) > 2_000 for t in switches if t < s)})
    print(f"\nbest fixed target changes level at about: {first_sw}")
    print("largest gains from adapting (budget: gain):",
          [(int(b) * du, round(float(gain[b]), 2)) for b in np.argsort(gain)[-3:][::-1]])

    # ---- figure: thresholds -------------------------------------------------------------
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    levels = np.arange(1, game.N)
    first = np.array([f if f is not None else np.nan for f, _ in th], float)
    settled = np.array([s if s is not None else np.nan for _, s in th], float)
    ax.vlines(levels, first, settled, color=ACCENT, lw=6, alpha=0.35, label="flicker band")
    ax.plot(levels, settled, "o-", color=ACCENT, label=r"threshold $b^*_L$ (always upgrade above)")
    ax.set_yscale("log")
    ax.set_xticks(levels)
    ax.set_xlabel("core level L")
    ax.set_ylabel("coins remaining (log scale)")
    ax.set_title("(a) upgrade iff coins ≥ threshold for your level")
    ax.legend(fontsize=8)
    Bs = np.arange(len(gap)) * du
    m = Bs <= 60_000
    ax2.plot(Bs[m], gap[m], color=ACCENT)
    ax2.set_xlabel("budget B (coins)")
    ax2.set_ylabel("expected stars lost")
    ax2.set_title("(b) DP − threshold rule")
    save(fig, __file__, "thresholds")


if __name__ == "__main__":
    main()
