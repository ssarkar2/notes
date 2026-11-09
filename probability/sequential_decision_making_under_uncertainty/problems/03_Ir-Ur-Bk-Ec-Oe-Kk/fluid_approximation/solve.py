"""
Problem 03 (Ir-Ur-Bk-Ec-Oe-Kk), method: fluid / mean-value approximation.

Replace every random quantity by its mean and solve the resulting deterministic problem
with problem 01's tools. An upgrade from L succeeds after a geometric number of tries, so
its expected cost is C_L / U_L:

    V_MV(B) = max_k  r_k * floor( (B - Σ_{i<k} C_i/U_i) / h )

Three ways to use it, all compared with the exact DP:
  1. prediction: V_MV(B) as an estimate of the optimal value;
  2. open-loop:  pick the MV target k once, then play "try to upgrade to k, then hit";
  3. re-planning (certainty-equivalent control): at every state (b, L), re-solve the MV
     problem from there and upgrade iff its best target is above L.
Also the one-step break-even rule: upgrade from L iff
     b > (C_L/U_L) * r_{L+1} / (r_{L+1} - r_L).

Run:  python solve.py
"""
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common").is_dir())
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from common.loader import load  # noqa: E402
from common.model import EXAMPLE, Game  # noqa: E402

B_MAX = 120_000
BI = "problems/03_Ir-Ur-Bk-Ec-Oe-Kk/backward_induction/solve.py"
OS = "problems/03_Ir-Ur-Bk-Ec-Oe-Kk/optimal_stopping/solve.py"


def mv_per_target(game: Game, b: float, L: int = 0) -> np.ndarray:
    """MV value of 'upgrade from L to k, then hit' for k = L..N-1 (-inf if unaffordable)."""
    c = np.array(game.C) / np.array(game.U)              # expected cost per level
    cum = np.concatenate([[0], np.cumsum(c[L:])])         # expected cost from L to L+i
    hits = np.floor((b - cum) / game.h)
    return np.where(b >= cum, hits * game.r[L:], -np.inf)


def mv_value(game: Game, B: float) -> tuple[float, int]:
    v = mv_per_target(game, B)
    k = int(np.argmax(v))
    return float(max(v[k], 0.0)), k


def replan_policy(game: Game):
    """Certainty-equivalent control: re-solve the MV problem at every state."""
    def pi(b, L):
        if L < game.N - 1 and b >= game.C[L] and int(np.argmax(mv_per_target(game, b, L))) > 0:
            return "up"
        return "hit" if b >= game.h else "stop"
    return pi


def breakeven(game: Game) -> np.ndarray:
    """One-step rule thresholds: (C_L/U_L) r_{L+1} / (r_{L+1} - r_L)."""
    c = np.array(game.C) / np.array(game.U)
    r = game.r
    return c * r[1:] / (r[1:] - r[:-1])


def main():
    from common.plotting import ACCENT, plt, save

    bi, os_ = load(BI), load(OS)
    game = EXAMPLE
    du = game.unit
    V, A = bi.solve(game, B_MAX)
    Bs = np.arange(V.shape[0]) * du

    pred = np.array([mv_value(game, B)[0] for B in Bs])
    k_mv = np.array([mv_value(game, B)[1] for B in Bs])
    # open-loop MV plan: evaluate each fixed target once, then pick the MV-chosen one per budget
    Vk = np.stack([bi.evaluate(game, bi.target_policy(game, k), B_MAX)[:, 0]
                   for k in range(game.N)], 1)
    openloop = Vk[np.arange(len(Bs)), k_mv]
    replan = bi.evaluate(game, replan_policy(game), B_MAX)[:, 0]

    print(f"{'budget':>8} {'DP':>8} {'MV predicts':>11} {'MV open-loop':>12} {'MV re-plan':>10}")
    for B in [5_000, 10_000, 20_000, 50_000, 100_000]:
        i = B // du
        print(f"{B:>8} {V[i, 0]:>8.2f} {pred[i]:>11.2f} {openloop[i]:>12.2f} {replan[i]:>10.2f}")
    pos = V[:, 0] > 0
    for name, x in [("MV prediction error", pred - V[:, 0]), ("open-loop loss", V[:, 0] - openloop),
                    ("re-plan loss", V[:, 0] - replan)]:
        rel = np.abs(x[pos]) / V[pos, 0]
        print(f"{name:>20}: max |abs| {np.max(np.abs(x)):.2f} stars, max relative {rel.max():.2%}")

    # thresholds: DP vs re-planning vs one-step break-even
    th_dp = [s for _, s in os_.thresholds(game, A)]
    pi = replan_policy(game)
    A_rp = np.array([[pi(b * du, L) for L in range(game.N)] for b in range(V.shape[0])])
    th_rp = [s for _, s in os_.thresholds(game, A_rp)]
    th_be = breakeven(game)
    print(f"\n{'level':>5} {'DP':>9} {'MV re-plan':>11} {'break-even':>11}")
    for L in range(game.N - 1):
        print(f"{L + 1:>5} {th_dp[L]!s:>9} {th_rp[L]!s:>11} {th_be[L]:>11.0f}")

    # ---- figure ---------------------------------------------------------------------------
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    m = Bs <= 60_000
    ax.plot(Bs[m], pred[m] - V[m, 0], color="#999999", lw=0.5, alpha=0.6, label="MV prediction − DP (sawtooth: whole hits)")
    ax.plot(Bs[m], openloop[m] - V[m, 0], color="#2171b5", lw=1, label="MV open-loop plan − DP")
    ax.plot(Bs[m], replan[m] - V[m, 0], color=ACCENT, lw=1.5, label="MV re-planning − DP")
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xlabel("budget B (coins)")
    ax.set_ylabel("expected stars (difference from DP)")
    ax.set_title("(a) how good is the mean-value approximation?")
    ax.legend(fontsize=8)
    levels = np.arange(1, game.N)
    f = lambda xs: np.array([x if x is not None else np.nan for x in xs], float)  # noqa: E731
    ax2.plot(levels, f(th_dp), "o-", color="black", label="DP (exact)")
    ax2.plot(levels, f(th_rp), "s--", color=ACCENT, label="MV re-planning")
    ax2.plot(levels, th_be, "^:", color="#2171b5", label="one-step break-even rule")
    ax2.set_yscale("log")
    ax2.set_xticks(levels)
    ax2.set_xlabel("core level L")
    ax2.set_ylabel("upgrade threshold (coins, log)")
    ax2.set_title("(b) upgrade thresholds")
    ax2.legend(fontsize=8)
    save(fig, __file__, "mv_vs_dp")


if __name__ == "__main__":
    main()
