"""
Problem 10 (Ir-Ur-Bk-Es-Oe-Kk), method: DP with an inventory state.

Items can be sold for coins; a sold item no longer counts. In the running example only
1-star items have a buyer, at s_1 coins each (s_1 < h). Selling is instant and can happen
at any time, so unsold 1-star items are a reserve of coins.

State (b, n, L), n = unsold 1-star items held. Actions: sell one 1-star item, hit,
upgrade, stop:

    V(b, n, L) = max{ -1 + V(b + s_1, n - 1, L),                                 # sell (n >= 1)
                      r_L + p_1 V(b - h, n + 1, L) + (1 - p_1) V(b - h, n, L),   # hit, p_1 = P(L, 1)
                      U_L V(b - C_L, n, L+1) + (1 - U_L) V(b - C_L, n, L),       # upgrade
                      0 }                                                        # stop

Coins can go up (selling), but "liquid wealth" w = b + s_1 n never does: selling keeps it,
a hit lowers it by at least h - s_1 > 0, an upgrade by C_L. And selling lowers n. So
fill the table in order of w, and for equal w in order of n: one pass, no iteration.

Run:  python solve.py
"""
import sys
from math import gcd
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common").is_dir())
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from common.loader import load  # noqa: E402
from common.model import EXAMPLE, LONG_SHOT, Game  # noqa: E402

BI = "problems/03_Ir-Ur-Bk-Ec-Oe-Kk/backward_induction/solve.py"
STOP, SELL, HIT, UP = 0, 1, 2, 3
NAMES = ["stop", "sell", "hit", "up"]


def solve(game: Game, s1: int, B_max: int, policy=None):
    """V[b_idx, n, L], A[b_idx, n, L] for all b + s1 n <= B_max (grid Δ = gcd(costs, s1)).

    With `policy(b, n, L) -> action code` it evaluates that policy instead.
    """
    assert 0 < s1 < game.h
    du = gcd(game.unit, s1)
    sv, h, C = s1 // du, game.h // du, [c // du for c in game.C]
    Wmax = B_max // du
    nmax = Wmax // sv
    p1, r, U = game.P[:, 0], game.r, game.U
    V = np.zeros((Wmax + 1, nmax + 2, game.N))
    A = np.zeros((Wmax + 1, nmax + 2, game.N), dtype=np.int8)
    for w in range(Wmax + 1):                     # liquid wealth, in grid units
        for n in range(w // sv + 1):              # within equal w: n ascending (sell -> n-1)
            b = w - sv * n
            for L in range(game.N):
                q = [0.0, -np.inf, -np.inf, -np.inf]
                if n >= 1:
                    q[SELL] = -1 + V[b + sv, n - 1, L]
                if b >= h:
                    q[HIT] = r[L] + p1[L] * V[b - h, n + 1, L] + (1 - p1[L]) * V[b - h, n, L]
                if L < game.N - 1 and b >= C[L]:
                    q[UP] = U[L] * V[b - C[L], n, L + 1] + (1 - U[L]) * V[b - C[L], n, L]
                if policy is not None:
                    a = policy(b * du, n, L)
                else:
                    a = int(np.argmax(q))         # ties: stop, then sell, then hit
                V[b, n, L], A[b, n, L] = q[a], a
    return V, A, du


def exchange_rule_policy(game: Game, s1: int, A03: np.ndarray):
    """Heuristic: sell a 1-star item iff its coins buy more stars than it holds,
    s_1 r_L / h > 1; otherwise play 03's policy on the coins in hand."""
    code = {"stop": STOP, "hit": HIT, "up": UP}

    def pi(b, n, L):
        if n >= 1 and s1 * game.r[L] / game.h > 1:
            return SELL
        return code[str(A03[b // 100, L])]
    return pi


def play(game: Game, A: np.ndarray, du: int, s1: int, B: int, rng, n0: int = 0) -> tuple:
    """One game under action table A, starting with n0 unsold 1-star items.
    Returns (stars kept, items sold)."""
    b, n, L, stars, sold = B, n0, 0, n0, 0
    while True:
        a = A[b // du, n, L]
        if a == STOP:
            return stars, sold
        if a == SELL:
            b, n, stars, sold = b + s1, n - 1, stars - 1, sold + 1
        elif a == HIT:
            b -= game.h
            j = rng.choice(game.M, p=game.P[L]) + 1
            stars += j
            n += j == 1
        else:
            b -= game.C[L]
            if rng.random() < game.U[L]:
                L += 1


def main():
    from common.plotting import ACCENT, plt, save
    from common.simulate import summary

    game = EXAMPLE
    B_max = 20_000
    V03, A03 = load(BI).solve(game, B_max)
    rng = np.random.default_rng(0)
    tables = {}

    # ---- running example: start with coins only ----------------------------------------
    print("Running example, start with B coins and no items.")
    for s1 in (50, 100, 150):
        V, A, du = solve(game, s1, B_max)
        Vh, _, _ = solve(game, s1, B_max, policy=exchange_rule_policy(game, s1, A03))
        tables[s1] = (V, A, du, Vh)
        lv = [L + 1 for L in range(game.N) if s1 * game.r[L] / game.h > 1]
        print(f"\ns1 = {s1}: one sale buys more stars than it costs from level {lv[0] if lv else '-'}")
        print(f"{'budget':>8} {'never sell (03)':>15} {'10 exact':>9} {'exchange rule':>13}   simulator")
        for B in (5_000, 10_000, 20_000):
            i = B // du
            sims = np.array([play(game, A, du, s1, B, rng) for _ in range(2000)])
            print(f"{B:>8,} {V03[B // 100, 0]:>15.2f} {V[i, 0, 0]:>9.2f} {Vh[i, 0, 0]:>13.2f}   "
                  f"{summary(sims[:, 0])}, items sold {sims[:, 1].mean():.1f}")

    # ---- LONG_SHOT (03's stress example): does selling help there? ---------------------
    Vls, _ = load(BI).solve(LONG_SHOT, B_max)
    diffs = []
    for s1 in (50, 100, 150):
        V, _, du = solve(LONG_SHOT, s1, B_max)
        Bs = np.arange(0, B_max + 1, 1_000)
        diffs.append(np.abs(V[Bs // du, 0, 0] - Vls[Bs // LONG_SHOT.unit, 0]).max())
    print(f"\nLONG_SHOT, s1 = 50/100/150: largest gain from selling at budgets 0..20,000 "
          f"(every 1,000): {max(diffs):.2e}")

    # ---- stress example: start with an inventory ---------------------------------------
    print("\nStress example: start with n0 one-star items and 2,000 coins (totals include kept items).")
    B = 2_000
    print(f"{'s1':>4} {'n0':>4} {'never sell':>10} {'sell all, then 03':>17} {'10 exact':>9} "
          f"{'exchange rule':>13} {'sold at start':>13}   simulator")
    curves = {}
    for s1 in (100, 150):
        V, A, du, Vh = tables[s1]
        sv = s1 // du
        n0s = np.arange(0, 101, 5)
        never = V03[B // 100, 0] + n0s
        sell_all = V03[(B + s1 * n0s) // 100, 0]
        exact = V[B // du, n0s, 0] + n0s
        exch = Vh[B // du, n0s, 0] + n0s
        curves[s1] = (n0s, never, sell_all, exact, exch)
        for n0 in (50, 100):
            k = list(n0s).index(n0)
            b, n, sold = B // du, n0, 0
            while A[b, n, 0] == SELL:
                b, n, sold = b + sv, n - 1, sold + 1
            sims = np.array([play(game, A, du, s1, B, rng, n0) for _ in range(2000)])
            print(f"{s1:>4} {n0:>4} {never[k]:>10.2f} {sell_all[k]:>17.2f} {exact[k]:>9.2f} "
                  f"{exch[k]:>13.2f} {sold:>13}   {summary(sims[:, 0])}, sold in total {sims[:, 1].mean():.1f}")

    # ---- figure ---------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, s1 in zip(axes, (100, 150)):
        n0s, never, sell_all, exact, exch = curves[s1]
        ax.plot(n0s, never, color="black", ls="--", lw=1, label="never sell")
        ax.plot(n0s, sell_all, color="#2171b5", lw=1, label="sell all first, then 03")
        ax.plot(n0s, exch, color="grey", ls=":", lw=1.5, label="exchange rule (one-step)")
        ax.plot(n0s, exact, color=ACCENT, lw=2, label="10 exact")
        ax.set_xlabel("1★ items you start with ($n_0$)")
        ax.set_title(f"sale price $s_1$ = {s1} (start: 2,000 coins)")
    axes[0].set_ylabel("expected total stars (kept items)")
    axes[0].legend(fontsize=8)
    save(fig, __file__, "inventory")


if __name__ == "__main__":
    main()
