"""
Problem 09 (Ir-Ur-Bk-Ei-Oe-Kk): items pay income. Method: time-indexed DP on a bigger state.

Same clock as 08: T ticks, ONE action per tick (hit, upgrade, wait), leftover coins worth
nothing. You start with B0 coins. No outside income; instead every item you hold pays
g_j coins per tick. In the running example only 5-star items pay (g_5 coins per tick),
so the income is g_5 * n, where n = number of 5-star items held. Total stars still count
every item.

State (t, b, n, L), b = coins before this tick's income:

    b' = b + g_5 n                                                     # income arrives
    V(t, b, n, L) = max{ V(t+1, b', n, L),                                                  # wait
                         r_L + P(L,5) V(t+1, b'-h, n+1, L) + (1-P(L,5)) V(t+1, b'-h, n, L),  # hit
                         U_L V(t+1, b'-C_L, n, L+1) + (1-U_L) V(t+1, b'-C_L, n, L) }       # upgrade
    V(T, ., ., .) = 0

Backward in t, as in 08. The new part is n: an item is now also capital.

Run:  python solve.py
"""
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common").is_dir())
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from common.loader import load  # noqa: E402
from common.model import EXAMPLE, Game  # noqa: E402

TID = "problems/08_Ir-Ur-Bt-Ec-Oe-Kk/time_indexed_dp/solve.py"
WAIT, HIT, UP = 0, 1, 2
NAMES = np.array(["wait", "hit", "up"])
B0, T = 10_000, 50


def solve(game: Game, g5: int, T: int = T, B0: int = B0, keep_V: bool = False, b_keep: int = 5_000):
    """Optimal V0 = V[0, B0, 0, 0] and action table A[t, b_idx, n, L].

    b_idx = b / Δ with Δ = gcd(costs, g5). n = 5-star items held, 0..T.
    With keep_V, also returns V[t] for every t, restricted to b <= b_keep and n <= 3
    (for shadow prices).
    """
    from math import gcd
    du = gcd(game.unit, g5) if g5 else game.unit
    nb = (B0 + g5 * T * T) // du + 1                       # n <= t, so income <= g5 * T per tick
    nn = T + 1
    h, C, g = game.h // du, [c // du for c in game.C], g5 // du
    p5 = game.P[:, -1]
    V = np.zeros((nb, nn, game.N))
    A = np.zeros((T, nb, nn, game.N), dtype=np.int8)
    Vs = [None] * (T + 1) if keep_V else None
    keep = lambda V: V[: b_keep // du + 1, :4].copy()      # noqa: E731
    if keep_V:
        Vs[T] = keep(V)
    bi = np.arange(nb)[:, None]
    ni = np.arange(nn)[None, :]
    b1 = np.minimum(bi + g * ni, nb - 1)                    # (nb, nn) coins after income
    n_up = np.minimum(ni + 1, nn - 1)
    for t in range(T - 1, -1, -1):
        Q = np.full((3, nb, nn, game.N), -np.inf)
        Q[WAIT] = V[b1, np.broadcast_to(ni, b1.shape)]
        ok = b1 >= h
        bh = np.where(ok, b1 - h, 0)
        nfull = np.broadcast_to(ni, b1.shape)
        stay = V[bh, nfull]                                  # (nb, nn, N)
        new5 = V[bh, np.broadcast_to(n_up, b1.shape)]
        hit = game.r + p5 * new5 + (1 - p5) * stay
        Q[HIT] = np.where(ok[..., None], hit, -np.inf)
        for L in range(game.N - 1):
            ok = b1 >= C[L]
            bc = np.where(ok, b1 - C[L], 0)
            after = V[bc, nfull]
            up = game.U[L] * after[..., L + 1] + (1 - game.U[L]) * after[..., L]
            Q[UP, ..., L] = np.where(ok, up, -np.inf)
        a = Q.argmax(0)                                      # ties: wait, then hit
        V = np.take_along_axis(Q, a[None], 0)[0]
        A[t] = a
        if keep_V:
            Vs[t] = keep(V)
    out = dict(V0=V[B0 // du, 0, 0], A=A, unit=du, V=Vs)
    return out


def income_policy(A: np.ndarray, unit: int, g5: int):
    """Plays the table in a simulator that tracks n (5-star items held)."""
    def pi(t, b_after_income, n, L):
        return int(A[t, (b_after_income - g5 * n) // unit, n, L])
    return pi


def play(game: Game, policy, g5: int, rng, T: int = T, B0: int = B0):
    """One game with item income. Returns (total stars, n_5 held at the end, final level,
    first tick at level >= 5 or -1)."""
    b, n, L, stars, first5 = B0, 0, 0, 0, -1
    for t in range(T):
        b += g5 * n
        a = policy(t, b, n, L)
        if a == HIT and b >= game.h:
            b -= game.h
            j = rng.choice(game.M, p=game.P[L])
            stars += j + 1
            n += j == game.M - 1
        elif a == UP and b >= game.C[L]:
            b -= game.C[L]
            if rng.random() < game.U[L]:
                L += 1
                if L >= 4 and first5 < 0:
                    first5 = t
    return stars, n, L, first5


def main():
    from common.plotting import ACCENT, LEVEL_COLORS, plt, save
    from common.simulate import summary

    game = EXAMPLE
    tid = load(TID)
    print(f"B0 = {B0:,}, T = {T} ticks, one action per tick; only 5-star items pay income")
    V08, _ = tid.solve(game, 0, T, B0)                       # 08's solver, no income
    Vtick, _ = tid.solve(game, max(game.h, *game.C), T)      # coins never short
    print(f"no income (08 solver, rho = 0): {V08[0, B0 // 100, 0]:.2f};  tick bound: {Vtick[0, 0, 0]:.2f}")

    rng = np.random.default_rng(0)
    rows, prices = [], {}
    REF_B, REF_L = 1_000, 4                                  # reference state: 1,000 coins, level 5
    for g5 in (0, 25, 50, 100, 200):
        out = solve(game, g5, keep_V=g5 in (50, 100, 200))
        du = out["unit"]
        pol = income_policy(out["A"], du, g5)
        sims = np.array([play(game, pol, g5, rng) for _ in range(4000)])
        rows.append((g5, out["V0"]))
        print(f"g5 = {g5:>3}: V = {out['V0']:.2f}   sim {summary(sims[:, 0])}, "
              f"5-star items held {sims[:, 1].mean():.2f}, final level {sims[:, 2].mean() + 1:.2f}, "
              f"P(reach level 5) {(sims[:, 3] >= 0).mean():.3f}, "
              f"median tick reaching it {np.median(sims[sims[:, 3] >= 0, 3]) if (sims[:, 3] >= 0).any() else '-'}")
        if out["V"] is not None:
            Vs = np.array(out["V"])                         # (T+1, b, n, L)
            b = REF_B // du
            item = Vs[:, b, 1, REF_L] - Vs[:, b, 0, REF_L]
            coin = (Vs[:, b + 1_000 // du, 0, REF_L] - Vs[:, b, 0, REF_L]) / 10   # per 100 coins
            left = T - np.arange(T + 1)
            prices[g5] = (left, item, coin)
            last_up = max(t for t in range(T) if (out["A"][t] == UP).any())
            print(f"    at {REF_B:,} coins, level {REF_L + 1}: value of one more 5-star item held "
                  f"(future stars) with 50/30/10/5 ticks left: "
                  f"{item[0]:.1f} / {item[20]:.1f} / {item[40]:.1f} / {item[45]:.1f};  "
                  f"100 coins: {coin[0]:.2f} / {coin[20]:.2f} / {coin[40]:.2f} / {coin[45]:.2f}")
            print(f"    last upgrade anywhere in the table: {T - last_up} ticks left")

    # ---- figure ---------------------------------------------------------------------------
    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
    g_, v_ = zip(*rows)
    ax.plot(g_, v_, "o-", color=ACCENT, label="09 exact")
    ax.axhline(V08[0, B0 // 100, 0], color="black", ls="--", lw=1, label="no income (03 with 10,000)")
    ax.axhline(Vtick[0, 0, 0], color="#2171b5", ls=":", lw=1.5, label="tick bound (coins never short)")
    ax.set_xlabel("income per 5★ item held, $g_5$ (coins per tick)")
    ax.set_ylabel("expected total stars")
    ax.set_ylim(0, 180)
    ax.set_title("(a) value")
    ax.legend(fontsize=8, loc="lower right")
    cols = {50: "#9ecae1", 100: "#2171b5", 200: "#08306b"}
    for g5, (left, item, coin) in prices.items():
        ax2.plot(left, item, color=cols[g5], label=f"$g_5$ = {g5}")
        ax3.plot(left, coin, color=cols[g5], label=f"$g_5$ = {g5}")
    for a in (ax2, ax3):
        a.invert_xaxis()
        a.set_xlabel(f"ticks left   (state: {REF_B:,} coins, level {REF_L + 1}, no 5★ yet)")
        a.legend(fontsize=8)
    ax2.set_ylabel("extra expected stars")
    ax2.set_title("(b) one more 5★ item held (its income)")
    ax3.set_ylabel("extra expected stars")
    ax3.set_title("(c) 100 more coins")
    save(fig, __file__, "income")


if __name__ == "__main__":
    main()
