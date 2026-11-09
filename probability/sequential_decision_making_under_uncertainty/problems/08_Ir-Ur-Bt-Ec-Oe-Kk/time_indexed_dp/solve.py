"""
Problem 08 (Ir-Ur-Bt-Ec-Oe-Kk), method: time-indexed DP.

Coins arrive from outside: rho coins at the start of each tick, for T ticks, on top of a
starting budget B0. Each tick you take ONE action: hit, upgrade, or wait. Leftover coins
at the end are worth nothing.

State (t, b, L), b = coins before this tick's income:

    V(t, b, L) = max{ V(t+1, b+rho, L),                                            # wait
                      r_L + V(t+1, b+rho-h, L),                                    # hit
                      U_L V(t+1, b+rho-C_L, L+1) + (1-U_L) V(t+1, b+rho-C_L, L) }  # upgrade
    V(T, b, L) = 0

Coins go up (income), but time only goes forward, so fill the table from t = T down:
backward induction in t, not in b.

Without the one-action-per-tick limit this is just 03 with budget B0 + rho T (waiting is
free, so save everything and play 03 at the end). The limit is what makes time matter.

Run:  python solve.py
"""
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common").is_dir())
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from common.loader import load  # noqa: E402
from common.model import EXAMPLE, Game  # noqa: E402

BI = "problems/03_Ir-Ur-Bk-Ec-Oe-Kk/backward_induction/solve.py"
WAIT, HIT, UP = 0, 1, 2
NAMES = np.array(["wait", "hit", "up"])


def solve(game: Game, rho: int, T: int, B0: int = 0):
    """V[t, b_idx, L] for t = 0..T and A[t, b_idx, L] (0 wait, 1 hit, 2 up), b = b_idx * Δ."""
    du = game.unit
    assert rho % du == 0 and B0 % du == 0
    nb = (B0 + rho * T) // du + 1
    p, h, C = rho // du, game.h // du, [c // du for c in game.C]
    V = np.zeros((T + 1, nb, game.N))
    A = np.zeros((T, nb, game.N), dtype=np.int8)
    idx = np.minimum(np.arange(nb) + p, nb - 1)          # coins after income (capped: unreachable)
    for t in range(T - 1, -1, -1):
        Vn = V[t + 1]
        Q = np.full((3, nb, game.N), -np.inf)
        Q[WAIT] = Vn[idx]
        ok = idx >= h
        Q[HIT, ok] = game.r[None, :] + Vn[idx[ok] - h]
        for L in range(game.N - 1):
            ok = idx >= C[L]
            after = Vn[idx[ok] - C[L]]
            Q[UP, ok, L] = game.U[L] * after[:, L + 1] + (1 - game.U[L]) * after[:, L]
        a = Q.argmax(0)                                   # ties: wait, then hit
        V[t] = np.take_along_axis(Q, a[None], 0)[0]
        A[t] = a
    return V, A


def time_policy(A: np.ndarray, unit: int, rho: int):
    """Plays the table in the simulator. The simulator adds the income before asking, so
    look up the coins held before this tick's income."""
    def pi(o):
        return str(NAMES[A[o.t, (o.b - rho) // unit, o.L]])
    return pi


def tick_thresholds(game: Game, T: int) -> list:
    """Coins never short (income >= the dearest action): upgrade from L iff ticks left >= tau_L.

    Returns tau_L per level (None if never), checked to be a threshold rule.
    """
    rho = max(game.h, *game.C)
    _, A = solve(game, rho, T)
    out = []
    for L in range(game.N - 1):
        up = np.array([A[t, -1, L] == UP for t in range(T)])     # b index irrelevant: coins plenty
        left = T - np.arange(T)
        if not up.any():
            out.append(None)
            continue
        tau = left[up].min()
        assert (up == (left >= tau)).all(), "not a threshold rule"
        out.append(int(tau))
    return out


def main():
    from common.plotting import ACCENT, plt, save
    from common.simulate import simulate, summary

    game, du = EXAMPLE, EXAMPLE.unit
    big = max(game.h, *game.C)                       # income that never limits an action
    T_MAX = 200
    V03, _ = load(BI).solve(game, 1000 * T_MAX)
    Vtick, _ = solve(game, big, T_MAX)               # tick bound for every T at once
    coin_bound = lambda rho, T: V03[rho * T // du, 0]          # noqa: E731
    tick_bound = lambda T: Vtick[T_MAX - T, -1, 0]             # noqa: E731

    print("reduction check: rho <= h  =>  08 = 03 with rho*T coins")
    for rho in (100, 200):
        for T in (10, 50, 100, 200):
            V, _ = solve(game, rho, T)
            print(f"  rho {rho}, T {T:>3}: 08 {V[0, 0, 0]:9.4f}   03 {coin_bound(rho, T):9.4f}")

    T = 50
    print(f"\nT = {T} ticks, income rho per tick:")
    print(f"{'rho':>6} {'08 exact':>9} {'coin bound':>10} {'tick bound':>10}   simulator")
    by_rho = []
    for rho in (100, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000, 2500):
        V, A = solve(game, rho, T)
        by_rho.append((rho, V[0, 0, 0], coin_bound(rho, T), tick_bound(T)))
        sim = ""
        if rho in (200, 400, 1000):
            sim = summary(simulate(game, time_policy(A, du, rho), 0, n=4000, seed=0,
                                   income=rho, ticks=T))
        print(f"{rho:>6} {V[0, 0, 0]:>9.2f} {coin_bound(rho, T):>10.2f} {tick_bound(T):>10.2f}   {sim}")

    rho = 400
    print(f"\nrho = {rho}, T ticks:")
    by_T = []
    for T in (5, 10, 20, 30, 40, 50, 60, 80, 100, 150, 200):
        V, _ = solve(game, rho, T)
        by_T.append((T, V[0, 0, 0], coin_bound(rho, T), tick_bound(T)))
        print(f"  T {T:>3}: 08 {V[0, 0, 0]:8.2f}  coin bound {coin_bound(rho, T):8.2f}  "
              f"tick bound {tick_bound(T):8.2f}  ratio to min {V[0, 0, 0] / min(coin_bound(rho, T), tick_bound(T)):.3f}")

    print("\ncoins never short: upgrade from level L while ticks left >=", tick_thresholds(game, T_MAX))

    # ---- figure ---------------------------------------------------------------------------
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    x, v, cb, tb = map(np.array, zip(*by_rho))
    ax.plot(x, cb, color="black", ls="--", lw=1, label="coin bound: 03 with all coins up front")
    ax.plot(x, tb, color="#2171b5", ls=":", lw=1.5, label="tick bound: coins never short")
    ax.plot(x, v, "o-", color=ACCENT, ms=4, label="08 exact")
    ax.axvline(game.h, color="grey", lw=0.8)
    ax.annotate("ρ = h", (game.h, 8), xytext=(game.h + 40, 8), fontsize=8, color="grey")
    ax.set_ylim(0, 260)
    ax.set_xlabel("income ρ per tick (coins)")
    ax.set_ylabel("expected total stars")
    ax.set_title("(a) T = 50 ticks, vary income")
    ax.legend(fontsize=8, loc="lower right")
    Ts = np.array([5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 125, 150, 200])
    for rho, col in [(300, "#9ecae1"), (400, "#2171b5"), (600, "#08306b")]:
        frac = [solve(game, rho, T)[0][0, 0, 0] / min(coin_bound(rho, T), tick_bound(T)) for T in Ts]
        ax2.plot(Ts, frac, "o-", ms=3, color=col, label=f"ρ = {rho}")
    ax2.set_xlabel("ticks T")
    ax2.set_ylabel("08 / smaller of the two bounds")
    ax2.set_title("(b) how far below both bounds")
    ax2.legend(fontsize=8)
    save(fig, __file__, "time_indexed")


if __name__ == "__main__":
    main()
