"""
Problem 03 (Ir-Ur-Bk-Ec-Oe-Kk), mechanics variants: demotion and pity.

Both are rule tweaks, not new axes, and both are solved by the same backward induction:

  * demotion: a failed upgrade from L drops the core to L-1 with probability D_L.
    Still a DAG in the budget b (every action costs coins), so the table order still
    works; only the upgrade term gets a third outcome. What changes is the *approximation*:
    the expected cost to climb is no longer sum C/U but an expected hitting time of a
    birth-death chain (a small linear system).

  * pity: the success chance rises with consecutive failures f at the current level,
    U_L(f) = min(1, U_L + DELTA * f), reset on success. The odds depend on history, so f
    joins the state: (b, L, f).

Run:  python solve.py
"""
import sys
from dataclasses import replace
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common").is_dir())
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from common.loader import load  # noqa: E402
from common.model import EXAMPLE, Game  # noqa: E402

B_MAX = 100_000
BI = "problems/03_Ir-Ur-Bk-Ec-Oe-Kk/backward_induction/solve.py"
OS = "problems/03_Ir-Ur-Bk-Ec-Oe-Kk/optimal_stopping/solve.py"
DEMOTE = 0.3          # D_L for every level that can drop (level 2 and up)
DELTA = 0.1           # pity: +10 percentage points per consecutive failure


# ---- demotion ------------------------------------------------------------------------------
def demotion_game(game: Game, d: float = DEMOTE) -> Game:
    return replace(game, D=(0.0,) + (d,) * (game.N - 2))


def hitting_cost(game: Game, k: int) -> float:
    """Expected coins to climb from level 1 to level k+1 (0-indexed k) by always upgrading.

    T_L = C_L + U_L T_{L+1} + (1-U_L)(1-D_L) T_L + (1-U_L) D_L T_{L-1},  T_k = 0.
    """
    if k == 0:
        return 0.0
    M = np.zeros((k, k))
    rhs = np.zeros(k)
    for L in range(k):
        u, d = game.U[L], game.D[L]
        M[L, L] = 1 - (1 - u) * (1 - d) - ((1 - u) * d if L == 0 else 0.0)
        if L + 1 < k:
            M[L, L + 1] = -u
        if L > 0:
            M[L, L - 1] = -(1 - u) * d
        rhs[L] = game.C[L]
    return float(np.linalg.solve(M, rhs)[0])


# ---- pity ----------------------------------------------------------------------------------
def pity_prob(game: Game, L: int, f: int) -> float:
    return min(1.0, game.U[L] + DELTA * f)


def solve_pity(game: Game, B_max: int = B_MAX):
    """V[b, L, f], A[b, L, f] for the pity mechanic."""
    du = game.unit
    nb = B_max // du + 1
    F = int(np.ceil((1 - min(game.U)) / DELTA))          # beyond this, success is certain
    h, C = game.h // du, [c // du for c in game.C]
    r = game.r
    V = np.zeros((nb, game.N, F + 1))
    A = np.full((nb, game.N, F + 1), "stop", dtype="<U4")
    for b in range(nb):
        for L in range(game.N):
            for f in range(F + 1):
                best, act = 0.0, "stop"
                if b >= h:
                    v = r[L] + V[b - h, L, f]
                    if v > best:
                        best, act = v, "hit"
                if L < game.N - 1 and b >= C[L]:
                    p = pity_prob(game, L, f)
                    v = p * V[b - C[L], L + 1, 0] + (1 - p) * V[b - C[L], L, min(f + 1, F)]
                    if v > best:
                        best, act = v, "up"
                V[b, L, f], A[b, L, f] = best, act
    return V, A


def simulate_pity(game: Game, A: np.ndarray, B: int, n: int = 2000, seed: int = 0) -> np.ndarray:
    """Play the pity-DP policy; return total stars per game (checks solve_pity)."""
    rng = np.random.default_rng(seed)
    du, F = game.unit, A.shape[2] - 1
    out = np.empty(n)
    for i in range(n):
        b, L, f, stars = B, 0, 0, 0.0
        while (a := A[b // du, L, f]) != "stop":
            if a == "hit":
                b -= game.h
                stars += game.stars[rng.choice(game.M, p=game.P[L])]
            else:
                b -= game.C[L]
                if rng.random() < pity_prob(game, L, f):
                    L, f = L + 1, 0
                else:
                    f = min(f + 1, F)
        out[i] = stars
    return out


def main():
    from common.plotting import ACCENT, LEVEL_COLORS, plt, save
    from common.simulate import simulate, summary, table_policy

    bi, os_ = load(BI), load(OS)
    game = EXAMPLE
    du = game.unit
    V, _ = bi.solve(game)

    # demotion
    gd = demotion_game(game)
    Vd, Ad = bi.solve(gd)
    print(f"demotion (D = {DEMOTE} from level 2 up):")
    print(f"{'target':>7} {'sum C/U':>9} {'hitting cost':>13}")
    for k in range(1, game.N):
        print(f"{k + 1:>7} {sum(np.array(game.C[:k]) / np.array(game.U[:k])):>9.0f} "
              f"{hitting_cost(gd, k):>13.0f}")
    th_plain = [s for _, s in os_.thresholds(game, bi.solve(game, 200_000)[1])]
    th_dem = [s for _, s in os_.thresholds(gd, bi.solve(gd, 200_000)[1])]
    print("upgrade thresholds, plain vs demotion:", list(zip(th_plain, th_dem)))

    # pity
    Vp, Ap = solve_pity(game)
    print(f"\npity (U_L(f) = min(1, U_L + {DELTA}·f)): thresholds b*_L(f) by failures f")
    th_p = {}
    for L in range(game.N - 1):
        F_L = int(np.ceil((1 - game.U[L]) / DELTA))
        th_p[L] = [os_.thresholds(game, Ap[:, :, f])[L][1] for f in range(F_L + 1)]
        print(f"  level {L + 1}: " + ", ".join(f"f={f}: {t}" for f, t in enumerate(th_p[L])))

    print(f"\n{'budget':>8} {'plain':>8} {'demotion':>9} {'pity':>8}")
    for B in [5_000, 10_000, 20_000, 50_000, 100_000]:
        i = B // du
        print(f"{B:>8} {V[i, 0]:>8.2f} {Vd[i, 0]:>9.2f} {Vp[i, 0, 0]:>8.2f}")
    for B in (5_000, 20_000):
        sd = simulate(gd, table_policy(Ad, du), B, n=2000, seed=0)
        sp = simulate_pity(game, Ap, B)
        print(f"B={B}: demotion DP {Vd[B // du, 0]:.2f} vs sim {summary(sd)}; "
              f"pity DP {Vp[B // du, 0, 0]:.2f} vs sim {summary(sp)}")

    # ---- figure -------------------------------------------------------------------------
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    Bs = np.arange(V.shape[0]) * du
    m = Bs <= 60_000
    ax.plot(Bs[m], Vp[m, 0, 0] - V[m, 0], color="#2171b5", label="pity − plain")
    ax.plot(Bs[m], Vd[m, 0] - V[m, 0], color=ACCENT, label="demotion − plain")
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xlabel("budget B (coins)")
    ax.set_ylabel("expected stars (difference from plain 03)")
    ax.set_title("(a) value under each mechanic")
    ax.legend(fontsize=8)
    for L in range(1, game.N - 1):
        ys = np.array([t if t is not None else np.nan for t in th_p[L]], float)
        ax2.plot(range(len(ys)), ys, "o-", color=LEVEL_COLORS[L + 1], label=f"level {L + 1}")
    ax2.set_yscale("log")
    ax2.set_xlabel("consecutive failures f at this level")
    ax2.set_ylabel("upgrade threshold (coins, log)")
    ax2.set_title("(b) pity: threshold vs failures so far")
    ax2.legend(fontsize=8)
    save(fig, __file__, "mechanics")


if __name__ == "__main__":
    main()
