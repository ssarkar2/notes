"""
Problem 07 (Ir-Ur-Bk-Er-Oe-Kk), method: value iteration.

A j-star item refunds g_j coins when it drops (axis E: Er). In the running example only a
5-star item refunds, 250 coins, more than a hit costs (200). So a hit can leave you with
MORE coins than before, and the state (b, L) can come back to itself:

    V(b, L) = max{ r_L + sum_j P(L, j) V(b - h + g_j, L),                 # hit
                   U_L V(b - C_L, L+1) + (1 - U_L) V(b - C_L, L),         # upgrade
                   0 }                                                    # stop

V(b, .) now depends on V at LARGER b too, so there is no order in which to fill the table
once (03's backward induction). Instead, iterate: start from V = 0 and apply the right-hand
side to the whole table until it stops changing (value iteration). After k sweeps V is
the best value of the game cut off after k actions, so it rises towards the true V.

Two ways to sweep:
  - Jacobi: compute the new table from the old one, all at once (vectorised).
  - Gauss-Seidel: go through b in increasing order and overwrite in place, so most
    lookups (b - h, b - C_L, both smaller) already see this sweep's values.

The table is cut at b_cap: a refund that would take you past it is capped. Checked by
comparing caps.

Also here: the mean-refund approximation (reduction map: Er ≈ Ec with hit cost h - E[g]),
solved by backward induction since coins then only go down.

Run:  python solve.py
"""
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common").is_dir())
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from common.loader import load  # noqa: E402
from common.model import EXAMPLE, Game, with_refund  # noqa: E402

REFUND = (0, 0, 0, 0, 250)          # g_j for 1..5 stars
GAME = with_refund(EXAMPLE, REFUND)
B_CAP = 40_000
BI = "problems/03_Ir-Ur-Bk-Ec-Oe-Kk/backward_induction/solve.py"


def _tables(game: Game, nb: int):
    du = game.unit
    h, C = game.h // du, [c // du for c in game.C]
    g = np.array(game.refund) // du
    b = np.arange(nb)
    hit_to = np.minimum(b[:, None] - h + g[None, :], nb - 1)    # (nb, M), valid where b >= h
    return h, C, hit_to


def bellman(game: Game, V: np.ndarray, h, C, hit_to, policy=None):
    """One Jacobi sweep: Q-values of all actions from table V; returns (V_new, A)."""
    nb, N = V.shape
    Q = np.full((3, nb, N), -np.inf)                       # stop, hit, up
    Q[0] = 0.0
    ok = np.arange(nb) >= h
    nxt = V[hit_to[ok]]                                    # (n_ok, M, N)
    Q[1, ok] = game.r[None, :] + np.einsum("bjl,lj->bl", nxt, game.P)
    for L in range(N - 1):
        c, u, d = C[L], game.U[L], game.D[L]
        Va = V[: nb - c]
        fail = (1 - d) * Va[:, L] + d * Va[:, max(L - 1, 0)]
        Q[2, c:, L] = u * Va[:, L + 1] + (1 - u) * fail
    if policy is not None:
        return np.take_along_axis(Q, policy[None], 0)[0], policy
    A = Q.argmax(0)
    A[(A == 0) & (Q[1] >= 0)] = 1                          # ties: hit over stop
    return np.take_along_axis(Q, A[None], 0)[0], A


def value_iteration(game: Game, b_cap: int = B_CAP, tol: float = 1e-9, max_iter: int = 100_000,
                    policy=None):
    """Jacobi value iteration from V = 0. Returns V, A (0 stop, 1 hit, 2 up), sweeps, history.

    history[k] = V after k sweeps at a few budgets (for the convergence plot).
    With `policy` (an int table) it evaluates that policy instead (iterative policy evaluation).
    """
    nb = b_cap // game.unit + 1
    h, C, hit_to = _tables(game, nb)
    V = np.zeros((nb, game.N))
    hist = []
    for k in range(1, max_iter + 1):
        V_new, A = bellman(game, V, h, C, hit_to, policy)
        hist.append(V_new[:, 0].copy())
        if np.abs(V_new - V).max() < tol:
            return V_new, A, k, np.array(hist)
        V = V_new
    raise RuntimeError("value iteration did not converge")


def gauss_seidel(game: Game, b_cap: int = B_CAP, tol: float = 1e-9, max_iter: int = 10_000,
                 ascending: bool = True):
    """In-place sweeps over b (increasing, or decreasing). Returns V, A, sweeps."""
    nb = b_cap // game.unit + 1
    h, C, hit_to = _tables(game, nb)
    V = np.zeros((nb, game.N))
    A = np.zeros((nb, game.N), dtype=int)
    r, P, U, D = game.r, game.P, game.U, game.D
    for k in range(1, max_iter + 1):
        delta = 0.0
        for b in (range(nb) if ascending else range(nb - 1, -1, -1)):
            q = np.zeros((3, game.N))
            q[1] = -np.inf if b < h else r + np.einsum("jl,lj->l", V[hit_to[b]], P)
            q[2] = -np.inf
            for L in range(game.N - 1):
                if b >= C[L]:
                    Va = V[b - C[L]]
                    q[2, L] = U[L] * Va[L + 1] + (1 - U[L]) * ((1 - D[L]) * Va[L] + D[L] * Va[max(L - 1, 0)])
            a = q.argmax(0)
            a[(a == 0) & (q[1] >= 0)] = 1
            new = q[a, np.arange(game.N)]
            delta = max(delta, np.abs(new - V[b]).max())
            V[b], A[b] = new, a
        if delta < tol:
            return V, A, k
    raise RuntimeError("Gauss-Seidel did not converge")


def mean_refund_value(game: Game, B_max: int):
    """Approximation: no refunds, but a hit at level L costs h - E[g | L] (deterministic).

    Coins only go down again, so backward induction works. Grid step 2.5 coins here.
    Returns (budgets, V[:, 0]).
    """
    net = game.h - game.mean_refund                          # per level
    vals = np.concatenate([net, game.C])
    step = 0.5
    while not np.allclose(vals / step, np.round(vals / step)):
        step /= 2
    unit = step
    while np.allclose(vals / (unit * 2), np.round(vals / (unit * 2))):
        unit *= 2
    hk = np.round(net / unit).astype(int)
    Ck = [int(round(c / unit)) for c in game.C]
    nb = int(B_max / unit) + 1
    V = np.zeros((nb, game.N))
    for b in range(nb):
        for L in range(game.N):
            best = 0.0
            if b >= hk[L]:
                best = max(best, game.r[L] + V[b - hk[L], L])
            if L < game.N - 1 and b >= Ck[L]:
                Va = V[b - Ck[L]]
                best = max(best, game.U[L] * Va[L + 1] + (1 - game.U[L]) * Va[L])
            V[b, L] = best
    return np.arange(nb) * unit, V[:, 0]


def thresholds(A: np.ndarray, unit: int, b_max: int) -> list:
    """Per level: smallest b from which the policy upgrades at every b' >= b (up to b_max)."""
    out = []
    for L in range(A.shape[1] - 1):
        up = A[: b_max // unit + 1, L] == 2
        if not up.any():
            out.append(None)
            continue
        not_up = np.flatnonzero(~up)
        out.append((not_up[-1] + 1) * unit if len(not_up) else 0)
    return out


def table_policy_capped(A: np.ndarray, unit: int):
    names = np.array(["stop", "hit", "up"])

    def pi(o):
        return str(names[A[min(o.b // unit, len(A) - 1), o.L]])
    return pi


def main():
    import time

    from common.plotting import ACCENT, plt, save
    from common.simulate import simulate, summary

    game, du = GAME, GAME.unit
    print(f"refunds g_j = {REFUND}; grid Δ = {du}; mean refund per hit by level = {game.mean_refund}")
    print(f"net cost of a hit by level (h - E[g]) = {game.h - game.mean_refund}")

    # ---- sanity: with no refunds, value iteration = 03's backward induction --------------
    V03, A03 = load(BI).solve(EXAMPLE, 20_000)
    V0, _, k0, _ = value_iteration(EXAMPLE, 20_000)
    print(f"no refunds: value iteration ({k0} sweeps) vs 03 backward induction, max diff "
          f"{np.abs(V0 - V03).max():.2e}")

    # ---- the refund game -------------------------------------------------------------------
    t = time.time(); V, A, kj, hist = value_iteration(game); tj = time.time() - t
    print(f"Jacobi: {kj} sweeps ({tj:.1f}s)")
    for asc in (True, False):
        Vg, _, kg = gauss_seidel(game, ascending=asc)
        print(f"Gauss-Seidel, b {'increasing' if asc else 'decreasing'}: {kg} sweeps, "
              f"max diff from Jacobi {np.abs(V - Vg).max():.1e}")
    for B in (5_000, 10_000, 20_000):
        k = int(np.argmax(np.abs(hist[:, B // du] - V[B // du, 0]) < 0.01)) + 1
        print(f"  Jacobi sweeps until V({B:,}) is within 0.01 of its limit: {k}")
    for cap in (30_000, 60_000):
        Vc = value_iteration(game, cap)[0]
        print(f"cap {cap:,}: V(5k) {Vc[5_000 // du, 0]:.4f}, V(20k) {Vc[20_000 // du, 0]:.4f}  "
              f"(cap {B_CAP:,}: {V[5_000 // du, 0]:.4f}, {V[20_000 // du, 0]:.4f})")

    Bm, Vm = mean_refund_value(game, 20_000)
    print(f"\n{'budget':>8} {'03 (no refund)':>14} {'07 exact':>9} {'mean-refund approx':>18}")
    for B in (1_000, 2_000, 5_000, 10_000, 20_000):
        i = int(np.argmin(np.abs(Bm - B)))
        print(f"{B:>8,} {V03[B // 100, 0]:>14.2f} {V[B // du, 0]:>9.2f} {Vm[i]:>18.2f}")

    th = thresholds(A, du, 30_000)
    th03 = thresholds(np.where(A03 == "up", 2, np.where(A03 == "hit", 1, 0)), 100, 20_000)
    print("upgrade thresholds (coins), 07:", th, "\n                            03:", th03)
    # is it a threshold rule? upgrade region per level contiguous up to 30k
    for L in range(game.N - 1):
        up = np.flatnonzero(A[: 30_000 // du + 1, L] == 2)
        if len(up):
            print(f"  level {L + 1}: up at {len(up)} grid points from {up[0] * du:,} to {up[-1] * du:,}, "
                  f"contiguous {up[-1] - up[0] + 1 == len(up)}")

    sims = {}
    for B in (5_000, 20_000):
        sims[B] = sim = simulate(game, table_policy_capped(A, du), B, n=4000, seed=0)
        print(f"B={B:,}: value iteration {V[B // du, 0]:.2f}, simulator {summary(sim)}, "
              f"sd {sim.std(ddof=1):.1f}, 10/50/90% {np.percentile(sim, [10, 50, 90])}")

    from common.simulate import table_policy
    sim03 = simulate(EXAMPLE, table_policy(A03, 100), 20_000, n=4000, seed=0)
    print(f"03 (no refunds) at 20,000, simulated: sd {sim03.std(ddof=1):.1f}, "
          f"10/50/90% {np.percentile(sim03, [10, 50, 90])}")

    # ---- figure: convergence, value vs budget, outcome spread -------------------------------
    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
    k = np.arange(1, len(hist) + 1)
    for B, col in [(5_000, "#2171b5"), (20_000, ACCENT)]:
        ax.plot(k, hist[:, B // du], color=col, lw=1.5, label=f"B = {B:,}")
        ax.axhline(V[B // du, 0], color=col, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("sweeps k")
    ax.set_ylabel("V after k sweeps (expected stars)")
    ax.set_title("(a) value iteration: best value with at most k actions")
    ax.legend(fontsize=8, loc="upper left")
    Bs = np.arange(len(V)) * du
    m = Bs <= 20_000
    ax2.plot(np.arange(len(V03)) * 100, V03[:, 0], color="black", lw=1.2, label="03: no refunds")
    ax2.plot(Bs[m], V[m, 0], color=ACCENT, lw=2, label="07: 5★ refunds 250 (exact)")
    ax2.plot(Bm, Vm, color="#2171b5", lw=1, ls="--", label="07: mean-refund approximation")
    ax2.set_xlabel("budget B (coins)")
    ax2.set_ylabel("expected total stars")
    ax2.set_title("(b) value vs budget")
    ax2.set_xticks(range(0, 20_001, 5_000))
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax2.legend(fontsize=8, loc="upper left")
    bins = np.arange(0, 2_001, 25)
    ax3.hist(sim03, bins=bins, color="black", alpha=0.5, label="03: no refunds")
    ax3.hist(np.minimum(sims[20_000], 2_000), bins=bins, color=ACCENT, alpha=0.6,
             label="07: refunds (≥ 2,000 shown at 2,000)")
    ax3.set_xlabel("total stars")
    ax3.set_ylabel(f"games (of {len(sim03):,})")
    ax3.set_title("(c) outcomes at B = 20,000 (simulated)")
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax3.legend(fontsize=8, loc="upper right")
    save(fig, __file__, "value_iteration")


if __name__ == "__main__":
    main()
