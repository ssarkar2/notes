"""
Problem 04 (Ir-Ur-Bk-Ec-Ou-Kk), method: state augmentation.

The utility u is no longer a plain sum of stars, so the stars can't be paid out hit by
hit. Instead, carry whatever summary s of the items u depends on, and pay u only at the
end:

    V(b, L, s) = max{ sum_j P(L,j) V(b-h, L, next(s, j)),                     # hit
                      U_L V(b-C_L, L+1, s) + (1-U_L) V(b-C_L, L, s),           # upgrade
                      u(s) }                                                   # stop

Summaries used here:
    total stars W           u = f(W) for any f (linear, sqrt, square, goal 1[W >= G])
    "got a 5* yet?" flag    u = 1[any item is 5*]

Same backward induction as problem 03, over a bigger state. b still only goes down.

Run:  python solve.py     (prints tables, checks against the simulator, writes svgs)
"""
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common").is_dir())
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from common.model import EXAMPLE, Game  # noqa: E402
from common.simulate import Obs  # noqa: E402

B_MAX = 20_000
STOP, HIT, UP = 0, 1, 2
ACTIONS = np.array(["stop", "hit", "up"])


@dataclass
class Utility:
    """u expressed through a summary s of the items: s in 0..S-1, updated by next[s, j]."""
    name: str
    nxt: np.ndarray                     # (S, M): summary after receiving a (j+1)-star item
    u: np.ndarray                       # (S,): utility when the game ends in summary s
    of_obs: Callable[[Obs], int]        # summary index for a simulator observation
    of_items: Callable[[list], float]   # u computed straight from the item list (simulator)
    W_based: bool = True                # True if s = min(W, cap): the summary is total stars

    @property
    def S(self) -> int:
        return len(self.u)


def max_stars(game: Game, B_max: int) -> int:
    return int(game.stars[-1]) * (B_max // game.h)


def total_stars_utility(game: Game, f, name: str, B_max: int = B_MAX) -> Utility:
    """u = f(total stars). Summary = W, exact (no cap reached within B_max)."""
    W = max_stars(game, B_max)
    s = np.arange(W + 1)
    nxt = np.minimum(s[:, None] + game.stars[None].astype(int), W)
    return Utility(name, nxt, f(s.astype(float)), lambda o: int(o.W), lambda it: f(float(sum(it))))


def goal_utility(game: Game, G: int) -> Utility:
    """u = 1[total stars >= G]. Summary = min(W, G): stars beyond the goal don't matter."""
    s = np.arange(G + 1)
    nxt = np.minimum(s[:, None] + game.stars[None].astype(int), G)
    return Utility(f"P(total ≥ {G})", nxt, (s >= G).astype(float),
                   lambda o: min(int(o.W), G), lambda it: float(sum(it) >= G))


def any_top_utility(game: Game, star: int = 5) -> Utility:
    """u = 1[some item has `star` stars]. Summary = a yes/no flag: 2 states."""
    nxt = np.array([[int(j >= star) for j in game.stars], [1] * game.M])
    return Utility(f"P(at least one {star}★)", nxt, np.array([0.0, 1.0]),
                   lambda o: int(o.top >= star), lambda it: float(max(it, default=0) >= star),
                   W_based=False)


def solve(game: Game, ut: Utility, B_max: int = B_MAX):
    """V[b_idx, L, s] and A[b_idx, L, s] (0 stop, 1 hit, 2 up) for b = b_idx * Δ."""
    du = game.unit
    nb = B_max // du + 1
    h, C = game.h // du, [c // du for c in game.C]
    N, S = game.N, ut.S
    P = game.P
    V = np.empty((nb, N, S))
    A = np.zeros((nb, N, S), dtype=np.int8)
    eps = 1e-12
    for b in range(nb):
        best = np.broadcast_to(ut.u, (N, S)).copy()
        act = np.zeros((N, S), dtype=np.int8)
        if b >= h:
            hit = np.einsum("lj,lsj->ls", P, V[b - h][:, ut.nxt])
            m = hit >= best - eps          # ties with stopping go to hitting (coins are worth 0)
            best[m], act[m] = hit[m], HIT
        for L in range(N - 1):
            if b < C[L]:
                continue
            Va = V[b - C[L]]
            d = game.D[L]
            fail = (1 - d) * Va[L] + d * Va[max(L - 1, 0)]
            up = game.U[L] * Va[L + 1] + (1 - game.U[L]) * fail
            m = up > best[L] + eps
            best[L][m], act[L][m] = up[m], UP
        V[b], A[b] = best, act
    return V, A


def final_distribution(game: Game, ut: Utility, A: np.ndarray, B: int):
    """Exact distribution of the game's end, playing action table A (made for utility ut).

    Pushes probability mass forward from (B, level 1, no items). Tracks total stars W
    (and the policy's own summary s when it isn't W). Returns pmf[L, W, s2]: probability
    of ending at level L with W total stars (s2 = the policy summary; 1 entry if W-based),
    and the expected number of upgrade attempts made after the first hit.
    """
    du = game.unit
    nb = B // du + 1
    h, C = game.h // du, [c // du for c in game.C]
    N, Wm = game.N, max_stars(game, B)
    S2 = 1 if ut.W_based else ut.S
    stars = game.stars.astype(int)
    mass = np.zeros((nb, N, Wm + 1, S2))
    out = np.zeros((N, Wm + 1, S2))
    mass[nb - 1, 0, 0, 0] = 1.0
    W = np.arange(Wm + 1)
    late_up = 0.0
    for b in range(nb - 1, -1, -1):
        for L in range(N):
            m = mass[b, L]
            if not m.any():
                continue
            for s2 in range(S2):
                s_idx = np.minimum(W, ut.S - 1) if ut.W_based else np.full(Wm + 1, s2)
                a = A[b, L, s_idx]
                col = m[:, s2]
                stop = col * (a == STOP)
                out[L, :, s2] += stop
                hit = col * (a == HIT)
                if hit.any():
                    for j, st in enumerate(stars):
                        p = game.P[L, j]
                        if p == 0:
                            continue
                        t2 = s2 if ut.W_based else ut.nxt[s2, j]
                        mass[b - h, L, st:, t2] += p * hit[:Wm + 1 - st]
                up = col * (a == UP)
                late_up += up[1:].sum()             # W > 0: at least one hit already made
                if up.any():
                    mass[b - C[L], L + 1, :, s2] += game.U[L] * up
                    mass[b - C[L], L, :, s2] += (1 - game.U[L]) * up
    return out, late_up


def stats(game: Game, ut: Utility, A: np.ndarray, B: int, utilities) -> dict:
    """Mean/sd of total stars, mean final level and E[u] for each u in `utilities`."""
    pmf, late_up = final_distribution(game, ut, A, B)
    pW = pmf.sum((0, 2))
    W = np.arange(len(pW))
    mean = (W * pW).sum()
    out = {"mean": mean, "sd": np.sqrt(((W - mean) ** 2 * pW).sum()),
           "level": ((np.arange(game.N) + 1) * pmf.sum((1, 2))).sum(), "pW": pW,
           "late_up": late_up}
    for name, f in utilities.items():
        out[name] = (f(W.astype(float)) * pW).sum()
    if not ut.W_based:
        out["flag"] = pmf[:, :, 1].sum()
    return out


def table_policy(A: np.ndarray, unit: int, ut: Utility):
    """Simulator policy reading A[b, L, s] with s from the observation."""
    def pi(o: Obs) -> str:
        return str(ACTIONS[A[o.b // unit, o.L, ut.of_obs(o)]])
    return pi


def main():
    from common.plotting import ACCENT, LEVEL_COLORS, plt, save
    from common.simulate import simulate, summary

    game = EXAMPLE
    du = game.unit
    fs = {"total stars": lambda W: W,
          "√(total stars)": np.sqrt,
          "(total stars)²": lambda W: W ** 2}
    uts = {k: total_stars_utility(game, f, k) for k, f in fs.items()}
    sols = {k: solve(game, ut) for k, ut in uts.items()}

    # ---- sanity: linear u reproduces problem 03 ------------------------------------
    V_lin = sols["total stars"][0]
    print(f"linear u: V(5k) = {V_lin[5_000 // du, 0, 0]:.2f}, V(20k) = {V_lin[20_000 // du, 0, 0]:.2f}"
          f"  (problem 03: 40.64, 250.35)")

    # ---- does the policy depend on stars already collected? ------------------------
    # only states that can occur: W stars so far, b coins left, W <= 5 * (B_MAX - b) / h
    nb = B_MAX // du + 1
    b_c = np.arange(nb)[:, None] * du
    reach = np.arange(uts["total stars"].S)[None] <= int(game.stars[-1]) * (B_MAX - b_c) // game.h
    for k, (V, A) in sols.items():
        diff = (A != A[:, :, :1]) & reach[:, None, :]
        varies = diff.any(2)
        print(f"{k:>16}: (b, L) cells where the action depends on W: {varies.sum()} of {varies.size}")

    # ---- each optimal policy, scored under every utility -----------------------------
    rows = {}
    for B in (5_000, 20_000):
        print(f"\nB = {B:,}: each row = policy optimal for that u; columns = outcome of playing it")
        print(f"{'optimal for':>16} {'E[stars]':>9} {'sd':>6} {'final lvl':>9} "
              f"{'E[√W]':>8} {'E[W²]':>10} {'late ups':>8}")
        for k, ut in uts.items():
            st = stats(game, ut, sols[k][1], B, fs)
            rows[(k, B)] = st
            print(f"{k:>16} {st['mean']:>9.2f} {st['sd']:>6.2f} {st['level']:>9.2f} "
                  f"{st['√(total stars)']:>8.3f} {st['(total stars)²']:>10.0f} {st['late_up']:>8.3f}")
            assert abs(st[k] - sols[k][0][B // du, 0, 0]) < 1e-6 * max(1, abs(st[k]))

    # ---- simulator check -----------------------------------------------------------------
    print()
    for k in ("√(total stars)", "(total stars)²"):
        ut, (V, A) = uts[k], sols[k]
        for B in (5_000, 20_000):
            sim = simulate(game, table_policy(A, du, ut), B, n=2000, seed=0, utility=ut.of_items)
            print(f"{k:>16}, B={B:,}: DP E[u] = {V[B // du, 0, 0]:.3f}, simulator {summary(sim)}")

    # ---- goals -----------------------------------------------------------------------------
    print("\nGoals at B = 20,000: P(reach goal) under the goal policy vs the 03 policy")
    A_lin = sols["total stars"][1]
    st_lin = rows[("total stars", 20_000)]
    goals = list(range(150, 351, 10))
    p_goal, p_lin, lvl_goal = [], [], []
    for G in goals:
        ut = goal_utility(game, G)
        V, A = solve(game, ut)
        st = stats(game, ut, A, 20_000, {})
        p_goal.append(V[20_000 // du, 0, 0])
        p_lin.append(st_lin["pW"][G:].sum())
        lvl_goal.append(st["level"])
        if G in (200, 250, 300):
            sim = simulate(game, table_policy(A, du, ut), 20_000, n=2000, seed=0,
                           utility=ut.of_items)
            print(f"  G={G}: goal policy {p_goal[-1]:.4f} (simulator {summary(sim)}), "
                  f"03 policy {p_lin[-1]:.4f}, mean final level {st['level']:.2f} "
                  f"vs {st_lin['level']:.2f}, E[stars] {st['mean']:.1f}, sd {st['sd']:.1f}, "
                  f"late ups {st['late_up']:.3f}")

    print("\nAt least one 5★: goal policy vs the 03 policy")
    ut5 = any_top_utility(game)
    V5, A5 = solve(game, ut5)
    budgets5 = np.arange(0, B_MAX + 1, 500)
    p5_goal = V5[budgets5 // du, 0, 0]
    p5_lin = []
    for B in budgets5:
        st = stats(game, ut5, A_lin[..., :1].repeat(2, 2), int(B), {})
        p5_lin.append(st["flag"])
    for B in (2_000, 5_000, 10_000, 20_000):
        i = B // 500
        st = stats(game, ut5, A5, B, {})
        sim = simulate(game, table_policy(A5, du, ut5), B, n=2000, seed=0, utility=ut5.of_items)
        print(f"  B={B:>6,}: goal policy {p5_goal[i]:.3f} (simulator {summary(sim)}), "
              f"03 policy {p5_lin[i]:.3f}; E[stars] {st['mean']:.1f} vs "
              f"{V_lin[B // du, 0, 0]:.1f}, final level {st['level']:.2f}, "
              f"late ups {st['late_up']:.3f}")

    # ---- figure 1: outcome spread at B = 20,000 --------------------------------------------
    colors = {"total stars": "black", "√(total stars)": "#2171b5", "(total stars)²": ACCENT}
    fig, ax = plt.subplots(figsize=(7, 4))
    for k in uts:
        pW = rows[(k, 20_000)]["pW"]
        W = np.arange(len(pW))
        ax.step(W, np.cumsum(pW), where="post", color=colors[k], lw=1.6,
                label=f"optimal for u = {k}: mean {rows[(k, 20_000)]['mean']:.0f}, "
                      f"sd {rows[(k, 20_000)]['sd']:.0f}")
    ax.set_xlim(100, 400)
    ax.set_xlabel("total stars at the end")
    ax.set_ylabel("P(total stars ≤ x)")
    ax.set_title("outcome distribution at B = 20,000 under each u's optimal policy")
    ax.legend(fontsize=8, loc="upper left")
    save(fig, __file__, "outcome_spread")

    # ---- figure 2: goals --------------------------------------------------------------------
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax.plot(goals, p_goal, color=ACCENT, lw=2, label="policy optimal for the goal")
    ax.plot(goals, p_lin, color="black", ls="--", lw=1, label="03 policy (max expected stars)")
    ax.axvline(st_lin["mean"], color="#999999", lw=0.8)
    ax.text(st_lin["mean"] + 3, 0.05, "03 mean", fontsize=8, color="#666666")
    ax.set_xlabel("goal G (total stars)")
    ax.set_ylabel("P(total stars ≥ G)")
    ax.set_title("(a) goal: total ≥ G, at B = 20,000")
    ax.legend(fontsize=8)
    ax2.plot(budgets5, p5_goal, color=ACCENT, lw=2, label="policy optimal for the goal")
    ax2.plot(budgets5, p5_lin, color="black", ls="--", lw=1, label="03 policy (max expected stars)")
    ax2.set_xlabel("budget B (coins)")
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax2.set_ylabel("P(at least one 5★)")
    ax2.set_title("(b) goal: at least one 5★")
    ax2.legend(fontsize=8)
    save(fig, __file__, "goals")


if __name__ == "__main__":
    main()
