"""
Problem 06 (Ir-Ur-Bh-Ec-Oe-Kk), method: belief MDP.

The budget B is drawn from a known pmf p(B) but never shown. The player sees only what it
has spent, s, and the core level L. Trying an action that costs more than what's left
(c > B - s) ends the game; the leftover coins are lost.

Belief. The only thing the player ever learns about B is "not broke yet", i.e. B >= s.
So the belief is the prior truncated at s:

    P(B = x | B >= s) = p(x) / S(s),      S(x) = P(B >= x)   (survival function)

The belief is a function of s alone, so this POMDP is an MDP on (s, L). An action of cost
c goes through with probability S(s + c) / S(s); otherwise the game ends:

    V(s, L) = max{ S(s+h)/S(s)   * (r_L + V(s+h, L)),                          # hit
                   S(s+C_L)/S(s) * (U_L V(s+C_L, L+1) + (1-U_L) V(s+C_L, L)),  # upgrade
                   0 }                                                         # stop

s only goes up, so fill the table from the largest s down ("backward" in coins spent).

Run:  python solve.py     (prints the tables, checks against the simulator, writes svgs)
"""
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common").is_dir())
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from common.loader import load  # noqa: E402
from common.model import EXAMPLE, Game  # noqa: E402

BI = "problems/03_Ir-Ur-Bk-Ec-Oe-Kk/backward_induction/solve.py"
P05 = "problems/05_Ir-Ur-Bs-Ec-Oe-Kk/verify.py"


def survival(pmf: np.ndarray, pad: int) -> np.ndarray:
    """S[x] = P(B >= x) on the grid, padded with zeros so S[s + c] is always defined."""
    S = np.cumsum(pmf[::-1])[::-1]
    return np.concatenate([S, np.zeros(pad)])


def solve(game: Game, pmf: np.ndarray, policy=None):
    """Optimal V[s_idx, L], A[s_idx, L] for coins spent s = s_idx * Δ.

    With `policy` (an action table of the same shape, entries 'hit'/'up'/'stop') it
    evaluates that policy instead of maximising. Unreachable states (S(s) = 0) get V = 0.
    """
    du = game.unit
    h, C = game.h // du, [c // du for c in game.C]
    pad = max(h, *C) + 1
    S = survival(pmf, pad)
    ns = len(pmf)
    r, U, D = game.r, game.U, game.D
    V = np.zeros((ns + pad, game.N))
    A = np.full((ns, game.N), "stop", dtype="<U4")
    for s in range(ns - 1, -1, -1):
        if S[s] <= 0:
            continue
        for L in range(game.N):
            vals = {"stop": 0.0, "hit": S[s + h] / S[s] * (r[L] + V[s + h, L])}
            if L < game.N - 1:
                c = C[L]
                Va = V[s + c]
                fail = (1 - D[L]) * Va[L] + D[L] * Va[max(L - 1, 0)]
                vals["up"] = S[s + c] / S[s] * (U[L] * Va[L + 1] + (1 - U[L]) * fail)
            if policy is not None:
                a = str(policy[s, L])
            else:                                  # ties: hit, then stop; up must win
                a = "hit" if vals["hit"] >= vals["stop"] else "stop"
                if vals.get("up", -1) > vals[a] * (1 + 1e-12):
                    a = "up"
            V[s, L], A[s, L] = vals[a], a
    return V[:ns], A


def hidden_policy(A: np.ndarray, unit: int):
    """Plays an (s, L) action table in the simulator (which passes b = None)."""
    def pi(o):
        return str(A[o.s // unit, o.L])
    return pi


def upgrade_region(A: np.ndarray, L: int, unit: int) -> str:
    """Where (in coins spent) the policy upgrades from level L, as compact intervals."""
    up = np.flatnonzero(A[:, L] == "up")
    if len(up) == 0:
        return "never"
    runs, start = [], up[0]
    for a, b in zip(up[:-1], up[1:]):
        if b != a + 1:
            runs.append((start, a)); start = b
    runs.append((start, up[-1]))
    return ", ".join(f"{a * unit:,}–{b * unit:,}" for a, b in runs)


def main():
    from common.simulate import simulate, summary

    game = EXAMPLE
    du = game.unit
    p05 = load(P05)
    bi = load(BI)
    V03, A03 = bi.solve(game, p05.B_MAX)
    dists = p05.budget_distributions(du)
    Bs = np.arange(len(V03)) * du

    print(f"{'prior':>26} {'seen (05)':>9} {'hidden':>8} {'EVPI':>6}   simulator (hidden)")
    tables = {}
    for name, p in dists.items():
        V, A = solve(game, p)
        tables[name] = A
        seen = (p * V03[:, 0]).sum()
        idx = np.flatnonzero(p)
        draw = lambda rng, idx=idx, p=p: int(rng.choice(idx, p=p[idx] / p[idx].sum())) * du  # noqa: E731
        sim = simulate(game, hidden_policy(A, du), draw, n=4000, seed=0, hidden_budget=True)
        print(f"{name:>26} {seen:>9.2f} {V[0, 0]:>8.2f} {seen - V[0, 0]:>6.2f}   {summary(sim)}")
        for L in range(game.N - 1):
            print(f"      up from level {L + 1}: spent {upgrade_region(A[: 60_000 // du], L, du)}")

    # memoryless prior: the action should not depend on s (up to the cut at B_MAX)
    A = tables["exponential, mean 10,000"]
    first = [np.flatnonzero(A[:, L] != A[0, L]) for L in range(game.N)]
    print("exponential prior: first s where the action differs from s = 0, per level:",
          [f"{f[0] * du:,}" if len(f) else "none" for f in first])

    # ---- figure: policy maps over (coins spent, level) ---------------------------------
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    from common.plotting import ACCENT, plt, save
    code = {"stop": 0, "hit": 1, "up": 2}
    s_max = 20_000
    fig, axes = plt.subplots(len(tables), 1, figsize=(9, 7.5), sharex=True)
    cmap = ListedColormap(["#eeeeee", "#9ecae1", ACCENT])
    for ax, (name, A) in zip(axes, tables.items()):
        M = np.vectorize(code.get)(A[: s_max // du + 1]).T
        ax.imshow(M, aspect="auto", origin="lower", cmap=cmap, vmin=0, vmax=2,
                  extent=[-du / 2, s_max + du / 2, 0.5, game.N + 0.5], interpolation="nearest")
        ax.set_yticks(range(1, game.N + 1))
        ax.set_ylabel("level L")
        ax.grid(False)
        ax.set_title(f"prior: {name}", fontsize=9, loc="left")
    axes[-1].set_xlabel("coins spent s")
    axes[-1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    axes[0].legend(handles=[Patch(color=ACCENT, label="upgrade"), Patch(color="#9ecae1", label="hit"),
                            Patch(color="#eeeeee", label="stop / unreachable")],
                   fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save(fig, __file__, "policy_maps")


if __name__ == "__main__":
    main()
