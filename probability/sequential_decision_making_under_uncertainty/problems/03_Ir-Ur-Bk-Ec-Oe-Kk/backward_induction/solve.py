"""
Problem 03 (Ir-Ur-Bk-Ec-Oe-Kk), method: backward induction (finite MDP).

Upgrades now fail with probability 1 - U_L, so the Bellman equation gets an expectation
over the upgrade outcome:

    V(b, L) = max{ r_L + V(b-h, L),                                   # hit
                   U_L V(b-C_L, L+1) + (1-U_L) V(b-C_L, L),           # upgrade
                   0 }                                                # stop

Every action costs coins, so V(b, .) only depends on smaller b: fill the table for
b = 0, Δ, 2Δ, ... ("backward" in remaining budget). The result is a *policy*: an action
for every (b, L), not a fixed plan.

The demotion mechanic (fail -> drop a level w.p. D_L) is supported too: it adds a third
outcome, and the table order still works because b still only goes down.

Run:  python solve.py     (prints a table, checks against the simulator, writes svgs)
"""
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common").is_dir())
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from common.model import EXAMPLE, Game, sure_upgrades  # noqa: E402

B_MAX = 100_000


def _upgrade_value(game: Game, V_after: np.ndarray, L: int) -> float:
    """Expected value of an upgrade attempt, given V at the budget after paying C_L."""
    u, d = game.U[L], game.D[L]
    fail = (1 - d) * V_after[L] + (d * V_after[L - 1] if L > 0 else d * V_after[L])
    return u * V_after[L + 1] + (1 - u) * fail


def solve(game: Game, B_max: int = B_MAX):
    """Optimal value V[b_idx, L] and action A[b_idx, L] for budgets b = b_idx * Δ."""
    du = game.unit
    nb = B_max // du + 1
    h, C = game.h // du, [c // du for c in game.C]
    r = game.r
    V = np.zeros((nb, game.N))
    A = np.full((nb, game.N), "stop", dtype="<U4")
    for b in range(nb):
        for L in range(game.N):
            best, act = 0.0, "stop"
            if b >= h:
                v = r[L] + V[b - h, L]
                if v > best:
                    best, act = v, "hit"
            if L < game.N - 1 and b >= C[L]:
                v = _upgrade_value(game, V[b - C[L]], L)
                if v > best:
                    best, act = v, "up"
            V[b, L], A[b, L] = best, act
    return V, A


def evaluate(game: Game, policy, B_max: int = B_MAX) -> np.ndarray:
    """Exact expected value of a given policy(b_coins, L) -> 'hit'|'up'|'stop'.

    Same recursion as `solve`, but the action is dictated instead of maximised
    ("policy evaluation"). Unaffordable actions end the game (value 0).
    """
    du = game.unit
    nb = B_max // du + 1
    h, C = game.h // du, [c // du for c in game.C]
    r = game.r
    V = np.zeros((nb, game.N))
    for b in range(nb):
        for L in range(game.N):
            a = policy(b * du, L)
            if a == "hit" and b >= h:
                V[b, L] = r[L] + V[b - h, L]
            elif a == "up" and L < game.N - 1 and b >= C[L]:
                V[b, L] = _upgrade_value(game, V[b - C[L]], L)
    return V


def target_policy(game: Game, k: int):
    """Fixed-target plan: try to upgrade until level k, then hit (hit if upgrade unaffordable)."""
    def pi(b, L):
        if L < k and b >= game.C[L]:
            return "up"
        return "hit" if b >= game.h else "stop"
    return pi


def best_fixed_target(game: Game, B_max: int = B_MAX):
    """Value of the best fixed-target plan for every budget: max_k of evaluate(target k)."""
    Vk = np.stack([evaluate(game, target_policy(game, k), B_max)[:, 0] for k in range(game.N)], 1)
    return Vk.max(1), Vk.argmax(1)


def main():
    from common.plotting import ACCENT, LEVEL_COLORS, plt, save
    from common.simulate import simulate, summary, table_policy

    game = EXAMPLE                                   # Ir-Ur: random items and upgrades
    du = game.unit
    V, A = solve(game)
    V01, _ = solve(sure_upgrades(game))              # problem 01/02 value (upper bound)
    Vfix, kfix = best_fixed_target(game)

    print(f"{'budget':>8} {'01 (sure)':>10} {'03 DP':>8} {'best fixed':>10} {'(level)':>7} "
          f"{'adapt gain':>10}")
    for B in [1_000, 5_000, 10_000, 20_000, 50_000, 100_000]:
        i = B // du
        print(f"{B:>8} {V01[i, 0]:>10.2f} {V[i, 0]:>8.2f} {Vfix[i]:>10.2f} {kfix[i] + 1:>7} "
              f"{V[i, 0] - Vfix[i]:>10.2f}")

    for B in (5_000, 20_000):
        sim = simulate(game, table_policy(A, du), B, n=2000, seed=0)
        print(f"B={B}: DP {V[B // du, 0]:.2f}, simulator playing the DP policy {summary(sim)}")

    # ---- figure 1: value vs budget ---------------------------------------------------
    Bs = np.arange(V.shape[0]) * du
    m = Bs <= 40_000
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax.plot(Bs[m], V01[m, 0], color="black", ls="--", lw=1, label="01: upgrades always succeed")
    ax.plot(Bs[m], V[m, 0], color=ACCENT, lw=2, label="03: optimal policy (DP)")
    ax.plot(Bs[m], Vfix[m], color="#2171b5", lw=1.2, label="03: best fixed target plan")
    ax.set_xlabel("budget B (coins)")
    ax.set_ylabel("expected total stars")
    ax.set_title("(a) value vs budget")
    ax.legend(fontsize=8)
    ax2.plot(Bs[m], V[m, 0] - Vfix[m], color=ACCENT)
    ax2.set_xlabel("budget B (coins)")
    ax2.set_ylabel("expected stars gained")
    ax2.set_title("(b) gain from adapting: DP − best fixed plan")
    save(fig, __file__, "value_vs_budget")

    # ---- figure 2: policy map ----------------------------------------------------------
    code = {"stop": 0, "hit": 1, "up": 2}
    m = Bs <= B_MAX                                   # full range: top thresholds are high
    M = np.vectorize(code.get)(A[m]).T
    fig, ax = plt.subplots(figsize=(10, 3.2))
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#eeeeee", "#9ecae1", ACCENT])
    ax.imshow(M, aspect="auto", origin="lower", cmap=cmap, vmin=0, vmax=2,
              extent=[Bs[m][0] - du / 2, Bs[m][-1] + du / 2, 0.5, game.N + 0.5],
              interpolation="nearest")
    ax.set_xlabel("coins remaining b")
    ax.set_ylabel("core level L")
    ax.set_yticks(range(1, game.N + 1))
    ax.grid(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ACCENT, label="upgrade"), Patch(color="#9ecae1", label="hit"),
                       Patch(color="#eeeeee", label="stop (can't afford a hit)")],
              fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_title("optimal action for every (coins remaining, level)")
    save(fig, __file__, "policy_map")


if __name__ == "__main__":
    main()
