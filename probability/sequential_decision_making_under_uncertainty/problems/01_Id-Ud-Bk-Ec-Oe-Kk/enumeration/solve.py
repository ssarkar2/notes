"""
Problem 01 (Id-Ud-Bk-Ec-Oe-Kk), method: enumeration over the target level.

Everything is deterministic and r_L rises with L, so the optimal plan is
"upgrade straight to some level k, then spend the rest on hits". Try every k:

    value(B) = max_k  r_k * floor( (B - K_k) / h ),     K_k = sum_{i<k} C_i

Run:  python solve.py     (prints a table, writes value_vs_budget.svg)
"""
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common").is_dir())
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from common.model import EXAMPLE, Game, deterministic_items, sure_upgrades  # noqa: E402


def per_target(game: Game, B: int) -> np.ndarray:
    """Value of the plan "go to level k, then hit" for every k (-inf if unaffordable)."""
    K = np.concatenate([[0], np.cumsum(game.C)])          # cumulative upgrade cost
    hits = np.floor((B - K) / game.h)
    return np.where(B >= K, hits * game.r, -np.inf)


def solve(game: Game, B: int) -> tuple[float, int]:
    """Return (optimal value, optimal target level k, 0-indexed)."""
    v = per_target(game, B)
    k = int(np.argmax(v))
    return float(max(v[k], 0.0)), k


def switch_budgets(game: Game) -> np.ndarray:
    """Budgets where target k+1 overtakes target k, ignoring the floor (continuous hits):

        (B - K_k) r_k = (B - K_{k+1}) r_{k+1}   =>   B*_k = (K_{k+1} r_{k+1} - K_k r_k) / (r_{k+1} - r_k)
    """
    K = np.concatenate([[0], np.cumsum(game.C)])
    r = game.r
    return (K[1:] * r[1:] - K[:-1] * r[:-1]) / (r[1:] - r[:-1])


def main():
    from common.plotting import ACCENT, LEVEL_COLORS, plt, save
    from common.simulate import simulate, summary, target_level_policy

    game = sure_upgrades(deterministic_items(EXAMPLE))
    print(f"tag {game.tag_IU}  r_L = {np.round(game.r, 2)}")
    print(f"{'budget':>8} {'value':>8} {'best level':>10}")
    for B in [1_000, 5_000, 10_000, 20_000, 50_000, 100_000]:
        v, k = solve(game, B)
        print(f"{B:>8} {v:>8.2f} {k + 1:>10}")

    # sanity check against the simulator
    B = 20_000
    v, k = solve(game, B)
    sim = simulate(game, target_level_policy(game, k), B, n=50)
    print(f"\nB={B}: enumeration {v:.2f}, simulator {summary(sim)}")

    # plot: one line per target level; the optimum is the upper envelope
    Bs = np.arange(0, 100_001, game.unit)
    V = np.array([per_target(game, B) for B in Bs])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for k in range(game.N):
        ax.plot(Bs, np.where(np.isfinite(V[:, k]), V[:, k], np.nan),
                color=LEVEL_COLORS[k], lw=1, label=f"target level {k + 1}")
    ax.plot(Bs, V.max(1), color="black", lw=2, ls="--", label="optimum (envelope)")
    ax.set_xlabel("budget B (coins)")
    ax.set_ylabel("total item value")
    ax.set_title("(a) value of each 'upgrade to k, then hit' plan")
    ax.legend(fontsize=8)

    ax2.step(Bs, V.argmax(1) + 1, where="post", color="black", lw=1, label="exact (integer hits)")
    for i, Bstar in enumerate(switch_budgets(game)):
        ax2.axvline(Bstar, color=ACCENT, ls=":", lw=1,
                    label="switch point, continuous hits" if i == 0 else None)
    ax2.legend(fontsize=8, loc="lower right")
    ax2.set_xlabel("budget B (coins)")
    ax2.set_ylabel("optimal target level k*")
    ax2.set_yticks(range(1, game.N + 1))
    ax2.set_title("(b) richer → invest more before harvesting")
    save(fig, __file__, "value_vs_budget")


if __name__ == "__main__":
    main()
