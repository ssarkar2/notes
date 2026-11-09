"""
Problem 02 (Ir-Ud-Bk-Ec-Oe-Kk): random items, everything else as in 01.

Claim: this reduces to 01. Under the additive utility (total stars), linearity of
expectation lets each random hit be replaced by its mean r_L, so problem 01's plan and
value are optimal here too. What changes is the *spread* of the outcome.

This script checks that, on the running example:
  1. plays 01's optimal plan in the random-item game with the shared simulator and
     compares the mean to 01's value;
  2. compares the simulated standard deviation to the exact one;
  3. plots the distribution of total stars (svg).

Run:  python verify.py
"""
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common").is_dir())
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from common.loader import load  # noqa: E402
from common.model import EXAMPLE, deterministic_items, sure_upgrades  # noqa: E402
from common.simulate import simulate, summary, target_level_policy  # noqa: E402

BUDGETS = (5_000, 20_000)


def plan_stats(game, B):
    """01's optimal plan for budget B: target level k, number of hits, exact mean and sd."""
    enum = load("problems/01_Id-Ud-Bk-Ec-Oe-Kk/enumeration/solve.py")
    value, k = enum.solve(deterministic_items(game), B)
    hits = int((B - sum(game.C[:k])) // game.h)
    p, s = game.P[k], game.stars
    mean_hit = p @ s
    var_hit = p @ s**2 - mean_hit**2
    return value, k, hits, hits * mean_hit, np.sqrt(hits * var_hit)


def main():
    from common.plotting import ACCENT, plt, save

    game = sure_upgrades(EXAMPLE)                  # Ir-Ud: random items, sure upgrades
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(1, len(BUDGETS), figsize=(10, 4))

    for ax, B in zip(axes, BUDGETS):
        value01, k, hits, mean, sd = plan_stats(game, B)
        sim = simulate(game, target_level_policy(game, k), B, n=2000, seed=0)
        print(f"B={B}: 01 plan = upgrade to level {k + 1}, then {hits} hits")
        print(f"  01 value (deterministic items): {value01:.2f}")
        print(f"  02 exact mean:                  {mean:.2f}")
        print(f"  02 simulated mean:              {summary(sim)}")
        print(f"  02 sd: exact {sd:.2f}, simulated {sim.std(ddof=1):.2f}")

        # full distribution: many draws of `hits` iid items at level k (fast, vectorised)
        totals = game.stars[rng.choice(game.M, size=(100_000, hits), p=game.P[k])].sum(1)
        lo, hi = int(totals.min()), int(totals.max())
        ax.hist(totals, bins=np.arange(lo, hi + 2) - 0.5, color="#9ecae1", density=True,
                label="random items (02)")
        ax.axvline(value01, color=ACCENT, lw=2, label=f"deterministic items (01): {value01:.1f}")
        ax.set_title(f"B = {B:,}: level {k + 1}, {hits} hits\nmean {mean:.1f}, sd {sd:.1f}")
        ax.set_xlabel("total stars")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("probability")
    save(fig, __file__, "spread")


if __name__ == "__main__":
    main()
