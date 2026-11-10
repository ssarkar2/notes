"""
Problem 11 (Ir-Ur-Bk-Ec-Oe-Kb), method: Thompson sampling.

Before every decision, draw the unknown odds from the posterior and act as if the draw were
the truth: here, draw U_L ~ Beta(a_L, b_L + f) for the level you're on, and play 03's
optimal action for that U_L (other levels at their prior mean). Draws are rounded to a
0.01 grid so 03's table can be cached.

It needs no Bayes-adaptive DP, only the known-odds solver. It's built for bandits, where
trying an arm is the only way to learn about it and a bad draw costs one pull. Here a
draw that says "the odds are bad" makes you hit, which spends coins, and a later draw can
send you back to upgrading with fewer coins.

Values are simulated: true U drawn from the prior, then played. Compared with the exact
Bayes-optimal value from ../bayesian_adaptive_mdp/.

Run:  python solve.py
"""
import sys
from dataclasses import replace
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common").is_dir())
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from common.loader import load  # noqa: E402
from common.model import EXAMPLE, LONG_SHOT, Game  # noqa: E402

BAM = "problems/11_Ir-Ur-Bk-Ec-Oe-Kb/bayesian_adaptive_mdp/solve.py"


def thompson_play(game: Game, kappa: float, U_true, B: int, rng, tables: dict) -> tuple:
    """One game. Returns (stars, number of times the action switched back from hit to up)."""
    bam = load(BAM)
    bi = load(bam.BI)
    a, bb = bam.prior(game, kappa)
    du = game.unit
    b, L, f, stars, back, last = B, 0, 0, 0.0, 0, None
    while True:
        if L < game.N - 1:
            u = round(float(rng.beta(a[L], bb[L] + f)), 2)
            key = (L, u)
            if key not in tables:
                U = list(game.U)
                U[L] = u
                tables[key] = bi.solve(replace(game, U=tuple(U)), B)[1]
            act = str(tables[key][b // du, L])
        else:
            act = "hit" if b >= game.h else "stop"
        if act == "stop":
            return stars, back
        if act == "up" and last == "hit":
            back += 1
        last = act
        if act == "hit":
            b -= game.h
            stars += rng.choice(game.M, p=game.P[L]) + 1
        else:
            b -= game.C[L]
            if rng.random() < U_true[L]:
                L, f, last = L + 1, 0, None
            else:
                f += 1


def main():
    from common.plotting import ACCENT, plt, save
    from common.simulate import summary

    bam = load(BAM)
    rng = np.random.default_rng(0)
    rows = []
    for game, name, B, kappas, n in ((EXAMPLE, "running example", 20_000, (2, 5, 20), 2000),
                                     (LONG_SHOT, "LONG_SHOT", 10_000, (1, 2, 5, 20), 4000)):
        print(f"\n{name}, B = {B:,}")
        print(f"{'kappa':>6} {'Bayes-optimal (exact)':>21} {'Thompson (simulated)':>28} {'back to upgrading':>17}")
        for kappa in kappas:
            V, _ = bam.solve(game, kappa, B)
            a, bb = bam.prior(game, kappa)
            tables = {}
            res = np.array([thompson_play(game, kappa, rng.beta(a, bb), B, rng, tables) for _ in range(n)])
            rows.append((name, kappa, V[-1, 0, 0], res[:, 0].mean(), 1.96 * res[:, 0].std(ddof=1) / np.sqrt(n)))
            print(f"{kappa:>6} {V[-1, 0, 0]:>21.2f} {summary(res[:, 0]):>28} {res[:, 1].mean():>17.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, name in zip(axes, ("running example", "LONG_SHOT")):
        r = [x for x in rows if x[0] == name]
        k = np.arange(len(r))
        ax.bar(k - 0.2, [x[2] for x in r], 0.4, color=ACCENT, label="Bayes-optimal (exact)")
        ax.bar(k + 0.2, [x[3] for x in r], 0.4, color="#2171b5", yerr=[x[4] for x in r],
               capsize=3, label="Thompson sampling (simulated)")
        ax.set_xticks(k, [f"κ = {x[1]}" for x in r])
        ax.set_title(f"{name}, {'20,000' if name != 'LONG_SHOT' else '10,000'} coins")
        ax.set_ylabel("expected total stars (averaged over the prior)")
        ax.legend(fontsize=8, loc="lower right")
    save(fig, __file__, "thompson")


if __name__ == "__main__":
    main()
