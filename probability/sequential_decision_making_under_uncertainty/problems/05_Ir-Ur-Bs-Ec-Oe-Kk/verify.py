"""
Problem 05 (Ir-Ur-Bs-Ec-Oe-Kk): random budget, seen before play.

Reduction to 03: once you see B you play 03's optimal policy for B, so

    value = E[ V(B) ] = sum_B  p(B) V(B).

No new solver: 03's backward induction table already holds V(b, L) for every b, and the
policy only looks at (b, L), so the same table plays every budget.

This script:
  - averages V over a few budget distributions with the same mean (10,000);
  - compares E[V(B)] with V(E[B])  (Jensen: V is not linear in B);
  - splits the outcome's spread into "which budget you got" and "luck during play"
    (law of total variance), with an exact second-moment recursion;
  - checks everything against the simulator, which draws B, then plays.

Run:  python verify.py
"""
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common").is_dir())
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from common.loader import load  # noqa: E402
from common.model import EXAMPLE, Game  # noqa: E402

B_MAX = 150_000
MEAN = 10_000
BI = "problems/03_Ir-Ur-Bk-Ec-Oe-Kk/backward_induction/solve.py"


def budget_distributions(du: int, B_max: int = B_MAX) -> dict:
    """pmf over the grid b = 0, Δ, 2Δ, ..., all with mean MEAN."""
    nb = B_max // du + 1
    k = np.arange(nb)
    out = {}
    p = np.zeros(nb); p[MEAN // du] = 1
    out["fixed 10,000 (= 03)"] = p
    p = np.zeros(nb); p[2_000 // du] = p[18_000 // du] = 0.5
    out["2,000 or 18,000 (½ each)"] = p
    p = np.zeros(nb); p[: 2 * MEAN // du + 1] = 1
    out["uniform 0 to 20,000"] = p / p.sum()
    q = (MEAN / du) / (MEAN / du + 1)          # geometric on the grid, mean MEAN
    p = (1 - q) * q ** k
    out["exponential, mean 10,000"] = p / p.sum()
    return out


def second_moment(game: Game, A: np.ndarray, V: np.ndarray) -> np.ndarray:
    """E[W^2] for every (b, L) when playing action table A (V = its mean, same table)."""
    du = game.unit
    h, C = game.h // du, [c // du for c in game.C]
    j = game.stars
    M2 = np.zeros_like(V)
    for b in range(V.shape[0]):
        for L in range(game.N):
            a = A[b, L]
            if a == "hit":
                p = game.P[L]
                M2[b, L] = p @ (j ** 2) + 2 * (p @ j) * V[b - h, L] + M2[b - h, L]
            elif a == "up":
                u, d = game.U[L], game.D[L]
                Ma = M2[b - C[L]]
                fail = (1 - d) * Ma[L] + d * Ma[max(L - 1, 0)]
                M2[b, L] = u * Ma[L + 1] + (1 - u) * fail
    return M2


def main():
    from common.plotting import ACCENT, plt, save
    from common.simulate import simulate, summary, table_policy

    bi = load(BI)
    game = EXAMPLE
    du = game.unit
    V, A = bi.solve(game, B_MAX)
    M2 = second_moment(game, A, V)
    v, m2 = V[:, 0], M2[:, 0]
    var_given_B = m2 - v ** 2
    Bs = np.arange(len(v)) * du

    dists = budget_distributions(du)
    print(f"V(E[B]) = V(10,000) = {v[MEAN // du]:.2f}\n")
    print(f"{'budget distribution':>26} {'E[B]':>7} {'E[V(B)]':>8} {'sd total':>9} "
          f"{'sd from B':>9} {'sd from luck':>12}   simulator")
    rows = {}
    for name, p in dists.items():
        EB = (p * Bs).sum()
        EV = (p * v).sum()
        between = np.sqrt((p * (v - EV) ** 2).sum())          # sd of V(B): which budget you got
        within = np.sqrt((p * var_given_B).sum())             # typical luck during play
        total = np.sqrt(between ** 2 + within ** 2)
        idx = np.flatnonzero(p)
        draw = lambda rng, idx=idx, p=p: int(rng.choice(idx, p=p[idx] / p[idx].sum())) * du  # noqa: E731
        sim = simulate(game, table_policy(A, du), draw, n=4000, seed=0)
        rows[name] = (EV, total, np.percentile(sim, [10, 50, 90]))
        print(f"{name:>26} {EB:>7.0f} {EV:>8.2f} {total:>9.2f} {between:>9.2f} {within:>12.2f}   "
              f"{summary(sim)}, sd {sim.std(ddof=1):.2f}, 10/50/90% {np.percentile(sim, [10, 50, 90])}")

    # ---- where is V convex / concave? -----------------------------------------------
    chord = lambda a, b: 0.5 * (v[a // du] + v[b // du]) - v[(a + b) // 2 // du]  # noqa: E731
    print("\nJensen, two-point checks: ½V(m−d) + ½V(m+d) − V(m)")
    for m_, d_ in [(10_000, 8_000), (10_000, 2_000), (40_000, 20_000), (100_000, 40_000)]:
        print(f"  m={m_:>7,}, d={d_:>6,}: {chord(m_ - d_, m_ + d_):+.2f}")
    print(f"\naverage stars per coin: V(B)/B at 5k {v[50] / 5_000 * 1000:.2f}, 20k {v[200] / 20_000 * 1000:.2f}, "
          f"50k {v[500] / 50_000 * 1000:.2f}, 100k {v[1000] / 100_000 * 1000:.2f}, "
          f"150k {v[-1] / 150_000 * 1000:.2f}  (per 1,000 coins)")

    # ---- figure ---------------------------------------------------------------------------
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    m = Bs <= 40_000
    ax.plot(Bs[m], v[m], color="black", lw=1.5, label="V(B), problem 03")
    ax.plot([2_000, 18_000], [v[20], v[180]], "o--", color=ACCENT, lw=1,
            label="two-point budget: chord")
    ax.plot([MEAN], [rows["2,000 or 18,000 (½ each)"][0]], "o", color=ACCENT)
    ax.plot([MEAN], [v[MEAN // du]], "o", color="black")
    ax.annotate(f"E[V(B)] = {rows['2,000 or 18,000 (½ each)'][0]:.0f}", (MEAN, rows["2,000 or 18,000 (½ each)"][0]),
                xytext=(MEAN - 9_500, rows["2,000 or 18,000 (½ each)"][0] + 25), fontsize=8, color=ACCENT)
    ax.annotate(f"V(E[B]) = {v[MEAN // du]:.0f}", (MEAN, v[MEAN // du]),
                xytext=(MEAN + 1_500, v[MEAN // du] - 25), fontsize=8)
    ax.set_xlabel("budget B (coins)")
    ax.set_ylabel("expected total stars")
    ax.set_title("(a) V is convex here: spread in B helps")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(fontsize=8, loc="upper left")
    names = list(rows)
    y = np.arange(len(names))
    for i, n in enumerate(names):
        lo, med, hi = rows[n][2]
        ax2.plot([lo, hi], [i, i], color="#9ecae1", lw=6, solid_capstyle="butt",
                 label="10th–90th percentile (simulated)" if i == 0 else None)
        ax2.plot(med, i, "|", color="black", ms=12, label="median (simulated)" if i == 0 else None)
        ax2.plot(rows[n][0], i, "o", color=ACCENT, label="mean (exact)" if i == 0 else None)
    ax2.set_yticks(y, names, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("total stars")
    ax2.legend(fontsize=7, loc="upper right")
    ax2.set_title("(b) same mean budget, different distributions")
    save(fig, __file__, "budget_spread")


if __name__ == "__main__":
    main()
