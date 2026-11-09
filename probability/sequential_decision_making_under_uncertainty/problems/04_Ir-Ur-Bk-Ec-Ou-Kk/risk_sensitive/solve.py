"""
Problem 04 (Ir-Ur-Bk-Ec-Ou-Kk), method: risk-sensitive DP (exponential utility).

For one family of utilities the extra state isn't needed:

    u_θ(W) = (1 - exp(-θ W)) / θ        θ > 0: concave (avoids risk), θ < 0: convex (seeks it),
                                         θ -> 0: W (problem 03)

Because exp(-θ (W_past + W_future)) = exp(-θ W_past) · exp(-θ W_future), the stars already
collected only scale the objective; they don't change which action is best. So the state
stays (b, L) and the recursion is a product instead of a sum:

    G(b, L) = opt{ m_L G(b-h, L),                                  # hit, m_L = E[exp(-θ j)] at L
                   U_L G(b-C_L, L+1) + (1-U_L) G(b-C_L, L),       # upgrade
                   1 }                                             # stop
    opt = min if θ > 0, max if θ < 0;   G = E[exp(-θ W)] under the best policy.

Certainty equivalent (the sure number of stars worth the same as the gamble):
    CE = -ln(G) / θ.

Run:  python solve.py
"""
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common").is_dir())
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from common.loader import load  # noqa: E402
from common.model import EXAMPLE, Game  # noqa: E402

B_MAX = 200_000   # the level-6 threshold sits near 100k
BI = "problems/03_Ir-Ur-Bk-Ec-Oe-Kk/backward_induction/solve.py"
OS = "problems/03_Ir-Ur-Bk-Ec-Oe-Kk/optimal_stopping/solve.py"
SA = "problems/04_Ir-Ur-Bk-Ec-Ou-Kk/state_augmentation/solve.py"


def solve(game: Game, theta: float, B_max: int = B_MAX, policy=None):
    """G[b_idx, L] = E[exp(-θ W_future)] and action table A (strings), for θ != 0.

    With `policy(b_coins, L)` given, evaluates that policy instead of optimising.
    """
    assert theta != 0
    du = game.unit
    nb = B_max // du + 1
    h, C = game.h // du, [c // du for c in game.C]
    m = game.P @ np.exp(-theta * game.stars)             # per-hit factor at each level
    sign = 1.0 if theta > 0 else -1.0                    # minimise sign * G
    G = np.ones((nb, game.N))
    A = np.full((nb, game.N), "stop", dtype="<U4")
    for b in range(nb):
        for L in range(game.N):
            opts = {"stop": 1.0}
            if b >= h:
                opts["hit"] = m[L] * G[b - h, L]
            if L < game.N - 1 and b >= C[L]:
                u = game.U[L]
                opts["up"] = u * G[b - C[L], L + 1] + (1 - u) * G[b - C[L], L]
            if policy is not None:
                a = policy(b * du, L)
                a = a if a in opts else "stop"
            else:
                # a hit never hurts (u is increasing), so it beats stopping; an upgrade must
                # beat the hit clearly (relative tolerance: G can be as small as 1e-40)
                a = "hit" if "hit" in opts else "stop"
                if "up" in opts and sign * opts["up"] < sign * opts[a] * (1 - sign * 1e-12):
                    a = "up"
            G[b, L], A[b, L] = opts[a], a
    return G, A


def ce(G: np.ndarray, theta: float) -> np.ndarray:
    return -np.log(G) / theta


def main():
    from common.plotting import ACCENT, LEVEL_COLORS, plt, save

    bi, os_, sa = load(BI), load(OS), load(SA)
    game = EXAMPLE
    du = game.unit

    # ---- cross-check against the general method (state augmentation) ----------------
    th = 0.02
    G, _ = solve(game, th, 20_000)
    ut = sa.total_stars_utility(game, lambda W: (1 - np.exp(-th * W)) / th, "exp", 20_000)
    Vsa, _ = sa.solve(game, ut, 20_000)
    for B in (5_000, 20_000):
        print(f"θ={th}, B={B:,}: E[u] here {(1 - G[B // du, 0]) / th:.6f}, "
              f"state augmentation {Vsa[B // du, 0, 0]:.6f}")

    # ---- a dial: thresholds and outcomes vs θ -------------------------------------------
    thetas = [-0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.1]
    _, A03 = bi.solve(game, B_MAX)
    tables = {}
    print(f"\n{'θ':>6} | thresholds b*_L (coins), levels 1-6"
          f"{'':>22} | B=20,000: E[stars]   sd  final lvl   CE   CE of 03 policy")
    for t in thetas:
        A = A03 if t == 0 else solve(game, t)[1]
        b_star = [s for _, s in os_.thresholds(game, A)]
        # outcome at 20k: exact distribution with the (b, L) table, which ignores W
        Aint = np.vectorize({"stop": 0, "hit": 1, "up": 2}.get)(A[: 20_000 // du + 1])
        lin = sa.total_stars_utility(game, lambda W: W, "lin", 20_000)
        st = sa.stats(game, lin, np.repeat(Aint[:, :, None], lin.S, 2), 20_000, {})
        if t == 0:
            ce_opt = ce03 = st["mean"]
        else:
            G, _ = solve(game, t, 20_000)
            G03, _ = solve(game, t, 20_000, policy=lambda b, L: str(A03[b // du, L]))
            ce_opt, ce03 = ce(G[-1, 0], t), ce(G03[-1, 0], t)
        fmt = " ".join(f"{s:>7,}" if s is not None else f"{'—':>7}" for s in b_star)
        print(f"{t:>6} | {fmt} | {st['mean']:>8.2f} {st['sd']:>6.2f} {st['level']:>6.2f} "
              f"{ce_opt:>8.2f} {ce03:>8.2f}")
        tables[t] = (A, b_star, st)

    # ---- figure: thresholds vs θ, and spread vs θ ----------------------------------------
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for L in range(game.N - 1):
        ys = [tables[t][1][L] for t in thetas]
        ax.plot(thetas, ys, "o-", color=LEVEL_COLORS[L + 1], label=f"level {L + 1} → {L + 2}")
    ax.set_yscale("log")
    ax.axvline(0, color="#999999", lw=0.8)
    ax.set_xlabel("θ   (< 0 seeks risk, > 0 avoids it)")
    ax.set_ylabel("upgrade threshold b*_L (coins, log scale)")
    ax.set_title("(a) keep upgrading from L while coins ≥ b*_L")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.text(1.03, 0.35, "no point =\nnever upgrades\nfrom that level\n(budgets up to\n200,000)",
            transform=ax.transAxes, fontsize=7, color="#666666")
    means = [tables[t][2]["mean"] for t in thetas]
    sds = [tables[t][2]["sd"] for t in thetas]
    ax2.errorbar(thetas, means, yerr=sds, fmt="o-", color=ACCENT, capsize=3)
    ax2.axvline(0, color="#999999", lw=0.8)
    ax2.set_xlabel("θ")
    ax2.set_ylabel("total stars: mean ± sd")
    ax2.set_title("(b) outcome at B = 20,000")
    save(fig, __file__, "theta_dial")


if __name__ == "__main__":
    main()
