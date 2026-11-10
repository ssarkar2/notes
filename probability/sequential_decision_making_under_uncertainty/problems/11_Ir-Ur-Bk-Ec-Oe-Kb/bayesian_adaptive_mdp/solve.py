"""
Problem 11 (Ir-Ur-Bk-Ec-Oe-Kb), method: Bayes-adaptive MDP.

The item odds P are known; the upgrade odds U_L are NOT. Each U_L has a Beta(a_L, b_L)
prior (independent across levels), with mean equal to the running example's U_L and
concentration kappa = a_L + b_L (small kappa = very unsure).

Belief. Without demotion you never return to a level, so the only belief that matters is
about the level you're on. While at L you've had f failures and no success (a success
moves you up), so the posterior is Beta(a_L, b_L + f) and the chance the next attempt
works is its mean

    m_L(f) = a_L / (a_L + b_L + f)

The belief is one counter, f: the Bayes-adaptive MDP is an ordinary MDP on (b, L, f):

    V(b, L, f) = max{ r_L + V(b - h, L, f),                                      # hit
                      m_L(f) V(b - C_L, L+1, 0) + (1 - m_L(f)) V(b - C_L, L, f+1), # upgrade
                      0 }

b only goes down: 03's order still works. V(B, 0, 0) is the expected value averaged over
the prior (over which U the game really has), under the Bayes-optimal policy.

Compared with:
  - known odds: 03 with U = the prior mean (what you'd play if you ignored the uncertainty);
  - plug in the posterior mean m_L(f), and play 03 as if it were the truth;
  - oracle: told the true U before playing (average of 03's value over the prior).

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

BI = "problems/03_Ir-Ur-Bk-Ec-Oe-Kk/backward_induction/solve.py"
STOP, HIT, UP = 0, 1, 2
CODE = {"stop": STOP, "hit": HIT, "up": UP}


def prior(game: Game, kappa: float):
    mu = np.array(game.U)
    return kappa * mu, kappa * (1 - mu)


def solve(game: Game, kappa: float, B_max: int, policy=None):
    """V[b_idx, L, f], A[b_idx, L, f]; with `policy(b_idx, L, f) -> code` evaluates it."""
    du = game.unit
    nb = B_max // du + 1
    h, C = game.h // du, [c // du for c in game.C]
    F = nb // min(C) + 2
    a, bb = prior(game, kappa)
    V = np.zeros((nb, game.N, F + 1))
    A = np.zeros((nb, game.N, F + 1), dtype=np.int8)
    f = np.arange(F)
    for b in range(nb):
        for L in range(game.N):
            q = np.zeros((3, F))
            q[HIT] = -np.inf if b < h else game.r[L] + V[b - h, L, :F]
            q[UP] = -np.inf
            if L < game.N - 1 and b >= C[L]:
                m = a[L] / (a[L] + bb[L] + f)
                q[UP] = m * V[b - C[L], L + 1, 0] + (1 - m) * V[b - C[L], L, 1:F + 1]
            if policy is None:
                act = q.argmax(0)
                act[(act == STOP) & (q[HIT] >= 0)] = HIT
            else:
                act = np.array([policy(b, L, k) for k in range(F)])
            V[b, L, :F] = q[act, f]
            A[b, L, :F] = act
    return V, A


def plug_in_policy(game: Game, kappa: float, B_max: int, update: bool = True):
    """Play 03 as if U_L were the posterior mean m_L(f) (update=True) or the prior mean."""
    bi = load(BI)
    a, bb = prior(game, kappa)
    cache = {}

    def table(L, f):
        u = a[L] / (a[L] + bb[L] + (f if update else 0))
        key = (L, round(u, 12))
        if key not in cache:
            U = list(game.U)
            U[L] = u
            cache[key] = bi.solve(replace(game, U=tuple(U)), B_max)[1]
        return cache[key]

    def pi(b, L, f):
        if L == game.N - 1:
            return CODE[str(table(0, 0)[b, L])]
        return CODE[str(table(L, f)[b, L])]
    return pi


def oracle_value(game: Game, kappa: float, B: int, n: int, rng) -> tuple:
    """Average over the prior of 03's optimal value when the true U is known."""
    bi = load(BI)
    a, bb = prior(game, kappa)
    vals = np.empty(n)
    for i in range(n):
        U = tuple(rng.beta(a, bb))
        vals[i] = bi.solve(replace(game, U=U), B)[0][-1, 0]
    return vals.mean(), 1.96 * vals.std(ddof=1) / np.sqrt(n)


def play(game: Game, A: np.ndarray, U_true, B: int, rng) -> float:
    """One game with true odds U_true, under a table over (b, L, f)."""
    du = game.unit
    b, L, f, stars = B, 0, 0, 0.0
    while True:
        a = A[b // du, L, f]
        if a == STOP:
            return stars
        if a == HIT:
            b -= game.h
            stars += rng.choice(game.M, p=game.P[L]) + 1
        else:
            b -= game.C[L]
            if rng.random() < U_true[L]:
                L, f = L + 1, 0
            else:
                f += 1


def run(game: Game, name: str, budgets, kappas, n_oracle: int, rng):
    from common.simulate import summary
    bi = load(BI)
    B_max = max(budgets)
    V03, _ = bi.solve(game, B_max)
    du = game.unit
    print(f"\n{name}: value averaged over the prior (exact), B = {budgets}")
    print(f"{'kappa':>6} {'budget':>7} {'known odds':>10} {'ignore learning':>15} {'plug in m(f)':>12} "
          f"{'Bayes-optimal':>13} {'oracle':>16}   simulator (Bayes policy)")
    out = {}
    for kappa in kappas:
        V, A = solve(game, kappa, B_max)
        Vp, _ = solve(game, kappa, B_max, policy=plug_in_policy(game, kappa, B_max))
        Vn, _ = solve(game, kappa, B_max, policy=plug_in_policy(game, kappa, B_max, update=False))
        a, bb = prior(game, kappa)
        for B in budgets:
            i = B // du
            orc, ci = oracle_value(game, kappa, B, n_oracle, rng)
            sims = np.array([play(game, A, rng.beta(a, bb), B, rng) for _ in range(4000)])
            out[(kappa, B)] = (V03[i, 0], Vn[i, 0, 0], Vp[i, 0, 0], V[i, 0, 0], orc)
            print(f"{kappa:>6} {B:>7,} {V03[i, 0]:>10.2f} {Vn[i, 0, 0]:>15.2f} {Vp[i, 0, 0]:>12.2f} "
                  f"{V[i, 0, 0]:>13.2f} {orc:>9.2f} ± {ci:.2f}   {summary(sims)}")
        out[(kappa, "A")] = A
    return out


def main():
    from common.plotting import ACCENT, plt, save

    rng = np.random.default_rng(0)
    ex = run(EXAMPLE, "Running example", (5_000, 20_000), (2, 5, 20), 400, rng)
    ls = run(LONG_SHOT, "LONG_SHOT (one upgrade, prior mean 10%)", (5_000, 10_000), (1, 2, 5, 20), 2000, rng)

    # ---- how many failed tries before giving up (LONG_SHOT) -------------------------------
    print("\nLONG_SHOT at 10,000 coins: tries before giving up if every try fails")
    du = LONG_SHOT.unit
    tries = {}
    for kappa in (1, 2, 5, 20, None):
        if kappa is None:
            A = load(BI).solve(LONG_SHOT, 10_000)[1]
            b, k = 10_000, 0
            while A[b // du, 0] == "up":
                b, k = b - 1000, k + 1
        else:
            A = ls[(kappa, "A")]
            b, f = 10_000, 0
            while A[b // du, 0, f] == UP:
                b, f = b - 1000, f + 1
            k = f
        tries["known" if kappa is None else kappa] = k
        print(f"  kappa = {kappa if kappa else 'inf (known)'}: {k}")

    # ---- figure ---------------------------------------------------------------------------
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    B = 10_000
    ks = (1, 2, 5, 20)
    labels = ["odds known, = prior mean (03)", "ignore learning", "plug in m(f)", "Bayes-optimal", "oracle"]
    cols = ["black", "grey", "#2171b5", ACCENT, "#6a3d9a"]
    x = np.arange(len(ks))
    w = 0.16
    for j, (lab, col) in enumerate(zip(labels, cols)):
        ax.bar(x + (j - 2) * w, [ls[(k, B)][j] for k in ks], w, color=col, label=lab)
    ax.set_xticks(x, [f"κ = {k}" for k in ks])
    ax.set_ylabel("expected total stars (averaged over the prior)")
    ax.set_title("(a) LONG_SHOT, 10,000 coins: value of each policy")
    ax.set_ylim(0, 125)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    kk = list(tries)
    ax2.bar(range(len(kk)), [tries[k] for k in kk], color=[ACCENT] * (len(kk) - 1) + ["black"])
    ax2.set_xticks(range(len(kk)), [f"κ = {k}" if k != "known" else "known\n(κ = ∞)" for k in kk])
    ax2.set_ylabel("failed tries before giving up")
    ax2.set_title("(b) LONG_SHOT, 10,000 coins: when to give up")
    save(fig, __file__, "bayes_adaptive")


if __name__ == "__main__":
    main()
