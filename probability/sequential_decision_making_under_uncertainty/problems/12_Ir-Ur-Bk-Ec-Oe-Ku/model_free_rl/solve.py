"""
Problem 12 (Ir-Ur-Bk-Ec-Oe-Ku), method: model-free RL (Q-learning), vs a model-based learner.

Nothing is known about the odds: no P, no U, no prior. You play the same game (budget B)
over and over and learn from what happens. Item ratings are 1..5 stars (you can see that).

Q-learning keeps a value Q(b, L, a) for every state and action and nudges it towards
what one step of experience says:

    Q(b, L, a) <- Q(b, L, a) + alpha * ( reward + max_a' Q(b', L', a') - Q(b, L, a) )

with reward = stars of the item for a hit, 0 for an upgrade. No model of P or U is ever
built. It explores with epsilon-greedy: a random action with probability epsilon. The rules are
known, so "stop" is only used when nothing is affordable.

Model-based learner, for comparison: count what happened (items per level, upgrade
tries and successes), plug the estimates into 03's backward induction every 100 games,
and play that policy (also epsilon-greedy). Levels never reached get an optimistic guess
(5 stars per hit, upgrade odds 1/2), so it is drawn to try them.

Score: the TRUE value of the greedy policy learned so far (exact, 03's policy evaluation),
after a given number of training games.

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
NAMES = np.array(["stop", "hit", "up"])


class Env:
    """The game as a black box: step(b, L, a) -> (reward, b', L', done)."""

    def __init__(self, game: Game, B: int, rng):
        self.g, self.B, self.rng = game, B, rng
        self.cumP = np.cumsum(game.P, 1)
        self.du = game.unit

    def legal(self, b, L):
        """Rules are known, odds aren't. Stopping early never helps (leftover coins are
        worth nothing), so "stop" is only offered when nothing is affordable."""
        g = self.g
        acts = []
        if b >= g.h:
            acts.append(HIT)
        if L < g.N - 1 and b >= g.C[L]:
            acts.append(UP)
        return acts or [STOP]

    def step(self, b, L, a):
        g = self.g
        if a == HIT:
            j = int(np.searchsorted(self.cumP[L], self.rng.random(), side="right")) + 1
            return j, j, b - g.h, L                           # reward, item stars, b', L'
        if a == UP:
            ok = self.rng.random() < g.U[L]
            return 0, None if ok else -1, b - g.C[L], L + ok
        raise ValueError


def greedy_table(Q, env):
    """Greedy action for every (b, L) from Q (ties: hit, then stop, then up)."""
    nb, N = Q.shape[:2]
    A = np.full((nb, N), "stop", dtype="<U4")
    for bi in range(nb):
        for L in range(N):
            acts = env.legal(bi * env.du, L)
            best = max(acts, key=lambda a: (Q[bi, L, a], a == HIT, a == STOP))
            A[bi, L] = NAMES[best]
    return A


def true_value(game, A, B):
    bi = load(BI)
    pol = lambda b, L: str(A[b // game.unit, L])           # noqa: E731
    return bi.evaluate(game, pol, B)[B // game.unit, 0]


def q_learning(game: Game, B: int, episodes: int, eps: float, alpha: str, seed: int, checkpoints):
    """alpha = 'const' (0.1) or 'visits' (1 / number of visits to (s, a))."""
    rng = np.random.default_rng(seed)
    env = Env(game, B, rng)
    du = env.du
    nb = B // du + 1
    Q = np.zeros((nb, game.N, 3))
    n = np.zeros((nb, game.N, 3))
    curve = []
    for ep in range(1, episodes + 1):
        b, L = B, 0
        while True:
            acts = env.legal(b, L)
            if acts == [STOP]:
                break
            bi = b // du
            if rng.random() < eps:
                a = acts[rng.integers(len(acts))]
            else:
                a = max(acts, key=lambda x: (Q[bi, L, x], x == HIT))
            if a == STOP:
                break
            r, _, b2, L2 = env.step(b, L, a)
            target = r + max(Q[b2 // du, L2, x] for x in env.legal(b2, L2))
            n[bi, L, a] += 1
            lr = 0.1 if alpha == "const" else 1.0 / n[bi, L, a]
            Q[bi, L, a] += lr * (target - Q[bi, L, a])
            b, L = b2, L2
        if ep in checkpoints:
            curve.append(true_value(game, greedy_table(Q, env), B))
    return np.array(curve), Q


def model_based(game: Game, B: int, episodes: int, eps: float, seed: int, checkpoints, every=100):
    rng = np.random.default_rng(seed)
    env = Env(game, B, rng)
    bi_mod = load(BI)
    du = env.du
    items = np.zeros((game.N, game.M))                        # counts of j-star items per level
    tries = np.zeros(game.N - 1)
    wins = np.zeros(game.N - 1)
    A = None
    curve = []

    def estimate():
        P = np.empty_like(game.P)
        for L in range(game.N):
            if items[L].sum() > 0:
                P[L] = items[L] / items[L].sum()
            else:
                P[L] = np.eye(game.M)[-1]                      # optimistic: all 5-star
        U = tuple((wins + 1) / (tries + 2))                   # 1/2 when never tried
        g = replace(game, P=P, U=U)
        return bi_mod.solve(g, B)[1]

    for ep in range(1, episodes + 1):
        if A is None or ep % every == 1:
            A = estimate()
        b, L = B, 0
        while True:
            acts = env.legal(b, L)
            if acts == [STOP]:
                break
            if rng.random() < eps:
                a = acts[rng.integers(len(acts))]
            else:
                a = {"stop": STOP, "hit": HIT, "up": UP}[str(A[b // du, L])]
                if a not in acts:
                    a = HIT if HIT in acts else STOP
            if a == STOP:
                break
            r, j, b2, L2 = env.step(b, L, a)
            if a == HIT:
                items[L, j - 1] += 1
            else:
                tries[L] += 1
                wins[L] += L2 > L
            b, L = b2, L2
        if ep in checkpoints:
            curve.append(true_value(game, estimate(), B))
    return np.array(curve)


def main():
    import time

    from common.plotting import ACCENT, plt, save

    bi = load(BI)
    seeds = range(5)
    checkpoints = [10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (game, name, B) in zip(axes, [(EXAMPLE, "running example", 10_000),
                                          (LONG_SHOT, "LONG_SHOT", 5_000)]):
        V, _ = bi.solve(game, B)
        hit_only = bi.evaluate(game, bi.target_policy(game, 0), B)[B // game.unit, 0]
        print(f"\n{name}, B = {B:,}: optimal {V[-1, 0]:.2f}, never upgrade {hit_only:.2f}")
        runs = {}
        for label, fn in [
            ("Q-learning, α = 1/visits, ε = 0.1", lambda s: q_learning(game, B, checkpoints[-1], 0.1, "visits", s, checkpoints)[0]),
            ("Q-learning, α = 0.1, ε = 0.1", lambda s: q_learning(game, B, checkpoints[-1], 0.1, "const", s, checkpoints)[0]),
            ("model-based (counts → 03), ε = 0.1", lambda s: model_based(game, B, checkpoints[-1], 0.1, s, checkpoints)),
        ]:
            t = time.time()
            c = np.array([fn(s) for s in seeds])
            runs[label] = c
            print(f"  {label:38s} ({time.time() - t:.0f}s)  mean over {len(seeds)} seeds of the greedy "
                  f"policy's true value after " + ", ".join(f"{k:,}: {m:.2f}" for k, m in zip(checkpoints, c.mean(0))))
            print(f"  {'':38s}        worst seed: " + ", ".join(f"{m:.2f}" for m in c.min(0)))
        cols = ["#9ecae1", "#2171b5", ACCENT]
        for (label, c), col in zip(runs.items(), cols):
            ax.plot(checkpoints, c.mean(0), "o-", ms=3, color=col, label=label)
            ax.fill_between(checkpoints, c.min(0), c.max(0), color=col, alpha=0.15)
        ax.axhline(V[-1, 0], color="black", ls="--", lw=1, label="optimal (knows the odds)")
        ax.axhline(hit_only, color="grey", ls=":", lw=1, label="never upgrade")
        ax.set_xscale("log")
        ax.set_xlabel("training games played")
        ax.set_ylabel("true value of the learned greedy policy")
        ax.set_title(f"{name}, B = {B:,} (5 seeds; band = min to max)")
        ax.legend(fontsize=7, loc="lower right")
    save(fig, __file__, "learning_curves")


if __name__ == "__main__":
    main()
