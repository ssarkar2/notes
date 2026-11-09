"""
Problem 01 (Id-Ud-Bk-Ec-Oe-Kk), method: deterministic dynamic programming.

A deterministic DP is a longest-path problem on a DAG: nodes are states, edges are
actions (each with a cost and a reward). Value of a state = best reward-to-go:

    V(state) = max( 0,  max_{action}  reward(action) + V(next_state) )

The solver below is *generic*: it only needs a function  actions(state) -> list of
(name, reward, next_state). It needs no insight about the problem's structure.

Two demos on the running example:
  (a) plain game, state (b, L): reproduces the enumeration answer exactly.
  (b) "worn core" (hit-cap mechanic): each level can be hit at most CAP times. Now "upgrade to k,
      then hit" is no longer optimal (you want hits at several levels), so enumeration
      over k fails, while the DP (state (b, L, hits_at_L)) still just works.

Run:  python solve.py     (prints comparison + an optimal plan, writes dp_vs_enumeration.svg)
"""
import sys
from functools import lru_cache
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common").is_dir())
sys.path.insert(0, str(ROOT))
sys.setrecursionlimit(100_000)

import numpy as np  # noqa: E402

from common.loader import load  # noqa: E402
from common.model import EXAMPLE, Game, deterministic_items, sure_upgrades  # noqa: E402

CAP = 15  # max hits per core level in the "worn core" (hit-cap mechanic)


def solve(start, actions):
    """Generic deterministic DP. Returns (value, plan) where plan is a list of action names."""
    @lru_cache(maxsize=None)
    def best(state):
        v, choice = 0.0, None                       # "stop" is always allowed, value 0
        for name, reward, nxt in actions(state):
            cand = reward + best(nxt)[0]
            if cand > v:
                v, choice = cand, (name, nxt)
        return v, choice

    value, plan, state = best(start)[0], [], start
    while (choice := best(state)[1]) is not None:
        plan.append(choice[0])
        state = choice[1]
    return value, plan


# ---- the two models, written as action functions -------------------------------------
def plain_actions(game: Game):
    r = game.r

    def actions(state):
        b, L = state
        out = []
        if b >= game.h:
            out.append((f"hit@{L + 1}", r[L], (b - game.h, L)))
        if L < game.N - 1 and b >= game.C[L]:
            out.append((f"up->{L + 2}", 0.0, (b - game.C[L], L + 1)))
        return out
    return actions


def capped_actions(game: Game, cap: int):
    r = game.r

    def actions(state):
        b, L, k = state
        out = []
        if b >= game.h and k < cap:
            out.append((f"hit@{L + 1}", r[L], (b - game.h, L, k + 1)))
        if L < game.N - 1 and b >= game.C[L]:
            out.append((f"up->{L + 2}", 0.0, (b - game.C[L], L + 1, 0)))
        return out
    return actions


def capped_target_plan(game: Game, B: int, cap: int) -> float:
    """Best 'upgrade to k, then hit' plan in the capped game (what enumeration would do)."""
    Ccum = np.concatenate([[0], np.cumsum(game.C)])
    hits = np.minimum(np.floor((B - Ccum) / game.h), cap)
    return float(max(0.0, np.max(np.where(B >= Ccum, hits * game.r, -np.inf))))


def compress(plan):
    """['hit@3','hit@3','up->4'] -> 'hit@3 x2, up->4'"""
    out, i = [], 0
    while i < len(plan):
        j = i
        while j < len(plan) and plan[j] == plan[i]:
            j += 1
        out.append(plan[i] + (f" x{j - i}" if j - i > 1 else ""))
        i = j
    return ", ".join(out)


def main():
    from common.plotting import ACCENT, plt, save

    enum = load("problems/01_Id-Ud-Bk-Ec-Oe-Kk/enumeration/solve.py")
    game = sure_upgrades(deterministic_items(EXAMPLE))
    Bs = np.arange(0, 40_001, 500)

    # (a) plain game: DP must equal enumeration
    dp_plain = np.array([solve((B, 0), plain_actions(game))[0] for B in Bs])
    en_plain = np.array([enum.solve(game, B)[0] for B in Bs])
    print(f"(a) plain:  max |DP - enumeration| over {len(Bs)} budgets = "
          f"{np.abs(dp_plain - en_plain).max():.2e}")

    # (b) capped game: DP beats the best target-level plan
    dp_cap = np.array([solve((B, 0, 0), capped_actions(game, CAP))[0] for B in Bs])
    en_cap = np.array([capped_target_plan(game, B, CAP) for B in Bs])
    i = int(np.argmax(dp_cap - en_cap))
    print(f"(b) capped (CAP={CAP}): largest gap at B={Bs[i]}: "
          f"DP {dp_cap[i]:.2f} vs best target plan {en_cap[i]:.2f}")
    B = 20_000
    v, plan = solve((B, 0, 0), capped_actions(game, CAP))
    print(f"    optimal plan at B={B} (value {v:.2f}):\n    {compress(plan)}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    ax = axes[0]
    ax.plot(Bs, en_plain, color="black", lw=3, alpha=0.3, label="enumeration")
    ax.plot(Bs, dp_plain, color=ACCENT, lw=1.2, ls="--", label="deterministic DP")
    ax.set_title("(a) plain game: identical")
    ax.set_xlabel("budget B (coins)")
    ax.set_ylabel("total stars")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(Bs, en_cap, color="black", lw=1.5, label="best 'upgrade to k, then hit'")
    ax.plot(Bs, dp_cap, color=ACCENT, lw=1.5, label="deterministic DP")
    ax.fill_between(Bs, en_cap, dp_cap, color=ACCENT, alpha=0.15)
    ax.set_title(f"(b) worn core, ≤{CAP} hits per level: DP wins")
    ax.set_xlabel("budget B (coins)")
    ax.legend(fontsize=8)
    save(fig, __file__, "dp_vs_enumeration")


if __name__ == "__main__":
    main()
