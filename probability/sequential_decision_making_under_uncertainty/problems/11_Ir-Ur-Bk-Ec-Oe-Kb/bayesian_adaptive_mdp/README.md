# Bayes-adaptive MDP

## Idea

Treat the unknown parameters like hidden state, and put the **belief** about them in the
state, as in [06](../../06_Ir-Ur-Bh-Ec-Oe-Kk/belief_mdp/). With a conjugate prior the belief
stays in its family and is carried by a few numbers. For a success probability $U_L$ with
prior Beta($a_L$, $b_L$), after $s$ successes and $f$ failures the posterior is
Beta($a_L + s$, $b_L + f$).

Here it shrinks further: you never come back to a level, and a success moves you up. So
while on level $L$, $s = 0$ and only $f$ matters. The chance the next try works is

```math
m_L(f) = \frac{a_L}{a_L + b_L + f}
```

and the Bayes-adaptive MDP is an ordinary MDP on $(b, L, f)$:

```math
V(b, L, f) = \max\Big\{ r_L + V(b-h, L, f),\;\; m_L(f)\, V(b - C_L, L+1, 0) + \big(1 - m_L(f)\big)\, V(b - C_L, L, f+1),\;\; 0 \Big\}
```

$b$ only goes down, so 03's table order works. The same recursion with a fixed policy
gives that policy's value averaged over the prior.

## Same family as SLAM, but simpler

SLAM (simultaneous localisation and mapping): a robot estimates its position and a map
while moving, and **how it moves changes what it gets to see**. Choosing moves partly to
learn is called **dual control**: each action both pursues the goal and gathers
information.

Upgrading here has that dual effect: a try either moves you up or teaches you that
$U_L$ is lower than you thought. The Bayes-adaptive DP solves the dual-control problem
exactly, for this tiny belief.

Where it differs from SLAM:

| | SLAM | here |
|---|---|---|
| unknown | robot pose (moves) and map | upgrade odds $U_L$ (fixed) |
| belief | large, continuous (Gaussian or particles) | one counter $f$ |
| learning action | separate choices (look around, revisit) | only by trying the upgrade itself |

With fixed unknowns and learning only through the action you're deciding on, this is
closer to a **Bayesian bandit** (or a stopping problem with learning) than to SLAM.

That last row may be why planning for learning adds so little here: there's no separate
"explore" move to plan. On these examples, playing 03 with the current posterior mean
$m_L(f)$ is within 0.02 stars of the Bayes optimum.

## When it applies

- Unknown parameters with a conjugate prior, and few of them in play at a time.
- Here the item odds $P$ are known. Learning them too (Dirichlet prior, counts of each
  tier at each level) would add counters for the current level's item counts.

## On the running example

**Checks:** simulator (true $U$ drawn from the prior, then the Bayes policy played; 4,000
games) agrees with every exact value on the [11 page](../), e.g. 229.48 ± 1.59 vs 227.91
($\kappa$ = 2, 20,000 coins). One `LONG_SHOT` check came out 88.24 ± 2.52 vs 85.65 at 4,000
games; at 40,000 games it is 86.22 ± 0.79.

**When to give up** on `LONG_SHOT` with 10,000 coins, if every try fails:

| prior | $\kappa$ = 1 | 2 | 5 | 20 | known (03) |
|---|---|---|---|---|---|
| failed tries before giving up | 2 | 3 | 5 | 6 | 7 |

A vague prior with mean 10% puts weight on both "hopeless" and "decent"; a few failures
rule out "decent" quickly.

## Code

[`solve.py`](solve.py)

- `prior(game, kappa)`: Beta parameters with the running example's $U$ as the mean.
- `solve(game, kappa, B_max, policy=None) -> (V, A)`: the DP on $(b, L, f)$; with `policy`
  it evaluates that policy (averaged over the prior).
- `plug_in_policy(game, kappa, B_max, update=True)`: 03's action with $U_L$ set to the
  posterior mean (`update=False`: the prior mean, never updated).
- `oracle_value(...)`: 03's value with the true $U$, averaged over prior draws.
- `play(...)`: one game with true odds drawn from the prior.
- `main()`: all numbers on the 11 page, `bayes_adaptive.svg`.

```
python solve.py
```
