# Model-free RL (Q-learning), and a model-based learner

## Idea

**Q-learning.** Keep a table $Q(b, L, a)$. After each move $(b, L) \xrightarrow{a} (b', L')$
with reward $\rho$ (the item's stars for a hit, 0 for an upgrade), nudge towards what that
one step says:

```math
Q(b, L, a) \leftarrow Q(b, L, a) + \alpha \Big( \rho + \max_{a'} Q(b', L', a') - Q(b, L, a) \Big)
```

It never estimates $P$ or $U$: the sampled next state stands in for the expectation.
Exploration: $\varepsilon$-greedy, a random legal action with probability $\varepsilon$.
Step size: constant $\alpha$ = 0.1, or $\alpha$ = 1/(visits to this state and action).

The rules are known; only the odds aren't. Stopping early never helps (leftover coins
are worth nothing), so "stop" is used only when nothing is affordable.

**Model-based learner.** Count what happened: items of each tier at each level, upgrade
tries and successes. Estimate $P(L, j)$ by frequencies and $U_L$ by
(successes + 1)/(tries + 2). Levels never reached get an optimistic guess (every hit 5★),
which pulls the learner towards trying them. Every 100 games, solve 03 with the
estimates and play that policy, $\varepsilon$-greedy.

## When it applies

- **Q-learning**: when you can play but can't write the transition model, or the model is
  too big to solve. It only needs the ability to try things.
- **Model-based**: when the model has few unknowns and a solver exists for known numbers.
  Here: 6 upgrade odds and 7 rows of item odds, plus 03's backward induction.

## On the running example

Full tables on the [12 page](../). Settings: 5 seeds, $\varepsilon$ = 0.1, budget 10,000
(running example) and 5,000 (`LONG_SHOT`), true value of the greedy policy computed
exactly with 03's policy evaluation.

- Model-based: optimal to 0.1 stars after 300 games on both examples (worst seed 100.20
  and 28.90).
- Q-learning after 100,000 games: 96.56 ($\alpha$ = 0.1) and 90.18 ($\alpha$ = 1/visits)
  against 100.25; on `LONG_SHOT` 25.46 and 27.52 against 28.90.
- Early on, Q-learning's greedy policy can be worse than never upgrading (24.40 vs 25.00 on
  `LONG_SHOT`): with most entries still 0, its comparisons are between guesses.

**A bug worth knowing about.** A first version ended the game whenever only one action
was legal, which includes "hit" at the top level. Q-learning then never saw those
states, and on `LONG_SHOT` it never learned to upgrade at all. The model-based learner
was unaffected: its optimistic guess for unseen levels stood in. Model-free learners
see only what the environment shows them.

## Code

[`solve.py`](solve.py)

- `Env(game, B, rng)`: the game as a black box (`legal`, `step`).
- `q_learning(game, B, episodes, eps, alpha, seed, checkpoints)`: returns the true value of
  the greedy policy at each checkpoint, and $Q$.
- `model_based(game, B, episodes, eps, seed, checkpoints, every=100)`.
- `greedy_table`, `true_value`: turn $Q$ into a policy and score it exactly.
- `main()`: all numbers and `learning_curves.svg` (about 3 minutes).

```
python solve.py
```
