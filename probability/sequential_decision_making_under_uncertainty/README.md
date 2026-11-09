# Sequential decision-making under uncertainty

*One toy game, made harder one assumption at a time. Each step names the tool that solves
it; the tools themselves live in sub-folders with notes and code.*

## The game

A **core** produces **items** rated 1 to $M$ stars. The core has levels $1..N$. Your
currency is **stars** (yes, same word, different thing), and you have a budget of them.
You can spend stars two ways:

- **Hit** the core: pay $h$, receive an item. A higher core level gives better odds of
  high-star items.
- **Upgrade** the core from $L$ to $L+1$: pay $C_L$, succeed with probability $U_L$.
  Upgrades usually get pricier and riskier as $L$ rises.

**Question:** with a given budget, how do you maximise the (expected) total value of your
items?

Every variant below is a version of one trade-off: **invest** (upgrade, so future hits
are better) vs. **harvest** (hit now, and get more items).

*"Sequential"* means you decide, see what happened, and decide again. In general the answer is
therefore a **policy** (a rule: in this situation, do this), not a fixed plan.

## Notation

| Symbol | Meaning |
|---|---|
| $N$, $L$ | number of core levels, current level |
| $M$, $j$ | number of star tiers, a tier |
| $P(L, j)$ | probability a hit at level $L$ gives a $j$★ item |
| $w_j$ | value of a $j$★ item (running example: $w_j = j$) |
| $r_L = \sum_j w_j P(L,j)$ | expected value of one hit at level $L$ |
| $h$ | cost of a hit |
| $C_L$, $U_L$ | cost and success probability of upgrading $L \to L+1$ |
| $D_L$ | probability of dropping a level when an upgrade fails |
| $B$, $F_B$, $S(x)$ | budget, its distribution, its survival function $P(B \ge x)$ |
| $b$, $s$ | stars remaining, stars spent so far |
| $g_j$, $\rho$, $T$, $t$ | stars per tick earned by a $j$★ item, total income rate, horizon, time |
| $V$, $\pi$ | value function, policy |

### Running example

The same numbers are used everywhere, so results can be compared across problems
([`common/model.py`](common/model.py)):

- $N = 7$ core levels, $M = 5$ star tiers, $h = 200$, $w_j = j$
- $C = (400, 600, 900, 1300, 1800, 2500)$
- $U = (0.70, 0.60, 0.50, 0.40, 0.30, 0.25)$
- $P(L, \cdot)$ shifts from mostly 1★ at level 1 to mostly 5★ at level 7, giving
  $r = (1.36, 1.78, 2.34, 3.00, 3.66, 4.22, 4.64)$

## Classifying the variants: six axes

Every variant gets a tag `I?-U?-B?-E?-O?-K?`, one letter per axis:

| Axis | Values |
|---|---|
| **I** item outcome | `Id` deterministic (each hit gives exactly $r_L$) · `Ir` random |
| **U** upgrade | `Ud` sure · `Ur` random · `Ux` random, failure may demote · `Up` pity (chance rises after failures) |
| **B** budget | `Bk` known · `Bs` random, seen before play · `Bh` hidden (you find out when you run out) · `Bt` trickle (stars arrive over time from outside) |
| **E** economy | `Ec` closed · `Er` items refund stars once · `Ei` items generate income (a factory) · `Es` items can be sold |
| **O** objective | `Oe` expected value · `Ou` utility (risk-averse) · `Ot` hit a target (e.g. P(at least one 5★)) |
| **K** knowledge of the probabilities | `Kk` known · `Kb` known family, unknown parameters (Bayesian) · `Ku` unknown |

Two notes on the **K** axis:

- **A parametric distribution vs. a given table is not an axis.** Both are `Kk`. The
  solver only ever sees the table. The parametric form still matters for closed-form
  insights (an exponential budget is memoryless, so the policy is stationary) and for how
  fast you can learn (`Kb`).
- **Known vs. unknown is an axis.** Not knowing the odds adds a new trade-off: explore
  (spend to learn) vs. exploit.

### Reductions: most tags collapse

There are $2 \cdot 4 \cdot 4 \cdot 4 \cdot 3 \cdot 3 = 1152$ tags, but most reduce to
something already solved. Most of the hard variants become a standard **MDP on a bigger
state**. Learning to spot these reductions is half the skill.

![reduction map](figures/reduction_map.svg)

## The ladder

Each problem loosens one axis of an earlier one.
Status: ✅ built · ⬜ planned.

![ladder map](figures/ladder_map.svg)

| # | Tag | What changes | Tools | |
|---|---|---|---|---|
| [01](problems/01_Id-Ud-Bk-Ec-Oe-Kk/) | `Id-Ud-Bk-Ec-Oe-Kk` | nothing random | enumeration, deterministic DP | ✅ |
| 02 | `Ir-Ud-Bk-Ec-Oe-Kk` | item quality random | reduction (linearity of expectation) | ⬜ |
| 03 | `Ir-Ur-Bk-Ec-Oe-Kk` | upgrades can fail | MDP / backward induction, optimal stopping, fluid approximation | ⬜ |
| 04 | `Ir-Ur-Bk-Ec-Ot/Ou-Kk` | nonlinear objective | state augmentation, risk-sensitive DP | ⬜ |
| 05 | `Ir-Ux-Bk-Ec-Oe-Kk` | failed upgrade may demote | value iteration | ⬜ |
| 06 | `Ir-Up-Bk-Ec-Oe-Kk` | pity counter | state augmentation | ⬜ |
| 07 | `Ir-Ur-Bs-Ec-Oe-Kk` | budget random, seen first | reduction (average) | ⬜ |
| 08 | `Ir-Ur-Bh-Ec-Oe-Kk` | budget hidden | survival analysis, POMDP → MDP, value of information | ⬜ |
| 09 | `Ir-Ur-Bk-Er-Oe-Kk` | items refund stars | value iteration | ⬜ |
| 10 | `Ir-Ur-Bt-Ec-Oe-Kk` | outside income over time | time-indexed DP | ⬜ |
| 11 | `Ir-Ur-Bk-Ei-Oe-Kk` | items generate income (factory) | time-indexed DP, optimal control (Pontryagin, turnpike) | ⬜ |
| 12 | `Ir-Ur-Bk-Es-Oe-Kk` | items can be sold | inventory state | ⬜ |
| 13 | `Ir-Ur-Bk-Ec-Oe-Kb` | odds unknown, family known | Bayesian adaptive MDP, Thompson sampling | ⬜ |
| 14 | `Ir-Ur-Bk-Ec-Oe-Ku` | nothing known | model-free RL | ⬜ |
| 15 | `Ir-Ur-Bh-Ei-Oe-Kb` | everything at once | policy search, approximate DP | ⬜ |

### 01 · Fully deterministic: `Id-Ud-Bk-Ec-Oe-Kk`

**What changed:** nothing; this is the baseline. Every hit gives exactly $r_L$, every
upgrade succeeds.

**Formulation:** state $(b, L)$; actions hit, upgrade, stop.

$$V(b,L) = \max\{\, r_L + V(b-h, L),\; V(b-C_L, L+1),\; 0 \,\}$$

**Key structure:** if $r_L$ increases with $L$, upgrade first, then hit. The only decision is the
target level $k$.

**Tools:** [enumeration](problems/01_Id-Ud-Bk-Ec-Oe-Kk/enumeration/) (try every $k$);
[deterministic DP](problems/01_Id-Ud-Bk-Ec-Oe-Kk/deterministic_dp/) (longest path in a
DAG; needed once the structure breaks, e.g. a per-level hit cap).

**On the running example:** richer means investing more. The best target level is 3 at
$B = 5\,000$, 6 at $20\,000$, and 7 from about $32\,600$ up. Value at $B = 20\,000$: **316.5**.

### 02 – 15

*Planned.*

## Results so far

Each problem reports its optimal (expected) value on the running example:

| # | $B = 5\,000$ | $B = 20\,000$ | Note |
|---|---|---|---|
| 01 | 46.80 | 316.50 | upgrades always succeed (an upper bound for 03) |

## Fields this touches

The same core problem appears under different names:

| Field | Name for it | Emphasis |
|---|---|---|
| Operations research | stochastic DP, Markov decision processes | exact solutions, structure of optimal policies |
| Control theory | stochastic optimal control | continuous time, HJB equation, Pontryagin |
| Economics | dynamic optimisation, capital accumulation | shadow prices, turnpike theorems |
| CS / ML | reinforcement learning | huge state spaces, learning unknown dynamics |
| Applied probability | optimal stopping, survival analysis | hazard rates, random horizons |
| Game design | gacha / idle-game economy | tuning $P$, $U$, $C$ so players face interesting trade-offs |

## References

- S. Ross, *Introduction to Stochastic Dynamic Programming*: short, great on optimal stopping. Start here.
- M. Puterman, *Markov Decision Processes*: the standard MDP reference.
- D. Bertsekas, *Dynamic Programming and Optimal Control*, Vol. 1: deterministic DP through control.
- W. Powell, *Reinforcement Learning and Stochastic Optimization*: one framework spanning OR, control and RL.
- R. Sutton & A. Barto, *Reinforcement Learning: An Introduction*: when the model is unknown or too big.

## Repo layout & running the code

```
common/      model (Game, EXAMPLE), Monte Carlo simulator, plot style, loader
figures/     graphviz sources + svg (./figures/build.sh)
problems/NN_<tag>/            problem README
problems/NN_<tag>/<method>/   method README + solve.py + svg
```

Each `solve.py` runs on its own (`python solve.py`). Every solver is checked against the
simulator in [`common/simulate.py`](common/simulate.py). The rule of thumb: **DP finds a
policy; simulation evaluates one.** Requires numpy and matplotlib.
