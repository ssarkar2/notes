# Sequential decision-making under uncertainty

*One toy game, made harder one assumption at a time. Each step names the tool that solves
it; the tools themselves live in sub-folders with notes and code.*

## The game

A **core** produces **items** rated 1 to $M$ stars. The core has levels $1..N$. Your
currency is **coins**, and you have a budget of them. You can spend coins two ways:

- **Hit** the core: pay $h$, receive an item. A higher core level gives better odds of
  high-star items.
- **Upgrade** the core from $L$ to $L+1$: pay $C_L$, succeed with probability $U_L$.
  Upgrades usually get pricier and riskier as $L$ rises.

**Question:** with a given budget, how do you play to maximise the expected **utility**
$E[u(j_1, j_2, \dots)]$? Here $j_k$ is the star rating of the $k$-th item you get, and $u$
says how much you like the whole collection. Coins left over at the end count for nothing.
Examples of $u$:

- **total stars** $j_1 + j_2 + \dots$: the **baseline**, used until problem 04;
- "at least one 5★": $u = 1$ if any $j_k = 5$, else 0;
- a concave function of total stars (e.g. $\sqrt{\cdot}$), for a player who dislikes risk;
- a convex function of total stars (e.g. $(\cdot)^2$), for a player who likes risk.

Every variant below is a version of one trade-off: **invest** (upgrade, so future hits
are better) vs. **harvest** (hit now, and get more items).

*"Sequential"* means you decide, see what happened, and decide again. In general the answer is
therefore a **policy** (a rule: in this situation, do this), not a fixed plan.

## Notation

Words: **stars** only ever describe item quality; **coins** are the currency; **utility**
$u$ is what we maximise (in expectation); **value** $V$ is the best achievable expected
utility (expected total stars, under the baseline).

| Symbol | Meaning |
|---|---|
| $N$, $L$ | number of core levels, current level |
| $M$, $j$ | number of star tiers, a tier |
| $P(L, j)$ | probability a hit at level $L$ gives a $j$★ item |
| $j_k$ | star rating of the $k$-th item obtained |
| $u(j_1, j_2, \dots)$ | utility of the collection; baseline $u = \sum_k j_k$ (total stars) |
| $W$ | total stars so far, $\sum_k j_k$; plays the role of wealth in $u(W)$ |
| $r_L = \sum_j j \cdot P(L,j)$ | expected stars from one hit at level $L$ |
| $h$ | cost of a hit |
| $C_L$, $U_L$ | cost and success probability of upgrading $L \to L+1$ |
| $C_{\lt k} = \sum_{i\lt k} C_i$ | total cost of upgrading from level 1 to level $k$ |
| $\Delta$ | budget grid step: gcd of all costs |
| $B$, $F_B$, $S(x)$ | budget, its distribution, its survival function $P(B \ge x)$ |
| $b$, $s$ | coins remaining, coins spent so far |
| $g_j$ | coins a $j$★ item pays: once when it drops (refund, `Er`), or every tick while held (income, `Ei`) |
| $\rho$, $T$, $t$ | total income rate, horizon, time |
| $V$, $\pi$ | value function, policy |

### Running example

The same numbers are used everywhere, so results can be compared across problems
([`common/model.py`](common/model.py)):

- $N = 7$ core levels, $M = 5$ star tiers, $h = 200$
- $C = (400, 600, 900, 1300, 1800, 2500)$
- $U = (0.70, 0.60, 0.50, 0.40, 0.30, 0.25)$
- $P(L, \cdot)$ shifts from mostly 1★ at level 1 to mostly 5★ at level 7, giving
  $r = (1.36, 1.78, 2.34, 3.00, 3.66, 4.22, 4.64)$

## Classifying the variants: six axes

Every variant gets a tag `I?-U?-B?-E?-O?-K?`. Each axis answers one question. On every axis
the first value is a special case of the others (e.g. a sure upgrade is a random one with
$U = 1$); it gets its own value because it unlocks simpler methods.

| Axis | Question | Values |
|---|---|---|
| **I** | Is the item outcome random? | `Id` no (each hit gives exactly $r_L$) · `Ir` yes |
| **U** | Is the upgrade outcome random? | `Ud` no (always succeeds) · `Ur` yes |
| **B** | What do you know about the budget, and does it grow? | `Bk` known · `Bs` random, seen before play · `Bh` hidden (you find out when you run out) · `Bt` trickle (coins arrive over time from outside) |
| **E** | Do items feed back into coins? | `Ec` no (closed) · `Er` one-time refund per item · `Ei` items pay income (every item held pays coins each tick) · `Es` items can be sold (they then no longer count toward $u$) |
| **O** | Is the utility $u$ a plain sum over items? | `Oe` yes: total stars $u = \sum_k j_k$ · `Ou` no: any other $u(j_1, j_2, \dots)$; its shape sets the behaviour: concave (e.g. $\sqrt{\cdot}$ of total stars) → avoids risk · convex (e.g. $(\cdot)^2$) → seeks risk · a step at a goal (e.g. "at least one 5★") → either, depending on whether a safe plan reaches the goal |
| **K** | Do you know the probabilities? | `Kk` known · `Kb` known family, unknown parameters (Bayesian) · `Ku` unknown |

What additivity (`Oe`) buys: the state needs no memory of past items; item randomness can be
replaced by its mean $r_L$ (linearity of expectation); rewards arrive hit by hit (the
standard MDP form); and the closed forms and approximations of problems 01–03 work.
`Ou` pays with a bigger state (whatever summary of the items $u$ depends on) and loses
those shortcuts.

Two notes on the **K** axis:

- **A parametric distribution vs. a given table is not an axis.** Both are `Kk`. The
  solver only ever sees the table. The parametric form still matters for closed-form
  insights (an exponential budget is memoryless, so the policy is stationary) and for how
  fast you can learn (`Kb`).
- **Known vs. unknown is an axis.** Not knowing the odds adds a new trade-off: explore
  (spend to learn) vs. exploit.

### Mechanics: rule tweaks, not axes

These change the rules but not the kind of problem. They are solved with the same DP, with
an extra outcome or one extra state variable:

| Mechanic | Rule | What it changes |
|---|---|---|
| **demotion** | a failed upgrade drops the core a level with probability $D_L$ | 3 outcomes per attempt; coins still only go down, so the same backward induction works |
| **pity** | success chance rises with the number of failures $f$, reset on success | odds depend on history → add $f$ to the state |
| **hit cap** | at most a fixed number of hits per core level | add "hits used at this level" to the state |
| **time-varying odds** | odds change on a schedule (events) | add time $t$ to the state |

### Reductions: most tags collapse

There are $2 \cdot 2 \cdot 4 \cdot 4 \cdot 2 \cdot 3 = 384$ tags, but most reduce to
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
| [02](problems/02_Ir-Ud-Bk-Ec-Oe-Kk/) | `Ir-Ud-Bk-Ec-Oe-Kk` | item quality random | reduction (linearity of expectation) | ✅ |
| [03](problems/03_Ir-Ur-Bk-Ec-Oe-Kk/) | `Ir-Ur-Bk-Ec-Oe-Kk` | upgrades can fail (+ demotion, pity mechanics) | MDP / backward induction, optimal stopping, fluid approximation | ✅ |
| [04](problems/04_Ir-Ur-Bk-Ec-Ou-Kk/) | `Ir-Ur-Bk-Ec-Ou-Kk` | other utilities: risk attitude, goals | state augmentation, risk-sensitive DP | ✅ |
| [05](problems/05_Ir-Ur-Bs-Ec-Oe-Kk/) | `Ir-Ur-Bs-Ec-Oe-Kk` | budget random, seen first | reduction (average) | ✅ |
| [06](problems/06_Ir-Ur-Bh-Ec-Oe-Kk/) | `Ir-Ur-Bh-Ec-Oe-Kk` | budget hidden | belief MDP (POMDP → MDP), survival analysis, value of information | ✅ |
| [07](problems/07_Ir-Ur-Bk-Er-Oe-Kk/) | `Ir-Ur-Bk-Er-Oe-Kk` | items refund coins (coins can go up) | value iteration | ✅ |
| [08](problems/08_Ir-Ur-Bt-Ec-Oe-Kk/) | `Ir-Ur-Bt-Ec-Oe-Kk` | outside income over time | time-indexed DP | ✅ |
| [09](problems/09_Ir-Ur-Bk-Ei-Oe-Kk/) | `Ir-Ur-Bk-Ei-Oe-Kk` | items pay income | time-indexed DP on a bigger state, shadow prices | ✅ |
| [10](problems/10_Ir-Ur-Bk-Es-Oe-Kk/) | `Ir-Ur-Bk-Es-Oe-Kk` | items can be sold | inventory DP | ✅ |
| [11](problems/11_Ir-Ur-Bk-Ec-Oe-Kb/) | `Ir-Ur-Bk-Ec-Oe-Kb` | odds unknown, family known | Bayes-adaptive MDP, Thompson sampling | ✅ |
| [12](problems/12_Ir-Ur-Bk-Ec-Oe-Ku/) | `Ir-Ur-Bk-Ec-Oe-Ku` | nothing known | model-free RL (Q-learning) vs model-based learning | ✅ |
| 13 | `Ir-Ur-Bh-Ei-Oe-Kb` | everything at once | policy search, approximate DP | ⬜ |

### 01 · Fully deterministic: `Id-Ud-Bk-Ec-Oe-Kk`

**What changed:** nothing; this is the baseline. Every hit gives exactly $r_L$, every
upgrade succeeds.

**Formulation:** state $(b, L)$; actions hit, upgrade, stop.

```math
V(b,L) = \max\{\, r_L + V(b-h, L),\; V(b-C_L, L+1),\; 0 \,\}
```

**Tools:** [deterministic DP](problems/01_Id-Ud-Bk-Ec-Oe-Kk/deterministic_dp/) (longest
path in a DAG; works for any rules, e.g. under the hit-cap mechanic);
[enumeration](problems/01_Id-Ud-Bk-Ec-Oe-Kk/enumeration/) (try every target level $k$;
works in the special case below).

**Special case:** if $r_L$ increases with $L$, any upgrades you make should come
*before* any hits. The only decision left is how far to upgrade.

**On the running example:** richer means investing more. Below about 1,700 coins, don't
upgrade at all. The best target level is 3 at $B$ = 5,000, 6 at 20,000, and 7 from about
32,600 up. Value at $B$ = 20,000: **316.5**.

### 02 · Random items: `Ir-Ud-Bk-Ec-Oe-Kk`

**What changed:** a hit gives a random item, a $j$★ item with probability $P(L, j)$,
instead of exactly $r_L$ stars.

**Formulation:** same state $(b, L)$; the hit term averages over the item,
$\sum_j P(L,j) \cdot [ j + V(b-h, L) ]$, which equals 01's $r_L + V(b-h, L)$.

**Why nothing new is needed:** the utility is a plain sum (linearity of expectation), and
the item doesn't change the state, so there's nothing to react to. 01's plan stays optimal.

**Tool:** none new, a reduction to 01, checked by
[`verify.py`](problems/02_Ir-Ud-Bk-Ec-Oe-Kk/verify.py).

**On the running example:** same values as 01 (simulated means agree within their 95%
intervals). Randomness only adds spread: sd 3.5 stars at 5,000 coins, 6.1 at 20,000.

### 03 · Random upgrades: `Ir-Ur-Bk-Ec-Oe-Kk`

**What changed:** an upgrade from $L$ succeeds only with probability $U_L$; you pay $C_L$
either way.

**Formulation:** state $(b, L)$; the upgrade term averages over the outcome:

```math
V(b, L) = \max\Big\{\; r_L + V(b-h,\, L),\;\; U_L\, V(b - C_L,\, L+1) + (1 - U_L)\, V(b - C_L,\, L),\;\; 0 \;\Big\}
```

The upgrade outcome changes the state, so the answer is a **policy** that reacts, not a
fixed plan.

**Tools:** [backward induction](problems/03_Ir-Ur-Bk-Ec-Oe-Kk/backward_induction/) (exact MDP
solution; the budget only goes down, so fill the table from small budgets up);
[optimal stopping](problems/03_Ir-Ur-Bk-Ec-Oe-Kk/optimal_stopping/) (the policy is one
threshold per level); [fluid approximation](problems/03_Ir-Ur-Bk-Ec-Oe-Kk/fluid_approximation/)
(expected costs $C_L/U_L$ plugged into 01's tools);
[mechanics variants](problems/03_Ir-Ur-Bk-Ec-Oe-Kk/mechanics_variants/) (demotion, pity).

**On the running example:** failing upgrades cut the value by 13% at 5,000 coins and 21% at
20,000 compared with sure upgrades. Reacting to luck recovers little: the best fixed plan is
within 2.4% of the optimal policy at every budget up to 100,000. The policy is "upgrade
from level $L$ while coins ≥ $b^*_L$", with thresholds 2,400 · 3,800 · 7,900 · 16,900 ·
41,200 · 104,700. On a stress example built for it (one 10% upgrade from 1★ to 5★
hits, `LONG_SHOT`), adapting gains up to 15% and mean-value re-planning loses up to 52%.

### 04 · Other utilities: `Ir-Ur-Bk-Ec-Ou-Kk`

**What changed:** the utility $u$ is no longer total stars: e.g. $\sqrt{W}$ (avoids risk),
$W^2$ (seeks risk), or a goal ($W \ge G$, at least one 5★), with $W$ the total stars.

**Formulation:** $u$ is applied once at the end, so the state carries the summary $s$ of
the items that $u$ reads (total stars, a capped counter, a yes/no flag):

```math
V(b, L, s) = \max\Big\{ \sum_j P(L,j)\, V(b-h, L, s \oplus j),\;\; U_L\, V(b - C_L, L+1, s) + (1 - U_L)\, V(b - C_L, L, s),\;\; u(s) \Big\}
```

**Tools:** [state augmentation](problems/04_Ir-Ur-Bk-Ec-Ou-Kk/state_augmentation/) (any $u$;
bigger state); [risk-sensitive DP](problems/04_Ir-Ur-Bk-Ec-Ou-Kk/risk_sensitive/)
(exponential utility: a risk dial $\theta$ with no extra state).

**On the running example:** with $\sqrt{W}$ and $W^2$ the policies barely change (same mean
within 0.3 stars at 20,000; sd 31.1 and 37.0 vs 32.8): over this range both are close to
straight lines. Goals change a lot: aiming for at least one 5★ raises its chance from 0.318
to 0.782 at 10,000 coins, at a cost of about 31 expected stars.

### 05 · Random budget, seen first: `Ir-Ur-Bs-Ec-Oe-Kk`

**What changed:** the budget is drawn from a known distribution $p(B)$ and shown to you
before the first move.

**Formulation:** once you see $B$ it's problem 03, so the value is $\sum_B p(B) V(B)$.
03's table holds $V(b, L)$ for every $b$, so one table plays every budget.

**Tool:** none new, a reduction to 03, checked by
[`verify.py`](problems/05_Ir-Ur-Bs-Ec-Oe-Kk/verify.py).

**On the running example:** with a mean budget of 10,000, a random budget is worth more
than a fixed one: 108.96 (uniform 0 to 20,000) and 118.35 (exponential) vs 100.25. $V$ is
convex (stars per coin rise with the budget), so Jensen gives $E[V(B)] \ge V(E[B])$. Most of
the outcome's spread then comes from which budget you got, not from luck during play.

### 06 · Hidden budget: `Ir-Ur-Bh-Ec-Oe-Kk`

**What changed:** the budget is drawn from a known $p(B)$ but never shown. You see what
you've spent, $s$. An action you can't afford is refused and ends the game; leftover coins
are lost.

**Formulation:** the only news about $B$ is "not broke yet", so the belief is the prior cut
off below $s$, fixed by $s$ alone. The hidden-state problem (POMDP) becomes an MDP on
$(s, L)$; each action goes through with probability $S(s+c)/S(s)$, where $S(x) = P(B \ge x)$:

```math
V(s, L) = \max\Big\{ \tfrac{S(s+h)}{S(s)} \big( r_L + V(s+h, L) \big),\;\; \tfrac{S(s+C_L)}{S(s)} \big( U_L\, V(s+C_L, L+1) + (1-U_L)\, V(s+C_L, L) \big),\;\; 0 \Big\}
```

**Tools:** [belief MDP](problems/06_Ir-Ur-Bh-Ec-Oe-Kk/belief_mdp/) (exact; same family as
the Kalman filter); [survival analysis](problems/06_Ir-Ur-Bh-Ec-Oe-Kk/survival_analysis/)
(hazard, mean residual budget $m(s)$, the heuristic "play 03 as if $m(s)$ coins were
left"); value of information (05's value minus 06's).

**On the running example:** with mean 10,000, not seeing $B$ costs 4.4 (2,000 or 18,000),
5.0 (uniform) and 10.6 stars (exponential), 4% to 9%. With the exponential prior
(memoryless) the policy ignores $s$: upgrade to level 4, then hit. Plugging in $m(s)$ is
within 0.02 stars of optimal for the uniform and exponential priors, and loses 2.5 on the
two-point one.

### 07 · Items refund coins: `Ir-Ur-Bk-Er-Oe-Kk`

**What changed:** a $j$★ item refunds $g_j$ coins once, when it drops. In the running
example a 5★ item refunds 250, more than a hit costs, so coins can go up.

**Formulation:** same as 03 except the hit term:

```math
V(b, L) = \max\Big\{ r_L + \sum_j P(L,j)\, V(b - h + g_j,\, L),\;\; U_L\, V(b - C_L,\, L+1) + (1 - U_L)\, V(b - C_L,\, L),\;\; 0 \Big\}
```

$(b, L)$ can now repeat, so there's no order for backward induction.

**Tool:** [value iteration](problems/07_Ir-Ur-Bk-Er-Oe-Kk/value_iteration/): start from
$V = 0$ and apply the update until nothing changes; after $k$ sweeps $V$ is the best value
with at most $k$ actions. Plus the mean-refund approximation (hit cost $h - E[g]$, back to
03's tools).

**On the running example:** at 5,000 coins nothing changes; at 20,000 the value rises
from 250.35 to 433.16, and the policy goes for level 7 once it's at level 4 with about
9,500 coins. The mean hides a lottery: median 195 stars, 90th percentile 1,177. The
mean-refund approximation gets the value within 0.8%.

### 08 · Coins trickle in: `Ir-Ur-Bt-Ec-Oe-Kk`

**What changed:** $\rho$ coins arrive each tick for $T$ ticks, and you take one action
per tick (hit, upgrade, or wait).

**Formulation:** add time to the state. With $b$ the coins before this tick's income:

```math
V(t, b, L) = \max\Big\{ V(t+1,\, b+\rho,\, L),\;\; r_L + V(t+1,\, b+\rho-h,\, L),\;\; U_L\, V(t+1,\, b+\rho-C_L,\, L+1) + (1-U_L)\, V(t+1,\, b+\rho-C_L,\, L) \Big\}
```

**Tool:** [time-indexed DP](problems/08_Ir-Ur-Bt-Ec-Oe-Kk/time_indexed_dp/): coins go up,
but $t$ only goes forward, so one backward pass in $t$ is exact. Two bounds frame it:
03 with all $\rho T$ coins up front, and the same ticks with coins never short.

**On the running example:** if $\rho \le h$ the clock doesn't matter at all: 08 is exactly
03 with $\rho T$ coins (a short proof in the method page). Above that, ticks bind: with
$T$ = 50 and $\rho$ = 400 (the same 20,000 coins) you get 138.90 stars instead of 250.35.
With coins never short, the policy is a threshold rule in ticks left: 6 · 7 · 9 · 13 · 23
· 42.

### 09 · Items pay income: `Ir-Ur-Bk-Ei-Oe-Kk`

**What changed:** every item you hold pays $g_j$ coins per tick (08's clock: one action
per tick). Items are capital: their stars count at the end, and they earn until then.

**Formulation:** the state must say what you earn. On the running example only 5★ items
pay ($g_5$ each), so one counter $n$ (5★ items held) is enough: state $(t, b, n, L)$, and
a hit adds an earner with probability $P(L,5)$.

**Tools:** [time-indexed DP](problems/09_Ir-Ur-Bk-Ei-Oe-Kk/time_indexed_dp/) (08's method
on the bigger state); shadow prices (what one more coin or earner is worth, by
differencing $V$; the costates of optimal control).

**On the running example:** $B_0$ = 10,000, $T$ = 50. With $g_5$ = 100 the value rises
from 100.25 to 130.29, and the policy now climbs to level 5 first (81% of games vs
none), where 13% of hits make an earner. Doubling $g_5$ to 200 adds only 3.5: once
income covers a hit every tick, the clock binds. An extra earner or extra coins are
worth exactly 0 in the last ticks, once the coins held already pay for every remaining
tick.

### 10 · Items can be sold: `Ir-Ur-Bk-Es-Oe-Kk`

**What changed:** items can be sold for coins, and a sold item no longer counts. On the
running example only 1★ items have a buyer, at $s_1 \lt h$ coins.

**Formulation:** state $(b, n, L)$ with $n$ the unsold 1★ items; a new action "sell one"
gives $s_1$ coins and costs 1 star. Coins can go up, but liquid wealth $b + s_1 n$ never
does, so one pass in that order is exact.

**Tool:** [inventory DP](problems/10_Ir-Ur-Bk-Es-Oe-Kk/inventory_dp/); compared with a
one-step exchange rule (sell iff $s_1 r_L / h \gt 1$).

**On the running example:** selling barely helps (at most 2.5 stars): a sale undoes a past
hit at a loss, and good play makes few 1★ items. On a stress example built for it
(start with 50 1★ items and 2,000 coins, $s_1$ = 150), selling raises the value from 63.60
to 96.40; the exact policy sells as it goes, keeping a reserve. The one-step exchange
rule can miss that coins fund upgrades: at $s_1$ = 100 and 100 starting items it gets
113.60 vs 130.00.

### 11 · Odds unknown, family known: `Ir-Ur-Bk-Ec-Oe-Kb`

**What changed:** the upgrade odds $U_L$ are unknown, with a known Beta prior (mean = the
running example's $U_L$, concentration $\kappa$). You learn from your own tries.

**Formulation:** you never return to a level, so the belief that matters is the current
level's, carried by $f$, the failures there so far. The chance the next try works is the
posterior mean $m_L(f) = a_L / (a_L + b_L + f)$, and the problem is an MDP on $(b, L, f)$:

```math
V(b, L, f) = \max\Big\{ r_L + V(b-h, L, f),\;\; m_L(f)\, V(b - C_L, L+1, 0) + \big(1 - m_L(f)\big)\, V(b - C_L, L, f+1),\;\; 0 \Big\}
```

**Tools:** [Bayes-adaptive MDP](problems/11_Ir-Ur-Bk-Ec-Oe-Kb/bayesian_adaptive_mdp/) (exact;
the SLAM / dual-control parallel); [Thompson sampling](problems/11_Ir-Ur-Bk-Ec-Oe-Kb/thompson_sampling/).

**On the running example:** learning from failures is worth 3.7% at $\kappa$ = 2 (20,000
coins) and 19% on `LONG_SHOT` ($\kappa$ = 1); the Bayes policy gives up after fewer failures
the vaguer the prior. Playing 03 with the current posterior mean is within 0.02 stars of
the optimum everywhere tried. Thompson sampling loses 3–8% with a vague prior: it keeps
switching between giving up and trying again.

### 12 · Nothing known: `Ir-Ur-Bk-Ec-Oe-Ku`

**What changed:** no odds and no prior. You play the same game many times and learn from
experience; the question is how many games until you play well.

**Formulation:** 03's optimum, written per action, $Q(b, L, a)$; each move you make is a
sample of its right-hand side.

**Tools:** [Q-learning](problems/12_Ir-Ur-Bk-Ec-Oe-Ku/model_free_rl/) (model-free: nudges
$Q$ towards each sample), against a model-based learner (count outcomes, estimate $P$ and
$U$, re-solve 03).

**On the running example:** the model-based learner is optimal (to 0.1 stars) after 300
games; Q-learning is still 3–10% short after 100,000 games, and on `LONG_SHOT` it has
learned only part of the gain from upgrading. The model-based learner pools its data
(every hit at level $L$ teaches it about $P(L, \cdot)$ at every budget); Q-learning learns
each state on its own.

### 13

*Planned.*

## Results so far

Each problem reports its value (best expected total stars) on the running example:

| # | $B$ = 5,000 | $B$ = 20,000 | Note |
|---|---|---|---|
| 01 | 46.80 | 316.50 | upgrades always succeed (an upper bound for 03) |
| 02 | 46.80 | 316.50 | same as 01; random items only add spread (sd 3.5 / 6.1) |
| 03 | 40.64 | 250.35 | upgrades can fail; best fixed plan: 40.14 / 248.43 |
| 04 | 40.63 | 250.28 | expected stars of the policy optimal for $\sqrt{W}$; sd 7.2 / 31.1 vs 03's 7.3 / 32.8 |
| 05 | — | — | budget random with mean 10,000: 108.96 (uniform 0 to 20,000), 118.35 (exponential) vs 100.25 fixed |
| 06 | — | — | same budgets, hidden: 104.00 (uniform), 107.76 (exponential) |
| 07 | 40.64 | 433.16 | 5★ items refund 250 coins; median at 20,000 is 195 |
| 08 | — | — | 20,000 coins as income over 50 ticks (400 per tick), one action per tick: 138.90 |
| 09 | — | — | 10,000 coins, 50 ticks, each 5★ item pays 100 per tick: 130.29 (100.25 without income) |
| 10 | 42.73 | 251.34 | 1★ items sell for 150 coins (03: 40.64 / 250.35) |
| 11 | 39.54 | 227.91 | upgrade odds unknown, Beta prior with $\kappa$ = 2, averaged over the prior |
| 12 | — | — | at 10,000 coins, after 100,000 games: Q-learning 96.56, model-based 100.25 (optimal) |

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
