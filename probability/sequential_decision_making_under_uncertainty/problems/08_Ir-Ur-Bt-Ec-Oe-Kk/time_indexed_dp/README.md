# Time-indexed DP

## Idea

When coins can go up, "fill the table from small $b$" stops working (07). If the game has
a clock, put time in the state: $t$ only goes forward, so $(t, b, L)$ has an order again.
Fill the table from the last tick back:

```math
V(t, b, L) = \max\Big\{ V(t+1,\, b+\rho,\, L),\;\; r_L + V(t+1,\, b+\rho-h,\, L),\;\; U_L\, V(t+1,\, b+\rho-C_L,\, L+1) + (1-U_L)\, V(t+1,\, b+\rho-C_L,\, L) \Big\}
```

with $V(T, \cdot, \cdot) = 0$; terms for actions you can't afford are dropped. One pass,
no iteration. The policy now depends on $t$ too: the same coins and level can call for
a different action with 5 ticks left than with 50.

## When the clock doesn't matter

**Claim.** With no starting coins and $\rho \le h$ (income per tick at most the cheapest
action), 08's value equals 03's value with $\rho T$ coins.

**Why.**

- *Not more:* any way of playing 08 spends at most $\rho T$ coins, so (skipping the
  waits) it's a legal way to play 03 with $\rho T$ coins.
- *Not less:* play 03's optimal policy, doing its next action as soon as you can afford
  it. If its first $i$ actions cost $c_1 + \dots + c_i$ in total, action $i$ happens at
  tick $\lceil (c_1 + \dots + c_i)/\rho \rceil$. Every action costs at least $h \ge \rho$,
  so these ticks are all different (one action per tick is enough), and the last is at
  most $\lceil \rho T / \rho \rceil = T$. 03's policy only needs the coins it has left,
  which you know.

So with slow income the clock never binds, and the extra state $t$ is wasted.

## When it applies

- Anything with a clock: a deadline, a season, income or prices that change over time.
- Cost: the table is about $T$ times bigger than 03's (here 200 ticks is still instant).

## On the running example

Starting coins 0. Checks:

- $\rho$ = 100 and 200, $T$ = 10, 50, 100, 200: equal to 03 with $\rho T$ coins to 4
  decimals (the claim above).
- Simulator (`simulate(..., income=ρ, ticks=T)`, 4,000 games, $T$ = 50): 100.73 ± 0.51
  vs 100.25 ($\rho$ = 200); 139.06 ± 0.51 vs 138.90 ($\rho$ = 400); 163.30 ± 0.56 vs
  163.44 ($\rho$ = 1,000).

**Tick bound** (income 2,500 per tick, so coins are never short): upgrade from level $L$
while ticks left ≥

| level | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| ticks left | 6 | 7 | 9 | 13 | 23 | 42 |

checked to be a clean threshold rule for $T$ up to 200.

**Both bounds vs the exact value**, $\rho$ = 400:

| $T$ | 08 exact | coin bound | tick bound | 08 / smaller bound |
|---|---|---|---|---|
| 10 | 15.71 | 30.52 | 16.24 | 0.968 |
| 50 | 138.90 | 250.35 | 164.97 | 0.842 |
| 100 | 338.24 | 609.99 | 394.73 | 0.857 |
| 200 | 785.51 | 1,420.75 | 858.73 | 0.915 |

The exact value sits well below both bounds: having enough coins in total and enough
ticks in total isn't the same as having the coins when you have the ticks.

## Code

[`solve.py`](solve.py)

- `solve(game, rho, T, B0=0) -> (V, A)`: $V[t, b, L]$ and actions (0 wait, 1 hit,
  2 upgrade), backward in $t$.
- `time_policy(A, unit, rho)`: plays the table in the simulator.
- `tick_thresholds(game, T)`: the time thresholds when coins are never short (asserts
  the threshold shape).
- `main()`: all numbers on this page and `time_indexed.svg`.

The simulator got a clock: `simulate(..., income=ρ, ticks=T)` adds $\rho$ coins each tick
and asks for one action ("hit", "up", "wait").

```
python solve.py
```
