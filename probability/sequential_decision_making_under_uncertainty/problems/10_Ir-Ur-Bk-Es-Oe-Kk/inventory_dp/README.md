# Inventory DP

## Idea

When what you hold matters later, put it in the state. Here: $n$, the unsold 1★ items,
each sellable for $s_1$ coins at any time.

```math
V(b, n, L) = \max\Big\{ -1 + V(b + s_1, n - 1, L),\;\; r_L + p_1\, V(b - h, n + 1, L) + (1 - p_1)\, V(b - h, n, L),\;\; U_L\, V(b - C_L, n, L+1) + (1 - U_L)\, V(b - C_L, n, L),\;\; 0 \Big\}
```

**Order.** Selling raises $b$, so "small $b$ first" fails. Use **liquid wealth**
$w = b + s_1 n$ instead:

- sell: $w$ stays the same, $n$ drops by 1;
- hit: $w$ drops by $h$, or by $h - s_1$ if the item is a 1★ (still a drop, since $s_1 \lt h$);
- upgrade: $w$ drops by $C_L$.

So fill the table by $w$ ascending, and within equal $w$ by $n$ ascending. Every lookup
is already done. One pass, like 03.

If $s_1 \ge h$ this breaks: hit, get a 1★, sell it, and $w$ doesn't drop. Then you'd need
07's value iteration.

## When it applies

- Inventory that can be turned into coins (or other resources) later.
- The state grows with every kind of item you track. One sellable tier keeps it to one
  counter.

## On the running example

**Checks:** with $s_1$ = 50 a sale only beats keeping from level 6, and the value matches
03 to within 0.08 at 5,000 to 20,000 coins. Simulator (2,000 games each) agrees with every exact value on
the [10 page](../), e.g. 42.97 ± 0.19 vs 42.73 ($s_1$ = 150, 5,000 coins) and 96.20 ± 0.42
vs 96.40 (start with 50 items, $s_1$ = 150).

**Exchange rule**, a one-step heuristic: sell a 1★ item iff the coins buy more stars at the
current level than the item holds, $s_1 r_L / h \gt 1$, and otherwise play 03's policy on
the coins in hand. It ignores that coins can pay for **upgrades**. It works when selling
is good at every level ($s_1$ = 150: within 1.3% of exact) and fails when it isn't good at low levels ($s_1$ = 100,
where it never funds the climb).

## Code

[`solve.py`](solve.py)

- `solve(game, s1, B_max, policy=None) -> (V, A, Δ)`: the table over $(b, n, L)$, in order
  of liquid wealth; with `policy` it evaluates that policy instead.
- `exchange_rule_policy(game, s1, A03)`: the heuristic.
- `play(...)`: a small simulator with selling (and a starting inventory).
- `main()`: all numbers on the 10 page, the `LONG_SHOT` check, `inventory.svg`.

```
python solve.py
```
