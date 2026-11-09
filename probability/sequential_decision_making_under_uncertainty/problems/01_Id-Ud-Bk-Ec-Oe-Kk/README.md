# 01 · `Id-Ud-Bk-Ec-Oe-Kk`: the fully deterministic game

| Axis | Value | Meaning here |
|---|---|---|
| I | `Id` | every hit at level $L$ yields exactly $r_L$ (no luck) |
| U | `Ud` | every upgrade succeeds |
| B | `Bk` | budget $B$ known up front |
| E | `Ec` | closed economy: items don't produce stars |
| O | `Oe` | maximise total item value |
| K | `Kk` | all numbers known |

Nothing is random. This is the baseline every later problem is compared against.

## Formulation

- **State** $(b, L)$: stars left, core level.
- **Actions**
  - *hit*: pay $h$, gain $r_L$, stay at $L$.
  - *upgrade*: pay $C_L$, gain nothing now, move to $L+1$.
  - *stop*.
- **Objective**: maximise total gain.

$$
V(b, L) = \max\Big\{\, r_L + V(b-h,\, L),\;\; V(b - C_L,\, L+1),\;\; 0 \,\Big\}
$$

(an action is only allowed if affordable). The answer is $V(B, 1)$.

## The structure that makes it easy

If $r_L$ increases with $L$, **never hit before an upgrade you're going to make anyway**:
moving that hit to after the upgrade costs the same and pays more. So every optimal plan
looks like

> **upgrade straight to some target level $k$, then spend everything else on hits.**

The whole problem collapses to one number, $k$:

$$
V(B) = \max_k \; r_k \left\lfloor \frac{B - K_k}{h} \right\rfloor, \qquad K_k = \sum_{i<k} C_i
$$

That is the **invest vs. harvest** trade-off in its purest form: a bigger $k$ means
fewer hits, each worth more.

## Tools

| Method | When | Link |
|---|---|---|
| **Enumeration** over target level | the "upgrade first" structure holds | [enumeration/](enumeration/) |
| **Deterministic DP** (longest path in a DAG) | always; needed once the structure breaks | [deterministic_dp/](deterministic_dp/) |

## On the running example

Upgrades always succeed here, at the listed costs $C = (400, 600, 900, 1300, 1800, 2500)$, with $h = 200$.

| Budget $B$ | 1 000 | 5 000 | 10 000 | 20 000 | 50 000 | 100 000 |
|---|---|---|---|---|---|---|
| Optimal value | 6.80 | 46.80 | 124.44 | 316.50 | 983.68 | 2143.68 |
| Target level $k^*$ | 1 | 3 | 5 | 6 | 7 | 7 |

![value vs budget](enumeration/value_vs_budget.svg)

**Takeaways**

- **Richer → invest more.** The best target level rises in steps with the budget (panel b).
- The switch points are roughly where one plan's line overtakes the next. With
  continuous hits they are exact:
  $B^*_k = \dfrac{K_{k+1} r_{k+1} - K_k r_k}{r_{k+1} - r_k}$ (≈ 1.7k, 2.9k, 5.1k, 9.1k, 16.8k, 32.6k).
- Near a switch point the exact answer flickers between two levels. That comes from the
  $\lfloor\cdot\rfloor$: a leftover of less than one hit can tip the choice. It's a small
  effect in value terms, but it's why integer problems rarely have clean closed forms.

## What's next

[02](../02_Ir-Ud-Bk-Ec-Oe-Kk/): make the **item quality random** ($I$: `Id → Ir`).
Surprisingly, nothing changes.
