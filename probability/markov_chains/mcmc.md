# Monte Carlo Markov Chain (MCMC)

## The Problem

We have a target distribution $\pi(x)$ that we want to sample from, but:
- We can't sample from it directly (no closed-form CDF inverse, too high-dimensional, etc.)
- We may only know $\pi(x)$ up to a normalizing constant: $\pi(x) = \frac{\tilde{\pi}(x)}{Z}$ where $Z = \int \tilde{\pi}(x) dx$ is intractable

This is extremely common in Bayesian inference where the posterior is:

$$
P(\theta \mid \text{data}) = \frac{P(\text{data} \mid \theta) P(\theta)}{P(\text{data})}
$$

The denominator $P(\text{data}) = \int P(\text{data} \mid \theta) P(\theta) d\theta$ is often intractable.

## The Idea

Construct a Markov chain whose **stationary distribution** is exactly $\pi(x)$. Run the chain long enough, and the samples it produces will be (approximately) drawn from $\pi$.

Recall from basic Markov chains: a stationary distribution $\pi$ satisfies $\pi = \pi P$. MCMC reverse-engineers a transition kernel $P$ such that this holds for our target $\pi$.

## Detailed Balance — The Design Constraint

So how do we reverse-engineer $P$? We need a tractable sufficient condition that guarantees $\pi = \pi P$. That condition is **detailed balance**:

$$
\pi(x) P(x \to y) = \pi(y) P(y \to x)
$$

This says: the probability flow from $x$ to $y$ under $\pi$ equals the flow from $y$ to $x$. Summing both sides over $x$ gives $\pi = \pi P$, so detailed balance implies stationarity (but is stronger — it also implies the chain is reversible).

Detailed balance is just the constraint. We still need to actually build a $P$ that satisfies it. That's the next section.

### Explanation

Probability of being in state $x$ is $\pi(x)$, and probability of moving to $y$ given we are in $x$ is $P(x,y)$

$Flow(x \to y) = \pi(x)P(x \to y)$, which is the amount of probability mass moving from $x$ to $y$.

In a stationary distribution, all we are saying is $Flow(x \to y). = Flow(y \to x)$

Why does summing over flows give $\pi = \pi P$?


$$
\sum_x \pi(x) P(x,y)
=
\sum_x \pi(y) P(y,x)
=
\pi(y) \sum_x P(y,x)
=
\pi(y)
$$

since

$$
\sum_x P(y,x) = 1.
$$

Therefore,

$$
\sum_x \pi(x) P(x,y) = \pi(y)
\quad \Longrightarrow \quad
\pi = \pi P.
$$

## Metropolis-Hastings — Building the Chain

**This is where we actually construct the reverse-engineered transition kernel $P$.**

We don't pick $P$ directly (that would be hard to get right). Instead we decompose it into two pieces we *can* control: a **proposal** $q$ (easy to sample from, our free choice) and an **acceptance correction** $\alpha$ (derived from $\pi$ and $q$ to force detailed balance). Together they define $P$:

### Setup

1. **Choose** a proposal distribution $q(y \mid x)$ — how we suggest the next state given current state. This is our free choice (e.g., a Gaussian centered at $x$). It does NOT need to relate to $\pi$ at all.
2. **Derive** an acceptance probability $\alpha(x, y)$ that corrects $q$ so that the combined transition satisfies detailed balance w.r.t. $\pi$.

### The Constructed Transition Kernel

Given current state $x$, proposed state $y$:

$$
\alpha(x, y) = \min\left(1, \frac{\pi(y)\, q(x \mid y)}{\pi(x)\, q(y \mid x)}\right)
$$

The overall transition: propose $y \sim q(\cdot \mid x)$, then accept (move to $y$) with probability $\alpha(x, y)$, otherwise stay at $x$.

**This defines our Markov chain.** The effective transition kernel is:

$$
P(x \to y) = q(y \mid x)\, \alpha(x, y) \quad \text{for } y \neq x
$$

This $P$ is the "reverse-engineered" transition matrix from The Idea section. We chose $q$ freely, then $\alpha$ was determined by the requirement that $P$ satisfies detailed balance w.r.t. $\pi$.

### Why This Works (Verifying Detailed Balance)

We claimed $\alpha$ was chosen to make detailed balance hold. Let's verify. The effective transition is $P(x \to y) = q(y \mid x)\, \alpha(x, y)$ for $y \neq x$. Check detailed balance:

$$
\pi(x)\, q(y \mid x)\, \alpha(x, y) = \pi(x)\, q(y \mid x) \min\left(1, \frac{\pi(y) q(x \mid y)}{\pi(x) q(y \mid x)}\right) = \min\left(\pi(x) q(y \mid x),\; \pi(y) q(x \mid y)\right)
$$

Now check the reverse direction — swap $x \leftrightarrow y$:

$$
\pi(y)\, q(x \mid y)\, \alpha(y, x) = \min\left(\pi(y) q(x \mid y),\; \pi(x) q(y \mid x)\right)
$$

These are the same two arguments inside the $\min$, just in opposite order. Since $\min(a, b) = \min(b, a)$, the two sides are equal. Detailed balance holds.

### Using the Chain to Sample

Now that we have the chain, we **run it**: start at some arbitrary $x_0$, repeatedly propose and accept/reject to get $x_1, x_2, \ldots, x_T$. After enough steps (burn-in), these $x_t$ are approximate samples from $\pi$. That's the "Monte Carlo" part — we use these samples to estimate expectations like $E_\pi[f(x)] \approx \frac{1}{T-B}\sum_{t=B+1}^{T} f(x_t)$.

### Key Insight: Normalizing Constants Cancel

In the acceptance ratio, $\pi$ always appears as a ratio $\frac{\pi(y)}{\pi(x)}$. If $\pi(x) = \tilde{\pi}(x)/Z$:

$$
\frac{\pi(y)}{\pi(x)} = \frac{\tilde{\pi}(y)/Z}{\tilde{\pi}(x)/Z} = \frac{\tilde{\pi}(y)}{\tilde{\pi}(x)}
$$

We never need to compute $Z$. This is why MCMC is so powerful for Bayesian inference.

## Special Case: Metropolis Algorithm

When the proposal is symmetric, i.e. $q(y \mid x) = q(x \mid y)$ (e.g., a Gaussian centered at the current state), the acceptance ratio simplifies to:

$$
\alpha(x, y) = \min\left(1, \frac{\pi(y)}{\pi(x)}\right)
$$

Intuition: always accept moves to higher-probability regions; accept moves to lower-probability regions proportionally to how much lower they are.

## Gibbs Sampling

A special case of Metropolis-Hastings for multivariate distributions. Instead of proposing a move in the full space, update one coordinate at a time from its **full conditional**.

For target $\pi(x_1, x_2, \ldots, x_d)$, iterate:

$$
x_1^{(t+1)} \sim \pi(x_1 \mid x_2^{(t)}, x_3^{(t)}, \ldots, x_d^{(t)}) \\
x_2^{(t+1)} \sim \pi(x_2 \mid x_1^{(t+1)}, x_3^{(t)}, \ldots, x_d^{(t)}) \\
\vdots \\
x_d^{(t+1)} \sim \pi(x_d \mid x_1^{(t+1)}, x_2^{(t+1)}, \ldots, x_{d-1}^{(t+1)})
$$

Each step is a Metropolis-Hastings move with acceptance probability 1 (the proposal is the conditional itself, so the ratio always works out). This is useful when conditionals are easy to sample from even though the joint is not.

### Why does this work despite only moving along one axis at a time?

It feels like Gibbs is "restricted" — at each step we only change one coordinate, moving along axis-aligned directions while the target distribution might have structure in every direction. Why doesn't this get stuck?

The key insight: **each conditional step preserves the target distribution.** When we sample $x_1 \sim \pi(x_1 \mid x_2, \ldots, x_d)$, we are drawing from the *exact* conditional of $\pi$. This means if the current state $(x_1, x_2, \ldots, x_d)$ is distributed according to $\pi$, then after resampling $x_1$ the new state is *still* distributed according to $\pi$. Each coordinate update leaves $\pi$ invariant.

Think of it as slicing. At a given point $(x_1, x_2)$, the conditional $\pi(x_1 \mid x_2)$ is a 1D slice through the joint density at the current $x_2$ value. Sampling from that slice places us at the correct relative density along the $x_1$ axis. Then we slice the other way. Alternating slices explores the full space — each slice is exact (no accept/reject needed), so no probability mass is wasted.

The restriction is real though: Gibbs can only take axis-aligned steps, so it explores diagonal structure slowly. If $x_1$ and $x_2$ are highly correlated, the "slices" are narrow and nearly parallel to each other, so the chain zigzags in small steps along the diagonal. This is why Gibbs mixes slowly when coordinates are correlated (as shown in the bivariate Normal example below).

## Examples

### Metropolis-Hastings: Sampling from a Gamma Distribution

Suppose we want to sample from $\pi(x) = \text{Gamma}(a, 1)$ with density $\pi(x) \propto x^{a-1} e^{-x}$ for $x > 0$, and we don't have a built-in Gamma sampler.

**Choose a proposal:** Use a log-normal random walk. If the current state is $x$, propose $y = x \cdot e^{z}$ where $z \sim N(0, \sigma^2)$.

**Deriving $q(y \mid x)$:** We defined $y = x \cdot e^z$ with $z \sim N(0, \sigma^2)$, so $z = \ln(y/x)$. The density of $z$ is the normal density $p(z) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{z^2}{2\sigma^2}\right)$. To get the density of $y$, apply the change of variables formula:

$$
q(y \mid x) = p(z) \cdot \left|\frac{dz}{dy}\right| = p(\ln(y/x)) \cdot \frac{1}{y}
$$

since $z = \ln(y/x)$ implies $\frac{dz}{dy} = \frac{1}{y}$. Substituting:

$$
q(y \mid x) = \frac{1}{y \sigma \sqrt{2\pi}} \exp\left(-\frac{(\ln y/x)^2}{2\sigma^2}\right)
$$

This proposal is **not** symmetric ($q(y \mid x) \neq q(x \mid y)$) because of the $1/y$ factor — swapping $x \leftrightarrow y$ changes the $1/y$ to $1/x$ but the Gaussian part stays the same (since $(\ln y/x)^2 = (\ln x/y)^2$). So we need the full Metropolis-Hastings ratio, not just Metropolis.

**Acceptance ratio:**

$$
\alpha(x, y) = \min\left(1, \frac{\pi(y) q(x \mid y)}{\pi(x) q(y \mid x)}\right)
$$

Compute each piece:

$$
\frac{\pi(y)}{\pi(x)} = \frac{y^{a-1} e^{-y}}{x^{a-1} e^{-x}} = \left(\frac{y}{x}\right)^{a-1} e^{-(y - x)}
$$

$$
\frac{q(x \mid y)}{q(y \mid x)} = \frac{1/x}{1/y} = \frac{y}{x}
$$

(The Gaussian parts cancel since $(\ln y/x)^2 = (\ln x/y)^2$.)

So:

$$
\alpha(x, y) = \min\left(1, \left(\frac{y}{x}\right)^{a-1} e^{-(y - x)} \cdot \frac{y}{x}\right) = \min\left(1, \left(\frac{y}{x}\right)^{a} e^{-(y - x)}\right)
$$

**The algorithm:**

1. Start at some $x_0 > 0$
2. At step $t$: propose $y = x_t \cdot e^z$, $z \sim N(0, \sigma^2)$
3. Compute $\alpha(x_t, y)$
4. Draw $u \sim \text{Uniform}(0,1)$. If $u < \alpha$, set $x_{t+1} = y$; else $x_{t+1} = x_t$
5. After burn-in, the $x_t$ are approximate samples from $\text{Gamma}(a, 1)$

Note: the $q(x \mid y) / q(y \mid x) = y/x$ correction matters here. Without it (i.e., using plain Metropolis), the samples would be biased — the chain would over-visit small $x$ values because the proposal is more likely to jump to small values than away from them.

### Gibbs Sampling: Bivariate Normal

Suppose we want to sample from a bivariate Normal:

$$
(x_1, x_2) \sim N\left(\begin{pmatrix}0\\0\end{pmatrix}, \begin{pmatrix}1 & \rho\\\rho & 1\end{pmatrix}\right)
$$

The joint density is messy to sample from directly in general, but the **full conditionals** are trivial:

$$
x_1 \mid x_2 \sim N(\rho\, x_2,\; 1 - \rho^2) \\
x_2 \mid x_1 \sim N(\rho\, x_1,\; 1 - \rho^2)
$$

Each conditional is just a univariate Normal — easy to sample from.

**The algorithm:**

1. Start at some $(x_1^{(0)}, x_2^{(0)})$
2. At step $t$:
   - Draw $x_1^{(t+1)} \sim N(\rho\, x_2^{(t)},\; 1 - \rho^2)$
   - Draw $x_2^{(t+1)} \sim N(\rho\, x_1^{(t+1)},\; 1 - \rho^2)$
3. After burn-in, the pairs $(x_1^{(t)}, x_2^{(t)})$ are approximate samples from the joint

**Mixing and $\rho$:** When $\rho \approx 0$, the two variables are nearly independent, the conditionals are wide, and the chain mixes fast. When $|\rho| \to 1$, the conditionals become narrow — each variable is tightly constrained by the other — so the chain takes small steps along the diagonal and mixes slowly. This illustrates the general weakness of Gibbs: it updates one coordinate at a time, so it struggles when coordinates are highly correlated.

## Practical Considerations

### Burn-in

The chain starts from an arbitrary state $x_0$ and needs time to "forget" the initialization and converge to $\pi$. Discard the first $B$ samples (the burn-in period).

### Mixing

Samples from the chain are correlated (each state depends on the previous). Good **mixing** means the chain explores the state space efficiently — low autocorrelation between samples.

Poor mixing happens when:
- The proposal step size is too small → chain takes tiny steps, explores slowly (high acceptance rate, high autocorrelation)
- The proposal step size is too large → most proposals land in low-probability regions and get rejected (low acceptance rate, chain gets stuck)
- The target has multiple well-separated modes → chain gets trapped in one mode

### Thinning

To reduce autocorrelation, keep only every $k$-th sample. E.g., with thinning factor 10, keep samples at iterations 10, 20, 30, ...

### Diagnostics

- **Trace plots**: plot $x^{(t)}$ vs $t$. A well-mixing chain looks like white noise around the target mean ("hairy caterpillar"). A poorly-mixing chain shows visible trends or long stays in one region.
- **Multiple chains**: run several chains from different starting points. If they converge to the same distribution, that's evidence of convergence. The Gelman-Rubin diagnostic ($\hat{R}$) formalizes this — $\hat{R} \approx 1$ indicates convergence.
- **Effective sample size (ESS)**: the number of independent samples equivalent to your correlated chain. ESS $\ll$ actual sample count means high autocorrelation.

## Summary

| Concept | Role |
|---|---|
| Target $\pi$ | The distribution we want samples from |
| Proposal $q$ | How we suggest moves (our design choice) |
| Acceptance $\alpha$ | Correction factor ensuring detailed balance |
| Burn-in | Discard early samples before convergence |
| Thinning | Reduce autocorrelation by subsampling |
| Detailed balance | The mathematical guarantee that $\pi$ is stationary |
