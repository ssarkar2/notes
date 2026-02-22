A concise set of notes, without much context/explanations, esp for elementary stuff

# Basics

$$
P(A \cup B) = P(A) + P(B) - P(A \cap B)
$$
Therefore, 
$$
P(A \cup B) \le P(A) + P(B)
$$

## Conditional probability

$$
P(A | B) = \frac{P(A \cap B)}{P(B)}
$$

## Independence
$$
P(A \cap B) = P(A)P(B)
$$

## Conditional independence
Same as conditional probability but with an extra given C" tacked on. Independence and conditional independence do not imply each other 


$P(A \cap B | C) = P(A | C)P(B | C)$


## Total probability

![Bayes diagram](bayes.svg)

$$
P(A) = \sum_{i}{P(A \cap B_i)}
$$

## Bayes rule
I can travel using a bus, car or a cycle (mutually exclusive partitioning) with certain probabilities ( $P(B_i)$ ), can be computed from data. Probabilities of being late for each mode of transportations ( $P(A | B_i)$ ) is calculated from data. I am late (event A, the observation). Given I am late, what is the probability I took the bus ($P(B_i|A)$) (or inferign the latent from observation)

$$
P(B_k \mid A)
= \frac{P(A \mid B_k)\,P(B_k)}{P(A)}
= \frac{P(A \mid B_k)\,P(B_k)}
       {\sum_{i} P(A \mid B_i)\,P(B_i)}
$$


## Distributions
$>= 0$, adds upto 1
PDF/PMF, CDF
discrete: bernoulli (coin toss), binomial (how many heads in n tosses), geometric (how many tosses till first head), poisson (kind of like binomial for large n, small p)
continuous: uniform, exponential, pareto (heavy tailed)


## Change of variable

The standard change of variables formula for probability densities. If you have a random variable $z$ with known density $p(z)$, and you
  define a new random variable $y = g(z)$, then the density of $y$ is:

  $$
  q(y) = p(z) \cdot \left|\frac{dz}{dy}\right|
  $$

  The intuition: probability mass must be conserved. The probability in a small interval $[z, z+dz]$ must equal the probability in the
  corresponding interval $[y, y+dy]$:

  $$
  p(z)dz = q(y)dy \implies q(y) = p(z)\frac{dz}{dy}
  $$

  The $dz/dy$ factor accounts for how the mapping stretches or compresses intervals. If a small $dy$ corresponds to a large $dz$, more probability
  mass gets "compressed" into that $dy$, so the density is higher there.


## Moments
Moment: $m_i = \mathbb{E}[X^i]$

Expectation/mean: $\mathbb{E}[X] = m_1$

Variance (Expectation of deviation from mean): $V(X) = \mathbb{E}[(X - m_1)^2] = m_2 - m_1^2$

See [this](ch_fn_clt.md) for moment generating function


## Joint probability
Function of both X and Y. Summing/integrating over one of the variables (by law of total probability) gives PDF/PMF of the other variable
Independent if the joint pdf/pmf can be written as product of individual pdf/pmfs $p_{X,Y}(x, y) = p_X(x)p_Y(y)$

If X and Y are independent then expectation of product is product of expectation (and we can wrap the variables in arbitrary functions as well)
$\mathbb{E}[g(X)f(Y)] = \mathbb{E}[g(X)] \mathbb{E}[f(Y)]$

## Conditional expectation
If we have a RV $X$, we can have a conditional RV $X|A$, then we can have the expectations/moments on that conditional RV $X|A$

For continuous case:

$$
f_{X|A}(x) = \frac{f_X(x)}{P(X \in A)}
$$

For computing the expectation we need only sum over $A$

## Revisiting total probability

Prob of an event can be calculated by summing over products of conditional probability and probability of that partition. This can be a very useful tool

we can calculate expectations too from this: $\mathbb{E}[g(X)] = \sum_i \mathbb{E}[g(X) \mid A_i]  \mathbb{P}(A_i)$, where $g$ is an arbitrary function

Sample: We can compute mean of geometric RV by partitioning it into 2 events, "first flip was head, and first flip was not head"

$$
\mathbb{E}[N] = \mathbb{E}[N | Y=1]P(Y=1) + \mathbb{E}[N | Y=0]P(Y=0) = p + (1-p)(1+\mathbb{E}(N))
$$

## Linearity of expectation
$$
\mathbb{E}[X+Y] = \mathbb{E}[X] + \mathbb{E}[Y]
$$

For example, we can express binomial RV as a sum of indicator variables, denoting if it was a success or not for each flip, and then use LoE to find expectation of binomial RV

## Normal distribution



### Linear combinations

Linear combinarions of gaussians are gaussians

Has the linear transform property

$$
X \sim \mathcal{N}(\mu, \sigma^2) \quad \Longrightarrow \quad Y = a X + b \sim \mathcal{N}(a \mu + b, a^2 \sigma^2)
$$


Univariate:
$$
X_i \sim \mathcal{N}(\mu_i, \sigma_i^2),
\quad i = 1, \dots, n,
\quad \text{independent}
$$

$$
Y = \sum_{i=1}^n a_i X_i
$$

$$
Y \sim \mathcal{N}
\left(
\sum_{i=1}^n a_i \mu_i,
\;
\sum_{i=1}^n a_i^2 \sigma_i^2
\right)
$$

multivariate:
$$
X \sim \mathcal{N}(\mu, \Sigma)
$$

$$
Y = A X + b
$$

$$
Y \sim \mathcal{N}
\big(
A\mu + b,
\;
A\Sigma A^T
\big)
$$

#### Converting to standard

We can convert any normal distribution to the standard one:

$$
X \sim \mathcal{N}(\mu, \sigma^2) \quad \Longrightarrow \quad
Z = \frac{X - \mu}{\sigma} \sim \mathcal{N}(0, 1)
$$

$$
\mathbb{P}(X \le x) = \mathbb{P}\!\Bigg( \frac{X - \mu}{\sigma} \le \frac{x - \mu}{\sigma} \Bigg)
= \Phi\Bigg(\frac{x - \mu}{\sigma}\Bigg)
$$

### Conditional

Conditionals are also gaussians

Bivariate:

$$
\begin{pmatrix}
X_1 \\
X_2
\end{pmatrix}
\sim
\mathcal{N}
\left(
\begin{pmatrix}
\mu_1 \\
\mu_2
\end{pmatrix},
\begin{pmatrix}
\sigma_1^2 & \rho \sigma_1 \sigma_2 \\
\rho \sigma_1 \sigma_2 & \sigma_2^2
\end{pmatrix}
\right)
$$

$$
X_1 \mid X_2 = x_2
\sim
\mathcal{N}
\left(
\mu_1 + \rho \frac{\sigma_1}{\sigma_2}(x_2 - \mu_2),
\;
\sigma_1^2 (1 - \rho^2)
\right)
$$


Multivariate:

$$
\begin{pmatrix}
X_1 \\
X_2
\end{pmatrix}
\sim
\mathcal{N}
\left(
\begin{pmatrix}
\mu_1 \\
\mu_2
\end{pmatrix},
\begin{pmatrix}
\Sigma_{11} & \Sigma_{12} \\
\Sigma_{21} & \Sigma_{22}
\end{pmatrix}
\right)
$$

$$
X_1 \mid X_2 = x_2
\sim
\mathcal{N}
\left(
\mu_1 + \Sigma_{12}\Sigma_{22}^{-1}(x_2 - \mu_2),
\;
\Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}
\right)
$$


## Central limit theorem
Sum of IID RVs tend to normal distribution


$$
X_1, X_2, \dots, X_n \overset{\text{i.i.d.}}{\sim} \text{with } \mathbb{E}[X_i] = \mu, \mathrm{Var}(X_i) = \sigma^2 < \infty
$$

$$
\frac{\sum_{i=1}^n X_i - n\mu}{\sigma \sqrt{n}} \xrightarrow{d} \mathcal{N}(0,1) \quad \text{as } n \to \infty
$$

If you consider the sample mean, which is just the sum divided by n, then:

$$
\bar{X}_n = \frac{1}{n} \sum_{i=1}^n X_i
$$

$$
\frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} \mathcal{N}(0,1) \quad \text{as } n \to \infty
$$

So, if we have say 100000 people we do 50 experiment. In each expt, I choose 20 people and measure their height then if we plot these 50 sample means, it will look like a normal distribution.

### Sample problem 1

Typically we are given a set of sampled data. We estimate the mean ($\mu^{'}$) and standard error ($\sigma^{'}$). Note the primes denote these are estimated from sample, as we do not actually know the underlying  true mean/sigma

Standard error is calculated as $SE = \frac{\sigma^{'}}{\sqrt{n}}$.
$\alpha$ Confidence interval is calculated as: $CI = \mu^{'} \pm t_{\frac{1+\alpha}{2}, n-1}.SE$
So for a 95% interval with 10 samples we look at 2 sided t distribution $t_{0.975, 9}$

This sould mean we are $\alpha \times 100$% confident that true population mean is between $CI$

### Sample problem 2
If we have iid RVs and we want to estimate probabilities of sum, we can approximate it using CLT, without bothering to compute the full actual pdf of the sum

### Proof
See [this](ch_fn_clt.md)