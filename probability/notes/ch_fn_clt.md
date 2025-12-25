


# Sum is convolution
PDF of sum of RVs is equivalent to convolving their PDFs, Why? because for $Z=X+Y$, we can use total probability to partition to partition it over all values of $X$, so integrating over $f_{Z|X}(z|x) f_X(x)dx$. As X and Y are IID, $f_{Z=X+Y | X}(z|x)$ is just $f_{Y}(z-x)$. Plugging it it, it looks a lot like convolution


#  Exponentiated expectations



| Property                | Fourier Transform                        | Moment Generating Function (MGF)         | Characteristic Function (CF)             | Cumulant Generating Function (CGF)         |
|-------------------------|------------------------------------------|------------------------------------------|------------------------------------------|---------------------------------------------|
| Definition              |  $\mathcal{F}[f]$ $(t)$ = $\int_{-\infty}^{\infty} e^{-itx} f(x)dx$ | $M_X(t) = \mathbb{E}[e^{tX}]$            | $\varphi_X(t) = \mathbb{E}[e^{itX}]$      | $K_X(t) = \log M_X(t)$                      |
| Exponent                | $e^{-itx}$                               | $e^{tX}$                                 | $e^{itX}$                                | $e^{tX}$                                    |
| Variable                | $t$ (real or complex)                    | $t$ (real, sometimes complex)            | $t$ (real, sometimes complex)            | $t$ (real, sometimes complex)               |
| Domain                  | Functions (signals, densities, etc.)     | Random variables                         | Random variables                         | Random variables                            |
| Use                     | Signal processing, analysis              | Calculating moments, probability theory  | Distribution analysis, CLT, inversion    | Calculating cumulants, independence, CLT    |
| Relation to PDF         | Transforms $f(x)$ to frequency domain    | Encodes all moments of $X$               | Uniquely determines distribution         | Encodes all cumulants of $X$                |
| Inverse formula         | Yes                                      | No (but moments can reconstruct)         | Yes                                      | No (but cumulants can reconstruct moments)  |


## MGF

Expanding $e^{tX}$ using its Taylor series:

$$
e^{tX} = 1 + tX + \frac{t^2 X^2}{2!} + \frac{t^3 X^3}{3!} + \cdots = \sum_{n=0}^{\infty} \frac{t^n X^n}{n!}
$$

Taking expectation:

$$
M_X(t) = \mathbb{E}[e^{tX}] = \mathbb{E}\left[\sum_{n=0}^{\infty} \frac{t^n X^n}{n!}\right] = \sum_{n=0}^{\infty} \frac{t^n}{n!} \mathbb{E}[X^n]
$$



### Moments of the Poisson Distribution from the MGF

Let $X \sim \mathrm{Poisson}(\lambda)$. The MGF of $X$ is:

$$
M_X(t) = \mathbb{E}[e^{tX}] = \sum_{k=0}^{\infty} e^{tk} \frac{e^{-\lambda} \lambda^k}{k!} = e^{-\lambda} \sum_{k=0}^{\infty} \frac{(\lambda e^{t})^k}{k!} = e^{-\lambda} e^{\lambda e^{t}} = e^{\lambda (e^{t} - 1)}
$$

The $k^{th}$ moment is the $k^{th}$ derivative at 0. Why? because the $k^{th}$ derivative kills the first $k-1$ terms of the expansion, and setting to 0 kills the terms after k, leaving only the coefficient at $k$

#### First Moment (Mean)
The first moment is the first derivative of the MGF at $t=0$:

$$
M_X'(t) = \frac{d}{dt} e^{\lambda (e^{t} - 1)} = e^{\lambda (e^{t} - 1)} \cdot \lambda e^{t}
$$

So,

$$
M_X'(0) = \lambda
$$

#### Second Moment
The second moment is the second derivative at $t=0$:

$$
M_X''(t) = \frac{d}{dt} \left[ e^{\lambda (e^{t} - 1)} \cdot \lambda e^{t} \right]
$$

At $t=0$:

$$
M_X''(0) = \lambda (\lambda + 1)
$$


# Convolving a lot of the same

Addition of RVs is convolution in pdf space, which is multiplication in characteristic function space

$$
Z = X+Y
$$

$$
f_Z(z) = (f_X * f_Y)(z) = \int_{-\infty}^{\infty} f_X(x) f_Y(z - x)dx
$$

$$
\varphi_Z(t) = \varphi_X(t) \cdot \varphi_Y(t)
$$

For $n$ iid variables: 

$$
\varphi_{X_1 + X_2 + \cdots + X_n}(t) = [\varphi_X(t)]^n
$$


# Proving CLT


## High level guidelines
In the following steps, the following principles guide us:
1. separate mean and variance
2. stable under raising to nth power: meaning its "easy/tractable" to raise to nth power
3. expanding upto $t^2$

## Characteristic function
Expanding $e^{itX}$ using its Taylor series:

$$
e^{itX} = 1 + itX + \frac{(itX)^2}{2!} + \frac{(itX)^3}{3!} + \cdots = \sum_{n=0}^{\infty} \frac{(itX)^n}{n!}
$$

Take expectations:

$$
\mathbb{E}[e^{itX}] = \mathbb{E}\left[1 + itX + \frac{(itX)^2}{2!} + \frac{(itX)^3}{3!} + \cdots \right]
$$

$$
\mathbb{E}[e^{itX}] \approx 1 + it\mathbb{E}[X] - \frac{t^2}{2}\mathbb{E}[X^2]
$$

$$
\varphi_X(t) = \mathbb{E}[e^{itX}] \approx 1 + it\mu - \frac{t^2}{2}(\sigma^2 + \mu^2)
$$

This is difficult to exponentiate though, so we need to find an alternate expression thats more pliable.


## Finding another expression with same expansion thats amenable to exponentiation
Key idea: If two expressions have the same Taylor expansion up to order $t^2$ they are equivalent for our purposes



Now consider this proposed form:

$$
e^{i \mu t}(1+at^2)
$$

$$
=(1 + it\mu - \frac{t^2}{2}(\sigma^2 + \mu^2) )(1+at^2)
$$

$$
= 1 + i\mu t + (a-\mu^2/2)t^2 + ...
$$

Note that it matches $e^{i \mu t}$ if we have $a=-\sigma^2/2$

Therefore $e^{i \mu t}(1-\sigma^2/2t^2)$ is a valid replacement for $\varphi_X(t)$ upto second order expansion. It separates out $\mu$ and $\sigma$ neatly



Furthur note the expansion of $e^{-\frac{\sigma^2}{2} t^2}$ in a Taylor series around $t=0$: $e^{-\frac{\sigma^2}{2} t^2} = 1 - \frac{\sigma^2}{2} t^2 + \cdots$. This lets us wrap up $1 - \frac{\sigma^2}{2} t^2$ as $e^{-\frac{\sigma^2}{2} t^2}$

## Putting the replacement into $\varphi_X(t)$

So the characteristic function which was originally:

$$
\varphi_X(t) = \mathbb{E}[e^{itX}] \approx 1 + it\mu - \frac{t^2}{2}(\sigma^2 + \mu^2) \approx e^{i \mu t} \left( 1 - \frac{\sigma^2}{2} \right) t^2 = e^{i \mu t}e^{-\frac{\sigma^2}{2} t^2}
$$

Remember why we did this: because we have to raise $\varphi_X(t)$ to the nth power stably

## Final steps: wrapping it into a normal distribution

Raising to the $n$ th power:

$$
\left( e^{i \mu t} e^{-\frac{\sigma^2}{2} t^2} \right)^n = e^{i n \mu t} e^{-\frac{n \sigma^2}{2} t^2}
$$


The characteristic function of the normal distribution $X \sim \mathcal{N}(\mu, \sigma^2)$ is:

$$
\varphi_X(t) = \mathbb{E}[e^{itX}] = e^{i\mu t - \frac{1}{2} \sigma^2 t^2}
$$


Matching these two, we see:

$$
X_1 + X_2 + ... + X_n \sim \mathcal{N}(n\mu, n\sigma^2)
$$


So, repeated convolution smooths out, and it starts to look like the normal/gaussian function.

