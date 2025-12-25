


At a very high level, the following are very simple formulae linearly mapping $x_{\text{min}}$ to $q_{\text{min}}$ and $x_{\text{max}}$ to $q_{\text{max}}$. Scale ($s$) is just the reciprocal of slope and zero point ($z$) is the y-intercept of the linear equation mapping floats to integers. We need to add a few other bells and whistles like rounding (to make sure its an integer) and clamping (to make sure its within the integer range). Finally for operations like matmuls, we have to take a little extra care.


# min/max vs scale/zero

## qmin, qmax

For $n$ bits:

### Symmetric quantization

$$
\boxed{
\begin{aligned}
q_{\text{min}} &= - (2^{n-1} - 1) + 1 \\
q_{\text{max}} &= 2^{n-1} - 1
\end{aligned}
}
$$

Note this is centered around $0$, but misses one possible value ($- (2^{n-1} - 1) $)


### Asymmetric quantization

$$
\boxed{
\begin{aligned}
q_{\text{min}} &= 0 \\
q_{\text{max}} &= 2^n - 1
\end{aligned}
}
$$



## min/max to scale/zero


Given $x_{\text{min}}$ and $x_{\text{max}}$ as the floating-point range, and $q_{\text{min}}$, $q_{\text{max}}$ as the quantized integer range:

$$
\boxed{
\begin{aligned}
\text{s} &= \frac{x_{\text{max}} - x_{\text{min}}}{q_{\text{max}} - q_{\text{min}}} \\
\text{z} &= \min\left(q_{\text{max}}, \max\left(q_{\text{min}}, \mathrm{round}\left(q_{\text{min}} - \frac{x_{\text{min}}}{\text{s}}\right) \right)\right) = \mathrm{clamp}\left(\mathrm{round}\left(q_{min} - \frac{x_\text{min}}{s}\right),\ q_{\text{min}},\ q_{\text{max}}\right)
\end{aligned}
}
$$


Note that $z$ is an integer in the allowed range $[q_{\text{min}}, q_{\text{max}}]$ (hence we round to get an integer, and clamp to stay within range)


### Zero point for symmetric quantization


For symmetric quantization, $x_{\text{min}} = -x_{\text{max}}$ and $q_{\text{min}} = -q_{\text{max}}$:

$$
\text{z} = q_{\text{min}} - \frac{x_{\text{min}}}{\text{s}}
$$

Substituting $q_{\text{min}} = -q_{\text{max}}$ and $x_{\text{min}} = -x_{\text{max}}$:

$$
\text{z} = -q_{\text{max}} - \frac{-x_{\text{max}}}{\text{s}}
$$

The scale is:

$$
\text{s} = \frac{x_{\text{max}} - x_{\text{min}}}{q_{\text{max}} - q_{\text{min}}} = \frac{x_{\text{max}} - (-x_{\text{max}})}{q_{\text{max}} - (-q_{\text{max}})} = \frac{2x_{\text{max}}}{2q_{\text{max}}} = \frac{x_{\text{max}}}{q_{\text{max}}}
$$

So,

$$
\text{z} = -q_{\text{max}} - \frac{-x_{\text{max}}}{x_{\text{max}}/q_{\text{max}}} = -q_{\text{max}} + q_{\text{max}} = 0
$$

Thus, for symmetric quantization, 

$$
\boxed{
\text{z} = 0
}
$$

# Matmul


## Quantization

Quantization is done by:

$$
x_q = Q(X; s, z) =\mathrm{clamp}\left(\mathrm{round}\left(\frac{x}{s}\right) + z,\ q_{\text{min}},\ q_{\text{max}}\right)
$$


Given floating-point tensors $A$ and $B$, their quantized representations $A_q$ and $B_q$ are computed as:

$$
A_q = Q(A; s_A, z_A)
$$

$$
B_q = Q(B; s_B, z_B)
$$

## Integer matmul
The product in integer domain is:

$$
C_q = (A_q - z_A) (B_q - z_B)
$$


Why are the zeros subtracted? See [this](#note-about-zero)

## Dequantization
Dequantization is done by:


$$
C = \text{DQ}(C_q) = s_A s_B C_q
$$

where the quantized matmul is:

$$
C_q = (A_q - z_A) (B_q - z_B)
$$

and the dequantized result is:

$$
C = s_A s_B C_q
$$



## Note about zero

What happens if we compute $A_q B_q$ without subtracting their respective zeros?

$$
A_q = \frac{A}{s_A} + z_A
B_q = \frac{B}{s_B} + z_B
$$




Expanding $A_q B_q$:

$$
\begin{aligned}
A_q B_q &= \left(\frac{A}{s_A} + z_A\right)\left(\frac{B}{s_B} + z_B\right) \\
&= \frac{A}{s_A} \frac{B}{s_B} + \frac{A}{s_A} z_B + z_A \frac{B}{s_B} + z_A z_B \\
&= \frac{AB}{s_A s_B} + \frac{A z_B}{s_A} + \frac{z_A B}{s_B} + z_A z_B
\end{aligned}
$$

Notice the ugliness here, there is no way to pack this back into $s(q - z)$ form for dequantization, except when we do not have the cross terms, which is possible if $z_A=0, z_B=0$.

Therefore we perform the quantized matmul as: $C_q = (A_q - z_A) (B_q - z_B)$, which implies the dequantizing scale is $s_As_B$ and dequantizing zero is just $0$

## Handling bias

We can add floating point bias after we dequantize $C$ of course. 

Alternatively we can add it in integer/quantized domain. Recall $C_q$ is in $s_C = s_As_B$ and $z_C = 0$ domain. In that space the bias is quantized as $b_q = round \left( \frac{b}{s_As_b} \right)$

Now the matmul+bias in integer domain becomes:

$$
C_q = (A_q - z_A) (B_q - z_B) + b_q
$$

## Code

```bash
python quantized_matmul.py
```