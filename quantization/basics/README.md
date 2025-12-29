


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
\boxed{
x_q = Q(X; s, z) =\mathrm{clamp}\left(\mathrm{round}\left(\frac{x}{s}\right) + z,\ q_{\text{min}},\ q_{\text{max}}\right)
}
$$


Given floating-point tensors $A$ and $B$, their quantized representations $A_q$ and $B_q$ are computed as:

$$
A_q^{(s_A,z_A)} = Q(A; s_A, z_A)
$$

$$
B_q^{(s_B,z_B)} = Q(B; s_B, z_B)
$$


Note that any quantized integer comes along with an implied scale/zero (which we will write as superscripts explicitly). We can perform arithmetic only on integers with the same scale/zero. Think of scale as units like "inch" and "centimeter". An even better analogy is farhenheit vs celcius scales which have both a scale and a zero offset.

Note that "scale" represents a single unit (the quantization scheme is blind to anything $<s$ which might get rounded away), because if x=s in the quantized world its $(x=s)/s = 1$ (assuming $z=0$)

## Integer matmul
The product in integer domain is:

$$
C_{Q}^{(s_As_B,0)} = (A_q - z_A) (B_q - z_B)
$$

Note we use subscript $Q$ instead of just $q$ to denote that this undergoes multiplication and accumulation, which expands the number of bits required to 32

Why are the zeros subtracted? See [this](#note-about-zero)

## Dequantization
Dequantization is done by:


$$
C = \text{DQ}(C_{Q}^{(s_As_B,0)}) = s_A s_B C_{Q}^{(s_As_B,0)}
$$

where the quantized matmul is:

$$
C_{Q}^{(s_As_B,0)} = (A_q^{(s_A,z_A)} - z_A) (B_q^{(s_B,z_B)} - z_B)
$$

and the dequantized result is:

$$
C = s_A s_B C_{Q}^{(s_As_B,0)}
$$



## Note about zero

What happens if we compute $A_q B_q$ without subtracting their respective zeros?


$$
\begin{aligned}
A_q^{(s_A,z_A)} B_q^{(s_B,z_B)} &= \left(\frac{A}{s_A} + z_A\right)\left(\frac{B}{s_B} + z_B\right) \\
&= \frac{A}{s_A} \frac{B}{s_B} + \frac{A}{s_A} z_B + z_A \frac{B}{s_B} + z_A z_B \\
&= \frac{AB}{s_A s_B} + \frac{A z_B}{s_A} + \frac{z_A B}{s_B} + z_A z_B
\end{aligned}
$$

Notice the ugliness here, there is no way to pack this back into $s(q - z)$ form for dequantization, except when we do not have the cross terms, which is possible if $z_A=0, z_B=0$.

Therefore we perform the quantized matmul as: $C_{Q}^{(s_As_B,0)} = (A_q^{(s_A,z_A)} - z_A) (B_q^{(s_B,z_B)} - z_B)$, which implies the dequantizing scale is $s_As_B$ and dequantizing zero is just $0$


## Code

```bash
python quantized_matmul.py
```



## Handling bias

We can add floating point bias after we dequantize $C$ of course. 

Alternatively we can add it in integer/quantized domain. Recall $C_{Q}^{(s_As_B,0)}$ is in $s_C = s_As_B$ and $z_C = 0$ domain. In that space the bias is quantized as $b_q^{(s_As_B,0)} = round \left( \frac{b}{s_As_b} \right)$

Now the matmul+bias in integer domain becomes:

$$
C_{Q}^{(s_As_B,0)} = (A_q^{(s_A,z_A)} - z_A) (B_q^{(s_A,z_A)} - z_B) + b_q^{(s_As_B,0)}
$$

## Requantization


## Feeding the next layer
If the inputs are int8, note that $C_Q^{(s_As_B,0)}$ is in int32 (because multiplying 2 int8 moves it to int16, and adding a lot of them needs even more space). However the next layer might need int8 again (assuming we dont dequantize and directly feed in the int/quant numbers).

During measurement phase, we might measure min/max of the output of the matmul (or input to next layer). These can be used to compute $s_C$ and $z_C$. Note that $C_{Q}^{(s_As_B,0)}$ is in $(s_As_B, 0)$ space, and we need to convert this to $(s_C, z_C)$ space. $s_As_B$ is the scale of the accumulator $C_Q$, while $s_C$ is calculated from data, which is the scale for fitting the next layer back into int8

Remember that "scale" denotes a "unit" length. This means a float value $s_As_B$ is $1$, and now after rescaling $s_C$ will denote that unit length. Also $C_{int32}$ is centered at $0$, but the new quantity will be centered at $z_C$ after requantization

Therefore we have:

$$
C_q^{(s_c,z_C)} = \text{clamp} \left( \text{round} \left( \frac{s_As_B}{s_C}C_{Q}^{(s_As_B,0)} + z_C \right), q^C_{min}, q^C_{max} \right)
$$

Because the accumulator is symmetric $z=0$, often times we use symmetric scheme for everything, so all the zero points are $0$ for ease of computation.


Thus requantization in general can be represented as:

$$
X_q^{(s_2,z_2)} = Q^{(s_2,z_2)}(DQ^{(s_1,z_1)}(X_q^{(s_1,z_1)}))
$$

## Additions

Consider addition of 2 quantized inputs (like say the residual addition in resnet). 2 quantized units cannot be added unless they are in the same scale/zero space

We have 2 options:
1. Dequantize, add in float domain, requantize, but this isn't hardware efficient
2. We must requantize one tensor to the other's scale


Whatever option we follow must give us same result in float domain:

$$
\begin{aligned}
A &= DQ^{(s_A,z_A)}(A_q^{(s_A,z_A)}) = s_A(A_q^{(s_A,z_A)} - z_A) \\
B &= DQ^{(s_B,z_B)}(B_q^{(s_B,z_B)}) = s_B(B_q^{(s_B,z_B)} - z_B) \\
C &= A+B
\end{aligned}
$$

### Option 1

Quantize back to A:

$$
\begin{aligned}
C_q^{(s_C,z_C)} &= Q^{(s_C,z_C)}(C) = Q^{(s_C,z_C)}(A+B) \\
=& \frac{\left(s_A(A_q^{(s_A,z_A)} - z_A) + s_B(B_q^{(s_B,z_B)} - z_B) \right)}{s_C} + z_C \\
 \end{aligned}
$$

What is $s_C$ and $z_C$? It can be arbitrary, however lets go through option 2 and see what values make both option 1 and option 2 equivalent.

### Option 2


First lets move $B$ to $A$'s scale/zero domain.

$$
\begin{aligned}
B_q^{(s_A,z_A)} &= Q^{(s_A,z_A)}( DQ^{(s_B,z_B)}( B_q^{(s_B,z_B)} )) \\
 &= \frac{s_B}{s_A}(B_q^{(s_B,z_B)} - z_B) +z_A 
 \end{aligned}
$$


Now we can safely add in quantized domain. However we do not know yet what C's scale zero is

$$
\begin{aligned}
C_q^{s_C, z_C} =& A_q^{(s_A,z_A)} + B_q^{(s_A,z_A)} \\
=& A_q^{(s_A,z_A)} + \frac{s_B}{s_A}(B_q^{(s_B,z_B)} - z_B) +z_A 
 \end{aligned}
$$

### $s_C$ and $z_C$ to make both paths equivalent

$$
\begin{aligned}
 A_q^{(s_A,z_A)} + \frac{s_B}{s_A}(B_q^{(s_B,z_B)} - z_B) +z_A &= \frac{\left(s_A(A_q^{(s_A,z_A)} - z_A) + s_B(B_q^{(s_B,z_B)} - z_B) \right)}{s_C} + z_C  \\
A_q^{(s_A,z_A)} + \frac{s_B}{s_A}B_q^{(s_B,z_B)} +z_A - \frac{s_B}{s_A}z_B &= \frac{s_A}{s_C} A_q^{(s_A,z_A)} + \frac{s_B}{s_C} B_q^{(s_B,z_B)} - \frac{s_A}{s_C}z_A - \frac{s_B}{s_C}z_B + z_C
 \end{aligned}
$$

Equating coefficients of like terms:

$$
1 = \frac{s_A}{s_C}  \implies s_C = s_A \\
$$

Replacing $s_C = s_A$ in: $z_A - \frac{s_B}{s_A}z_B = - \frac{s_A}{s_C}z_A - \frac{s_B}{s_C}z_B + z_C$ we get:

$$
z_C = 2z_A
$$



# Overflow analysis of quantized matmul

Assume we use $n_0$ bits for input matrices $X$ and $W$. The integer matmul itself is done in a wider integer type $n_1 > n_0$. Note that the int matmul is symmetric, as discussed in the quantized matmul section above

Range of inputs:

$$
\begin{aligned}
m_{n_0} = 2^{n_0-1} - 1\\
X_i, W_i \in \left[ -m_{n_0},m_{n_0} \right]
\end{aligned}
$$

Similarly range of accumulator is:

$$
\begin{aligned}
m_{n_1} = 2^{n_1-1} - 1 \\
acc \in \left[ - m_{n_1},m_{n_1} \right]
\end{aligned}
$$

### Worst case analysis
Assume max magnitude of product:

$$
X_iW_i = m^2_{n_0} = (2^{n_0-1}-1)^2
$$


Maximum safe number of accumulations:

$$
K^{worst}_{max} = round_{down}\left(\frac{m_{n_1}}{m^2_{n_0}}\right) = round_{down} \left( \frac{2^{n_1-1}-1}{(2^{n_0-1}-1)^2} \right)
$$

The approximate expression is (which is a stricter bound):

$$
\hat{K}^{worst}_{max} = 2^{n_1-2n_0 + 1}
$$

#### Concrete examples
For $n_0=8$, $n_1=16$, $K^{worst}_{max}=\left\lfloor \frac{2^{15}-1}{(2^{7}-1)^2} \right\rfloor = 2$

For $n_0=8$, $n_1=16$, $K^{worst}_{max}=\left\lfloor \frac{2^{31}-1}{(2^{7}-1)^2} \right\rfloor = 133144$

### Probabilistic analysis
Assume every element of $X_q$ and $W_q$ are independent and uniformly distributed. Then $X_q^{'} = X_q - z_x$ and $W_q^{'}=W_q-z_w$ both have zero means. Also they have same distribution:

$$
X_q^{'}, W_q^{'} \sim \mathcal{U}(-m_{n_0}, m_{n_0} ) =\mathcal{U}(- (2^{n_0-1} - 1), 2^{n_0-1} - 1)
$$

The mean of this is $0$ and the variance is $\sigma_x^2 = \frac{(2m_{n_0}+1)^2-1}{12} $


The RV $p_i = X_{q,i}^{'}W_{q,i}^{'}$ is zero mean and variance is a product (because iid): $\sigma^2_p = \text{Var}(p_i) = \text{Var}(X_{q,i})\text{Var}(W_{q,i})= {\left( \frac{(2m_{n_0}+1)^2-1}{12} \right)}^2$. 


Therefore:

$$
\sigma_p = \frac{(2m_{n_0}+1)^2-1}{12} = \frac{m_{n_0}(m_{n_0}+1)}{3} \approx \frac{m_{n_0}^2}{3}
$$

Now consider the RV $S_k = \sum_{i=1}^{k}{p_i}$. This is a discrete [Irwin-Hall distribution](https://en.wikipedia.org/wiki/Irwin%E2%80%93Hall_distribution), which is a bit janky


Instead we can use the Central Limit Theorem, to get a reasonable approximation, especially for large enough $k$


#### CLT refresher

See [this](./../probability/notes/basics.md#central-limit-theorem) for CLT statement and [this for proof](../../probability/notes/ch_fn_clt.md)


Let $X$ be a random variable with mean $\mu$ and variance $\sigma^2$. The sum of $k$ independent and identically distributed (i.i.d.) copies of $X$ is:

$$
S_k = \sum_{i=1}^k X_i
$$

The mean and variance of $S_k$ are:

$$
\mathbb{E}[S_k] = k\mu
$$

$$
\mathrm{Var}[S_k] = k\sigma^2
$$

By the Central Limit Theorem, for large $k$:

$$
S_k \approx \mathcal{N}(k\mu,\, k\sigma^2)
$$


By symmetry of Normal distribution
$$
P(|S_k| > A) = P(S_k > A) + P(S_k < -A) = 2P(S_k < -A)
$$

 

Standardize to $Z \sim  \mathcal{N}(0,1)$

$$
P(S_k<-A)=P\left(Z < \frac{-A-\mu}{\sigma}\right) = \Phi\left(\frac{-A-\mu}{\sigma}\right)\
$$

But we have:

$$
\begin{aligned}
\mu &= \sum{\mu_p} =0  \\
\sigma &= \sigma_p \sqrt{k}  \\
A &= m_{n_1} = acc_{max}
\end{aligned}
$$

Therefore 

$$
\begin{aligned}
P_{overflow} &= 2\Phi\left(-\frac{m_{n_1}}{\sigma_p \sqrt{k}}\right) \\
&=2\Phi\left( -\frac{2^{n_1-1}-1}{\sqrt{k} \left( \frac{m_{n_0}(m_{n_0}+1)}{3}\right)} \right) \\
&=2\Phi\left( -\frac{2^{n_1-1}-1}{\sqrt{k} \left( \frac{2^{n_0-1}(2^{n_0-1}-1)}{3}\right)} \right) \\
&\approx 2\Phi\left(- \frac{3.2^{n_1-1}}{2^{2(n_0-1)}\sqrt{k}} \right) = 2\Phi\left(- \frac{3.2^{n_1-2n_0+1}}{\sqrt{k}} \right) = 2\Phi\left(- \frac{3.\hat{K}^{worst}_{max}}{\sqrt{k}} \right)  
\end{aligned}
$$

Keeping the probability 3 std deviations away:

$$
\begin{aligned}
\frac{3.2^{n_1-2n_0+1}}{\sqrt{k}} \ge 3 \\
k \le 2^{2(n_1-2n_0+1)} \\
k \le (\hat{K}^{worst}_{max})^2
\end{aligned}
$$



Calculating some "safe" $k$ values for different quantization bits. Note that for $n_0=4, n_1=16$, its still not very safe, as the worst case supports only $512$ accumulations, but probabilistically it might work out.

| $n_0$ | $n_1$ | approx probabilistic safe $k$ | approx worst case safe $k$  |
|-------|-------|---------------------------|---------------------------|
| 8     | 16    | $2^{2}$             | $2$ | 
| 8     | 32    | $2^{34}$                  | $2^{17}$ |
| 4     | 16    | $2^{18}$             | $2^9$ |


Note, we can also use other bounds (like Hoeffding's) to bound this

# Simulating quantized matmul: bit-exact vs fake-quant
Till now we have been discussing the bit-exact method of simulating accuracy for quantized matmuls, whose pseudo-code is

$$
\begin{aligned}
x_q &= Q(x) \\
w_q &= Q(w) \\
m_{int32} &= (x_q-z_x)(w_q-z_w) \\
m_1 &= s_w s_x m_{int32} 
\end{aligned}
$$

Often times an alternate method is used, called fake-quant


$$
\begin{aligned}
x_q &= DQ(Q(x))\\
w_q &= DQ(Q(w))\\
m_2 &= x_q w_q
\end{aligned}
$$

These methods are equivalent if we are in a regime where no clamping/overflow is happening (its easy to see this, as the 3 operations, quantize, dequantize are scalar linear operations that can be swapped around with the matmul). However in real usecases we might have clamping and overflows.


## Some experiments:
Small $k$, using int32 accumulator: `python fakequant_vs_bitexact.py --M 40 --K 80 --N 40 --widen-type 32`, we will see practically no difference in the 2 methods


But, if we use less bits for accumulation, `python fakequant_vs_bitexact.py --M 40 --K 80 --N 40 --widen-type 16` we see that the 2 methods diverge, with about 5-10% elements overflowing





