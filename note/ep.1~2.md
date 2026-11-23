# INTRODUCTION
## limit:
- Industrialization of LLMs
- Scale matters
## aim:
- maximize efficiency
## LESSON OVERVIEW
## 1.Basics

## 2.transformer

## 3.scale up

## 4.data

# TOKENIZATION
## Concept 

- text: Unicode strings
- lm: distribution over sequences of tokens

- flaw:
	1. "hello" $\neq$ "_ hello"(blank can't be ignored)
	2. number explosion : one token for one number, but there are a lot numbers

## Compression ratio:
$$
\frac{N(byte)}{N(token)}
$$
**Trade-off: Token sequence length vs. Vocabulary size**

> [!note] Token sequence length vs. Vocabulary size
> For a fixed text, the number of UTF-8 bytes stays the same. A larger vocabulary can represent frequent, longer byte sequences with single tokens, often reducing the number of tokens and increasing the compression ratio (bytes/token).
>
> The cost is a larger vocabulary and larger embedding/output matrices at a fixed hidden dimension. The bit width of a token ID is a separate issue from the number of text bytes represented by that token.

## how to map
### One word, one token:
- pro: easy to understand, alien with human languages, good compression ratio
- con: unbounded vocab

### Byte-pair encoding (BPE)
- idea: learn a vocabulary and an ordered set of merge rules tailored to the training text.
- result: 

	common strings  $\rightarrow$ one token
	rare strings $\rightarrow$ multi


> [!note] how it works
> main idea: best mapping ways under a given  number of merges
> 1. Encode the text as UTF-8 bytes. One Unicode character may occupy multiple bytes.
> 2. Start with 256 basic tokens, one for each byte value from 0 to 255. Each token ID maps to a byte sequence.
> 3. Count adjacent token pairs and merge the most frequent pair into a new token. For example, if "he" occurs most frequently, merging `(104, 101)` can create token ID `256`, with `vocab[256] = "he"`. Here `256` is a token ID; a single byte can only have values from 0 to 255.
> 4. Replace occurrences of that pair with the new token ID, recount adjacent pairs, and repeat until the chosen number of merges is reached. Later pairs can include previously merged tokens, so one token may represent a longer byte sequence.
>
> Frequent byte sequences can therefore be represented by single tokens, while rare sequences can be represented by multiple tokens. For new text, encoding applies the learned merge rules in their learned order.

# Tensor
## Floating point
$$
sign + exponent + fraction
$$
- float 32 : 1 sign + 8 exponent + 23 fraction bits

**Trade-off:** Lower precision can reduce memory use and speed up computation, at the cost of numerical accuracy.

**Dynamic range vs. Precision:**
- dynamic range ---- exponent bits: fewer exponent bits generally mean a narrower range of representable magnitudes.
- precision ---- fraction bits: fewer fraction bits mean coarser spacing between nearby normal values and larger relative rounding errors.

- float 16 : 1 sign + 5 exponent + 10 fraction bits
- bfloat16 : 1 sign + 8 exponent + 7 fraction bits
- float 8 & float 4

FP16 has more fraction bits than BF16, so it is more precise over their shared normal range. BF16 has more exponent bits, giving it a dynamic range similar to FP32 and reducing the risk of overflow/underflow compared with FP16. Deep learning often tolerates reduced precision; mixed precision can retain FP32 for sensitive computations and optimizer states.

## Tensor in Python
### A useful pack ----- einop
#### def_sum: 
$$
\boxed{z_{s,t}=\sum_{h=0}^{H-1}x_{s,h}\,y_{h,t}} 
$$
```
z=einsum(x,y,"seq1 hidden, hidden seq2 -> seq1 seq2)
# z=x@y
```
#### def_reduce（lower the dimension):

$$
y_{b,s}=\sum_h x_{b,s,h}
$$
```
y=reduce(x,"... hidden" -> "...","sum" )
```
#### def_rearrange
$$
\boxed{y_{b,h,s}=x_{b,s,h}} 
$$
```
y = rearrange(x, "b s h -> b h s")
```
$$
\boxed{y_{b,\,sH+h}=x_{b,s,h}}
$$
```
y = rearrange(x, "b s h -> b (s h)")
```
# FLOP
	FLOPs:amount
	FLOP/S:speed
 
 - tips: why the promised FLOP/s should be divided by 2? They use sparse matrices to accelerate it.

- B: points
- D: input dimension
- K: output dimension

**data matrix: $\chi = B \times D$
weight matrix $\omega = D \times K$
```
x.shape = (B, D)
w.shape = (D, K)
y = x @ w             # 形状：(B, K)
```
actual_num_flops:  $B \times K \times (D + D-1) \approx 2 \times B \times D \times K$


## Buyer photos vs. seller photos

**Def_MFU:** model FLOP/s utilization
$$
\frac{actual\ FLOP/s}{promised \ FLOP/s}
$$
when:$MFU \geq 0.5$ ,pretty good.

**why is there a difference?**
Let $F$ be the number of FLOPs, $Q$ the bytes read and written, $P$ the hardware's peak FLOP/s at the chosen precision, and $\beta$ its memory bandwidth in bytes/s.

In an idealized timing model:
$$
t_{\mathrm{compute}}=\frac{F}{P},\qquad
t_{\mathrm{memory}}=\frac{Q}{\beta}
$$
Without overlap:
$$
t_{\mathrm{total}}=t_{\mathrm{compute}}+t_{\mathrm{memory}}
$$
Assuming perfect overlap, as in the lecture:
$$
t_{\mathrm{total}}=\max\left(t_{\mathrm{compute}},t_{\mathrm{memory}}\right)
$$
### Accelerator intensity:
**Def:** a hardware property: the maximum FLOPs the accelerator can perform during the time needed to transfer one byte, using peak compute and memory rates.
$$
I_{\mathrm{acc}}=\frac{P}{\beta}
$$

### Arithmetic intensity:
**Def:** a workload property: the FLOPs required per byte read or written at the memory level being analyzed.
$$
I_{\mathrm{arith}}=\frac{F}{Q}
$$
Both intensities have units of FLOP/byte.

### Compare
$$
I_{\mathrm{acc}}
=\frac{F/t_{\mathrm{compute}}}{Q/t_{\mathrm{memory}}}
=I_{\mathrm{arith}}\times\frac{t_{\mathrm{memory}}}{t_{\mathrm{compute}}}
$$
### Memory-bound vs. Compute-bound
- **Memory-bound:** $I_{\mathrm{arith}} < I_{\mathrm{acc}}$, equivalently $t_{\mathrm{memory}} > t_{\mathrm{compute}}$. Data transfer takes longer than computation.
- **Compute-bound:** $I_{\mathrm{arith}} > I_{\mathrm{acc}}$, equivalently $t_{\mathrm{compute}} > t_{\mathrm{memory}}$. Computation takes longer than data transfer.
- **Balanced:** $I_{\mathrm{arith}} = I_{\mathrm{acc}}$, so the two idealized times are equal.

# Gradients & network
One block has :
- an input: $x \in F^{B \times D \times layer}$ (B is the points count, D is dimension);
- a weight matrix: $w \in F^{D \times D \times layer}$
	- an activation ReLU for output $y=w x$: $\operatorname{ReLU}(y)=\max(0,y)$
Loss function:$\mathcal{L}(y)$

**network:** connect blocks like this together in the following way:
$$h^{(0)}=X \to z^{(\ell)}=h^{(\ell-1)}W^{(\ell)} \to  h^{(\ell)}=\operatorname{ReLU}(z^{(\ell)})$$

## forward:
$x ,w \to y=wx \to \mathcal{L}(y)$
### FLOPs:
As analyzed before:
$$
forward \_flops=2 \times B \times D \times D
$$

## backward:
- input: $\partial_y \mathcal{L} \to y.grad$
- weight matrix: $\partial _w \mathcal{L} \to w.grad$
If consider the essence of variables is tensor, the partial derivation should be replaced by gradients;

### FLOPs:
For the input, we need to do the gradient to both parameters, which actually refer to the weight matrix, and the inputs.Both parts need FLOPs:
$$
2 \times B \times D \times D
$$
Sum up:
$$
backward \_flops=4 \times B \times D \times D
$$
# Optimizer
**Def:** backward gives us how the loss changes with parameters, optimizer decide how to tune these parameters.

Let parameters be $\theta$ , the gradient in the iteration t of the training loop be:
$$
g_t=\nabla_\theta\mathcal L_t(\theta_t)
$$
## SGD:
$$
\theta_{t+1}=\theta_t-\eta g_t
$$
$\eta$ is the learning rate, controlling the step size.
**selling point :** move along the negative gradient. 
## Momentum:
$$
v_t=\mu v_{t-1}+g_t,\quad v_0=0 \\\\
\theta_{t+1}=\theta_t-\eta v_t
$$
**selling point :** keep some history information.
## AdaGrad:
$$
s_t=s_{t-1}+g_t\odot g_t,\qquad s_0=0
\\\\
\theta_{t+1}=\theta_t-\eta\frac{g_t}{\sqrt{s_t+\varepsilon}}
$$
**selling point :** change step size according to the history gradients' square .
$\varepsilon$ keeps denominator nonzero.

## RMSProp:
$$
s_t=\rho s_{t-1}+(1-\rho)(g_t\odot g_t)
\\\\
\theta_{t+1}=\theta_t-\eta\frac{g_t}{\sqrt{s_t}+\varepsilon}
$$
**selling point :** tune the extent of history weight by control $\rho$
## Adam:
$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
\\\\
v_t=\beta_2v_{t-1}+(1-\beta_2)(g_t\odot g_t)
$$
**selling point :** consider both gradients and gradients' square under a tunable history weight.

# Memory (bytes)
## parameter
$$
2  \times D \times D \times L
$$
- 2 for that there is 2 bytes for each BF16 token.
- D for dimension.
- L for layer
## activation:
$$
2 \times B \times D \times L
$$
- B for points.
## gradient_memory:
$$
2 \times Para\_ number
$$
- $Para\_number=D \times D \times L$
## optimizer:
$$
4 \times Para\_ number
$$
- to keep it stable, optimizers use FP32.That is why we time 4 here.
## total
$$
M_{\mathrm{AdaGrad}}\approx8N+2BDL\quad\text{bytes}
$$