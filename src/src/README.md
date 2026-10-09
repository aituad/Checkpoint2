Для файла README.md или ячейки Markdown в Jupyter можно оформить решение так:

# H3. Momentum Method

## Given

\[
f(x,y)=x^2+3y^2
\]

Initial point:

\[
x^{(0)}=(5,2)
\]

Parameters:

\[
\alpha=0.1,\qquad \beta=0.6
\]

Initial velocity:

\[
v^{(0)}=(0,0)
\]

---

## Momentum Method

Update equations:

\[
v^{(k+1)}=\beta v^{(k)}-\alpha \nabla f(x^{(k)})
\]

\[
x^{(k+1)}=x^{(k)}+v^{(k+1)}
\]

Gradient:

\[
\nabla f(x,y)=
\begin{bmatrix}
2x \\
6y
\end{bmatrix}
\]

### Iteration Table

| k | \(v^{(k)}\) | \(x^{(k)}\) | \(f(x^{(k)})\) |
|---|---|---|---|
| 0 | (0, 0) | (5, 2) | 37.00 |
| 1 | (-1.0, -1.2) | (4.0, 0.8) | 17.92 |
| 2 | (-1.4, -1.2) | (2.6, -0.4) | 7.24 |

---

## Plain Gradient Descent

Update rule:

\[
x^{(k+1)}=x^{(k)}-\alpha \nabla f(x^{(k)})
\]

### Step 1

\[
x^{(1)}=(5,2)-0.1(10,12)
=(4,0.8)
\]

### Step 2

\[
x^{(2)}=(4,0.8)-0.1(8,4.8)
=(3.2,0.32)
\]

Function value:

\[
f(x^{(2)})
=3.2^2+3(0.32)^2
=10.5472
\]

---

## Comparison

Momentum after two steps:

\[
x_{\text{mom}}^{(2)}=(2.6,-0.4)
\]

Gradient Descent after two steps:

\[
x_{\text{GD}}^{(2)}=(3.2,0.32)
\]

### Which coordinate advances faster?

The \(x\)-coordinate advances faster with Momentum:

\[
2.6 < 3.2
\]

so Momentum moves closer to the optimum in the \(x\)-direction.

### Which coordinate overshoots zero?

The \(y\)-coordinate overshoots zero:

\[
0.8 \rightarrow -0.4
\]

The sign changes from positive to negative, meaning Momentum crosses the minimizer.

### Is \(f\) monotone along the Momentum iterates?

Function values:

\[
37.00 \rightarrow 17.92 \rightarrow 7.24
\]

Since the objective decreases at every step, \(f\) is monotone decreasing along these Momentum iterates.