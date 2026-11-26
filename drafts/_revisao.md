## De forma simplificada
### Hipótese
O método assume a hipótese de que o sinal é uma soma de exponenciais
$$
y(t) \approx \sum_{i=1}^{M} a_i \exp(b_i t), \quad 0 \le t \le T,
$$
com forma discreta
$$
y(k) \approx \sum_{i=1}^{M} a_i \exp(b_i k \Delta t), \quad k = 0, \dots, N-1,
$$
em que $t = k \Delta t$. 

### Determinando as exponenciais
Montamos uma matriz de Hankel $\mathbf{Y}$, com parâmetro de pencil $L$ (geralmente $L$ está entre $N/3$ e $N/2$)
$$
\mathbf{Y} =
\begin{bmatrix}
y(0) & y(1) & \cdots & y(L) \\
y(1) & y(2) & \cdots & y(L+1) \\
\vdots & \vdots & \ddots & \vdots \\
y(N-L-1) & y(N-L) & \cdots & y(N-1)
\end{bmatrix}_{(N-L)\times(L+1)},
$$
e criamos duas submatrizes de $\mathbf{Y}$: 
- $\mathbf{Y}_1 = \mathbf{Y}[:, \, 0{:}L-1]$, remove a última coluna de $\mathbf{Y}$; 
- $\mathbf{Y}_2 = \mathbf{Y}[:, \, 1{:}L]$, remove a primeira coluna de $\mathbf{Y}$.

Computados a SVD de $\mathbf{Y}$
$$
\mathbf{Y} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^{T},
$$
O que nos permite determinar $M$ pela razão de valores singulares (note que $p$ é um hiperparâmetro da função)
$$
\frac{\sigma_M}{\sigma_{\max}} \approx 10^{-p}.
$$


Nós retemos, então, apenas os $M$ vetores singulares à direita dominantes
$$
\mathbf{V} = [\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_M],
$$
e os vetores singulares direitos da $\mathbf{Y}_1$ e $\mathbf{Y}_2$ podem ser determinados em termos de $\mathbf{V}$:
- $\mathbf{V}_1 = \mathbf{V}[0{:}M-1, :]$;
- $\mathbf{V}_2 = \mathbf{V}[1{:}M, :]$.


De alguma forma, extraimos o argumento da exponencial por meio de um problema de autovalores
$$
\{\mathbf{Y}_2 - \lambda \mathbf{Y}_1\} \mathbf{v} = 0
\quad\Longrightarrow\quad
\lambda_i = \exp(b_i \Delta t), \quad i = 1, \dots, M
$$
que é equivalente à encontrar os autovalores de
$$
\{\mathbf{Y}_2^\mathrm{T}\mathbf{Y}_1\}
$$

### Determinando as amplitudes
Sistema linear para as amplitudes $a_i$ (mínimos quadrados)
$$
\begin{bmatrix}
y(0) \\ y(1) \\ \vdots \\ y(N-1)
\end{bmatrix}
=
\begin{bmatrix}
1 & 1 & \cdots & 1 \\
\lambda_1 & \lambda_2 & \cdots & \lambda_M \\
\vdots & \vdots & \ddots & \vdots \\
\lambda_1^{N-1} & \lambda_2^{N-1} & \cdots & \lambda_M^{N-1}
\end{bmatrix}
\begin{bmatrix}
a_1 \\ a_2 \\ \vdots \\ a_M
\end{bmatrix}
$$
