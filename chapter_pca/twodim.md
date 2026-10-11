# Two-Dimensional Latent Space
:label:`sec_twodim`

The goal of dimensionality reduction is to take a high-dimensional dataset

$$
X = \{x_1,x_2,\ldots,x_n\}, \qquad x_i \in \mathbb{R}^m
$$

and represent each data point using a much smaller number of dimensions:

$$
x_i \in \mathbb{R}^m
\quad\longrightarrow\quad
z_i \in \mathbb{R}^d,
\qquad d \ll m.
$$

A particularly useful case is

$$
d=2,
$$

because the resulting representation can be directly visualized as a two-dimensional scatter plot.

We will call the original representation the **high-dimensional space** and the new representation the **latent space** or **embedding space**.

The central question is:

> **How should we place the points $z_i$ in a low-dimensional space so that important relationships between the original points $x_i$ are preserved?**

Different dimensionality-reduction methods answer this question differently.

### The General Idea

Most nonlinear dimensionality-reduction methods can be understood using the following pipeline:

$$
X
\longrightarrow
\text{relationships between points}
\longrightarrow
\text{low-dimensional embedding } Z.
$$

The important part is the middle step.

Instead of directly asking

> "What are the new coordinates?"

we first ask

> "Which points should be considered similar or close?"

This leads naturally to a matrix describing relationships between pairs of observations.

For $n$ observations, such a matrix has the form

$$
N \times N.
$$

For example,

$$
S =
\begin{bmatrix}
s_{11} & s_{12} & \cdots & s_{1n}\\
s_{21} & s_{22} & \cdots & s_{2n}\\
\vdots & \vdots & \ddots & \vdots\\
s_{n1} & s_{n2} & \cdots & s_{nn}
\end{bmatrix}.
$$

Here, $s_{ij}$ describes the relationship between points $x_i$ and $x_j$.

Depending on the method, this matrix may represent:

- covariance,
- inner products,
- Euclidean distances,
- geodesic distances,
- graph weights,
- probabilities,
- or fuzzy neighborhood memberships.

Therefore, a useful common theme is:

> **Many dimensional-reduction methods first construct an $N\times N$ relationship matrix and then find a low-dimensional representation that preserves some property of that matrix.**

However, this statement needs an important qualification.

**PCA does not normally construct an $N\times N$ similarity matrix.** Standard PCA works primarily with an $m\times m$ covariance matrix or directly with the centered data matrix. An $N\times N$ Gram matrix can also be used in a dual formulation.

---

### From $X$ to $Z$

Let

$$
X \in \mathbb{R}^{n\times m}
$$

contain $n$ observations and $m$ features.

Dimensionality reduction attempts to construct

$$
Z \in \mathbb{R}^{n\times d},
\qquad d\ll m.
$$

For two-dimensional visualization,

$$
Z =
\begin{bmatrix}
z_{11} & z_{12}\\
z_{21} & z_{22}\\
\vdots & \vdots\\
z_{n1} & z_{n2}
\end{bmatrix}.
$$

Each row

$$
z_i=(z_{i1},z_{i2})
$$

is the two-dimensional representation of the original point $x_i$.

The methods differ mainly in **what information about $X$ is preserved when constructing $Z$**.


### A Unifying View of Dimensionality Reduction

We can summarize several important methods using the following idea:

$$
X
\longrightarrow
\text{relationship representation}
\longrightarrow
Z.
$$

The relationship representation might be:

| Method | Relationship representation | Main idea |
|---|---|---|
| PCA | Covariance / Gram matrix | Preserve directions of maximum variance |
| Isomap | Geodesic distance matrix | Preserve global manifold distances |
| Laplacian Eigenmaps | Graph weight matrix / Laplacian | Preserve local neighborhoods |
| t-SNE | Probability affinities | Preserve local neighbor probabilities |
| UMAP | Fuzzy weighted neighborhood graph | Preserve local topology and neighborhood structure |

The important difference is therefore not simply whether a method uses a matrix.

The important difference is:

> **What does the matrix mean, and how is it converted into a low-dimensional embedding?**


### PCA

### Main Idea

PCA is a **linear** dimensionality-reduction method.

The data is first centered:

$$
X_c = X-\mathbf{1}\mu^T,
$$

where $\mu$ is the vector of feature means.

The covariance matrix is

$$
\Sigma =
\frac{1}{n-1}X_c^T X_c.
$$

Thus,

$$
\Sigma\in\mathbb{R}^{m\times m}.
$$

We then solve

$$
\Sigma w_k = \lambda_k w_k.
$$

The eigenvectors corresponding to the largest eigenvalues are selected.

If

$$
W_d =
\begin{bmatrix}
w_1&w_2&\cdots&w_d
\end{bmatrix},
$$

then the low-dimensional representation is

$$
Z=X_cW_d.
$$

### Why the Largest Eigenvalues?

The eigenvalue $\lambda_k$ represents the amount of variance captured by the corresponding principal direction $w_k$.

Therefore:

- large eigenvalue $\rightarrow$ large variance,
- small eigenvalue $\rightarrow$ small variance.

PCA keeps the directions with the **largest eigenvalues**.

For example,

$$
\lambda_1 > \lambda_2 > \lambda_3 > \cdots
$$

means that the first principal component explains the most variance.

Therefore, for PCA:

> **Keep the largest eigenvalues.**


### PCA and the Gram Matrix

There is another useful view of PCA.

For centered data,

$$
K=X_cX_c^T
$$

is an $n\times n$ Gram matrix.

Its entries are

$$
K_{ij}=x_i^Tx_j.
$$

Thus, $K_{ij}$ measures the inner-product relationship between observations $i$ and $j$.

This gives us an important connection:

$$
\boxed{
\text{PCA can be expressed using either feature-space covariance or sample-space Gram matrices.}
}
$$

However, the standard PCA implementation usually works with the covariance matrix or SVD rather than explicitly constructing the $n\times n$ Gram matrix.


### Isomap

Isomap is a nonlinear dimensional-reduction method designed to preserve the geometry of a manifold.

The basic pipeline is

$$
X
\rightarrow
\text{k-NN graph}
\rightarrow
\text{geodesic distances}
\rightarrow
\text{MDS}
\rightarrow
Z.
$$


### Step 1: Construct a Neighborhood Graph

For every point $x_i$, identify its $k$ nearest neighbors.

Connect neighboring points using edges whose weights are their Euclidean distances.

The graph contains only local connections.

For example:

$$
x_i
\leftrightarrow
x_j
$$

if $x_j$ is one of the neighbors of $x_i$.


### Step 2: Compute Geodesic Distances

The Euclidean distance between two points may not represent their true distance along a curved manifold.

Instead, Isomap approximates the manifold distance using the shortest path through the neighborhood graph.

For example,

$$
x_i
\rightarrow
x_a
\rightarrow
x_b
\rightarrow
x_j
$$

may be the shortest path between $x_i$ and $x_j$.

The resulting distance is called the **geodesic distance**:

$$
D_{ij}^{geo}.
$$

After computing shortest paths between all pairs, we obtain

$$
D^{geo}\in\mathbb{R}^{n\times n}.
$$


### Step 3: Apply Classical MDS

Classical MDS converts the geodesic distance matrix into a Gram matrix.

Define the centering matrix

$$
H=I-\frac{1}{n}\mathbf{1}\mathbf{1}^T.
$$

Then

$$
K=-\frac{1}{2}HD^2H,
$$

where $D^2$ means the element-wise squared distance matrix.

We then solve

$$
Kv_k=\lambda_kv_k.
$$

The embedding is constructed using the eigenvectors corresponding to the **largest positive eigenvalues**:

$$
Z=
V_d\Lambda_d^{1/2}.
$$

This point is important.

> **Isomap uses classical MDS after computing geodesic distances, and classical MDS keeps the largest positive eigenvalues of the centered Gram matrix.**

It is therefore incorrect to say that Isomap simply keeps the smallest eigenvalues.


### Isomap: Global vs. Local

Isomap begins with local information because the neighborhood graph is constructed using nearby points.

However, it then computes shortest paths between **all pairs of points**.

Therefore, the final distance matrix contains global relationships:

$$
D^{geo}_{ij}
$$

for every pair $(i,j)$.

This is why Isomap is generally described as emphasizing **global manifold geometry**.

A useful mental model is:

$$
\boxed{
\text{local graph}
\rightarrow
\text{global shortest paths}
\rightarrow
\text{global embedding}
}
$$


### Laplacian Eigenmaps

Laplacian Eigenmaps takes a different approach.

Instead of attempting to preserve distances between every pair of points, it focuses primarily on **local neighborhoods**.

The basic pipeline is

$$
X
\rightarrow
\text{k-NN graph}
\rightarrow
W
\rightarrow
D
\rightarrow
L
\rightarrow
\text{eigenvectors}
\rightarrow
Z.
$$


### Step 1: Construct the Neighborhood Graph

For each point $x_i$, identify its $k$ nearest neighbors.

Only these local relationships are represented in the graph.

This is an important difference from a full pairwise distance matrix.



### Step 2: Construct the Weight Matrix

Let

$$
W\in\mathbb{R}^{n\times n}
$$

be the graph weight matrix.

For a binary graph,

$$
W_{ij}=
\begin{cases}
1,&\text{if }i\text{ and }j\text{ are neighbors},\\
0,&\text{otherwise}.
\end{cases}
$$

Another common choice is the heat kernel:

$$
W_{ij}
=
\exp\left(
-\frac{\|x_i-x_j\|^2}{2\sigma^2}
\right).
$$

Nearby points receive larger weights.

Farther points receive smaller weights.


### Intuition Behind the Heat Kernel

Consider

$$
W_{ij}
=
\exp\left(
-\frac{\|x_i-x_j\|^2}{2\sigma^2}
\right).
$$

If

$$
\|x_i-x_j\|\approx0,
$$

then

$$
W_{ij}\approx1.
$$

If the distance becomes large,

$$
\|x_i-x_j\|\rightarrow\infty,
$$

then

$$
W_{ij}\rightarrow0.
$$

Therefore:

$$
\boxed{
\text{small distance}
\rightarrow
\text{large similarity}
}
$$

and

$$
\boxed{
\text{large distance}
\rightarrow
\text{small similarity}.
}
$$

The parameter $\sigma$ controls the scale of the kernel.

A smaller $\sigma$ gives a more local similarity measure, while a larger $\sigma$ produces a smoother neighborhood.

### Step 3: Degree Matrix

Define the degree of node $i$ as

$$
D_{ii}=\sum_jW_{ij}.
$$

The degree matrix is therefore

$$
D=
\begin{bmatrix}
D_{11}&0&\cdots&0\\
0&D_{22}&\cdots&0\\
\vdots&\vdots&\ddots&\vdots\\
0&0&\cdots&D_{nn}
\end{bmatrix}.
$$

### Step 4: Graph Laplacian

The unnormalized graph Laplacian is

$$
L=D-W.
$$

Other variants include the normalized Laplacians

$$
L_{\mathrm{sym}}
=
I-D^{-1/2}WD^{-1/2}
$$

and

$$
L_{\mathrm{rw}}
=
I-D^{-1}W.
$$


### Step 5: Solve the Eigenvalue Problem

For the unnormalized Laplacian formulation, Laplacian Eigenmaps solves the generalized eigenvalue problem

$$
Ly=\lambda Dy.
$$

The eigenvalues are ordered as

$$
\lambda_0\leq\lambda_1\leq\lambda_2\leq\cdots.
$$

For a connected graph,

$$
\lambda_0=0
$$

corresponds to a trivial constant eigenvector.

This eigenvector is discarded.

The embedding uses the eigenvectors corresponding to the next smallest eigenvalues:

$$
y_1,y_2,\ldots,y_d.
$$

Therefore:

$$
\boxed{
\text{Laplacian Eigenmaps}
\rightarrow
\text{smallest non-zero eigenvalues}.
}
$$


### Why Does Laplacian Eigenmaps Use Small Eigenvalues?

The objective function is

$$
\frac{1}{2}
\sum_{i,j}
W_{ij}
\|z_i-z_j\|^2.
$$

If two points have a large weight $W_{ij}$, they are considered strongly connected.

The objective therefore strongly penalizes placing them far apart.

The method tries to find an embedding where strongly connected points remain close.

This leads to the graph-smoothness interpretation:

> **The embedding should change slowly across strongly connected edges.**

The eigenvectors with small eigenvalues provide the smoothest nontrivial functions on the graph.


### PCA vs. Laplacian Eigenmaps

This gives an important distinction.

### PCA

PCA solves

$$
\Sigma w=\lambda w
$$

and keeps the directions with the **largest eigenvalues**.

Why?

Because large eigenvalues correspond to large variance.

### Laplacian Eigenmaps

Laplacian Eigenmaps solves

$$
Ly=\lambda Dy
$$

and keeps the eigenvectors with the **smallest non-zero eigenvalues**.

Why?

Because small eigenvalues correspond to smooth functions on the neighborhood graph.

Therefore:

$$
\boxed{
\text{PCA: largest eigenvalues}
}
$$

while

$$
\boxed{
\text{Laplacian Eigenmaps: smallest non-zero eigenvalues}
}
$$

These two statements should not be generalized to all dimensional-reduction methods.

In particular, **Isomap uses classical MDS and therefore keeps the largest positive eigenvalues of its centered Gram matrix.**

### The Laplacian Eigenmaps Objective

Let

$$
Z=
\begin{bmatrix}
z_1^T\\
z_2^T\\
\vdots\\
z_n^T
\end{bmatrix}.
$$

The objective is

$$
\sum_{i,j}
W_{ij}\|z_i-z_j\|^2.
$$

Expanding,

$$
\|z_i-z_j\|^2
=
z_i^Tz_i+z_j^Tz_j-2z_i^Tz_j.
$$

Using the degree matrix and graph Laplacian, this can be written as

$$
\sum_{i,j}
W_{ij}\|z_i-z_j\|^2
=
2\,\mathrm{Tr}(Z^TLZ).
$$

Thus, Laplacian Eigenmaps can be interpreted as minimizing

$$
\mathrm{Tr}(Z^TLZ)
$$

subject to an appropriate normalization such as

$$
Z^TDZ=I.
$$

This leads to the generalized eigenvalue problem

$$
Ly=\lambda Dy.
$$

### t-SNE

t-SNE takes a substantially different approach.

It does not simply solve an eigenvalue problem.

Instead, t-SNE constructs:

1. similarities in the high-dimensional space,
2. similarities in the low-dimensional space,
3. a loss measuring their disagreement,
4. and then optimizes the low-dimensional coordinates.

The basic idea is

$$
X
\rightarrow
P
\rightarrow
Z
\rightarrow
Q
\rightarrow
\text{KL divergence}.
$$

### High-Dimensional Similarities in t-SNE

For each point $x_i$, t-SNE defines a conditional probability

$$
p_{j|i}
=
\frac{
\exp\left(
-\frac{\|x_i-x_j\|^2}{2\sigma_i^2}
\right)
}{
\sum_{k\neq i}
\exp\left(
-\frac{\|x_i-x_k\|^2}{2\sigma_i^2}
\right)
}.
$$

This can be interpreted as:

> If we choose a neighbor of $x_i$ according to the Gaussian similarity, what is the probability that we choose $x_j$?

The bandwidth $\sigma_i$ is allowed to vary from point to point.


### Perplexity

The value of $\sigma_i$ is selected using the **perplexity** parameter.

For point $i$, define the Shannon entropy

$$
H(P_i)
=
-\sum_j
p_{j|i}\log_2p_{j|i}.
$$

Perplexity is

$$
\operatorname{Perplexity}(P_i)
=
2^{H(P_i)}.
$$

The algorithm searches for $\sigma_i$ such that

$$
2^{H(P_i)}
\approx
\text{desired perplexity}.
$$

This search is typically performed using binary search.

### Intuition for Perplexity

Perplexity can be thought of as an **effective neighborhood size**.

For example, a perplexity of approximately $30$ means that the Gaussian distribution around each point behaves roughly as though the point has about $30$ effective neighbors.

It is not exactly the same as saying:

> "Only 30 nearest neighbors are used."

Rather, the Gaussian probabilities are defined over the other points, while $\sigma_i$ is adjusted so that the resulting distribution has the desired entropy.

Therefore:

$$
\boxed{
\text{Perplexity controls the effective neighborhood scale in t-SNE.}
}
$$


### Symmetric High-Dimensional Probabilities

The conditional probabilities are converted into symmetric pairwise probabilities:

$$
p_{ij}
=
\frac{
p_{j|i}+p_{i|j}
}{
2n
}.
$$

This gives a symmetric probability distribution over pairs.

### Low-Dimensional Similarities

Suppose the low-dimensional points are

$$
z_1,z_2,\ldots,z_n.
$$

t-SNE uses a Student-$t$ distribution with one degree of freedom.

The pairwise similarity is

$$
q_{ij}
=
\frac{
(1+\|z_i-z_j\|^2)^{-1}
}{
\sum_{k\neq l}
(1+\|z_k-z_l\|^2)^{-1}
}.
$$

The heavy tails of the Student-$t$ distribution are important.

They allow moderately distant points in the low-dimensional space to retain non-negligible probability.

This helps reduce the tendency of many high-dimensional points to collapse into the same small region.

### t-SNE Objective

The goal is to make the low-dimensional distribution $Q$ resemble the high-dimensional distribution $P$.

t-SNE minimizes

$$
\operatorname{KL}(P\|Q)
=
\sum_{i\neq j}
p_{ij}
\log
\frac{p_{ij}}{q_{ij}}.
$$

This is an important correction to a common notation error:

$$
\boxed{
\text{t-SNE minimizes }KL(P\|Q),
\text{ not }KL(Q\|P).
}
$$


### t-SNE Optimization

Unlike PCA, Isomap, and Laplacian Eigenmaps, t-SNE does not obtain the embedding simply by solving one eigenvalue problem.

Instead:

1. Construct the high-dimensional probabilities $P$.
2. Initialize the low-dimensional coordinates $Z$.
3. Compute low-dimensional probabilities $Q$.
4. Compute the KL-divergence gradient.
5. Update $Z$ using gradient-based optimization.
6. Repeat.

Conceptually,

$$
Z^{(t+1)}
=
Z^{(t)}
-
\eta
\nabla_Z
\operatorname{KL}(P\|Q).
$$

Modern implementations use additional optimization techniques such as momentum and early exaggeration.

### The t-SNE Gradient

The gradient can be written as

$$
\frac{\partial C}{\partial z_i}
=
4
\sum_{j\neq i}
(p_{ij}-q_{ij})
(z_i-z_j)
(1+\|z_i-z_j\|^2)^{-1},
$$

where

$$
C=\operatorname{KL}(P\|Q).
$$

The important intuition is more useful than memorizing the exact equation.

If two points have a large high-dimensional similarity $p_{ij}$ but insufficient low-dimensional similarity $q_{ij}$, they experience an **attractive force**.

If the low-dimensional configuration produces too much similarity between points, the optimization produces a **repulsive effect**.

Thus t-SNE can be viewed as a system of attractive and repulsive forces.


### Early Exaggeration

Many t-SNE implementations temporarily multiply the high-dimensional affinities by an exaggeration factor during the early stages of optimization.

Conceptually,

$$
P
\rightarrow
\alpha P
$$

for some $\alpha>1$ during the initial optimization phase.

This encourages strongly related points to form coherent local groups early in the optimization.

After the early phase, the exaggeration is reduced or removed.


### UMAP

UMAP is another nonlinear dimensionality-reduction method.

Like Laplacian Eigenmaps, UMAP begins by constructing a neighborhood graph.

However, UMAP does not simply perform an eigenvalue decomposition of the graph Laplacian.

Instead, it constructs a **weighted fuzzy neighborhood graph** and then directly optimizes a low-dimensional layout.

The conceptual pipeline is

$$
X
\rightarrow
\text{k-NN graph}
\rightarrow
\text{fuzzy weighted graph}
\rightarrow
Z
\rightarrow
\text{cross-entropy optimization}.
$$


### UMAP Neighborhood Graph

UMAP first finds the $k$ nearest neighbors of each point.

For each point $i$, it estimates a local connectivity scale.

A simplified form of the membership strength is

$$
w_{ij}
=
\exp
\left(
-\frac{d(x_i,x_j)-\rho_i}{\sigma_i}
\right),
$$

for appropriate neighboring points, where:

- $\rho_i$ represents a local distance offset,
- $\sigma_i$ controls the local scale.

The exact UMAP construction involves additional details concerning fuzzy simplicial sets and the combination of directed memberships.

The important intuition is:

> **UMAP creates a weighted local neighborhood graph whose edge weights describe how strongly points are connected.**

### Low-Dimensional UMAP Similarity

UMAP defines a smooth similarity function in the low-dimensional space.

A commonly used form is

$$
f(d)
=
\frac{1}{1+a d^{2b}},
$$

where

$$
d=\|z_i-z_j\|
$$

and $a,b$ are parameters determined from UMAP's embedding settings.

Thus:

- small $d$ $\rightarrow$ similarity near $1$,
- large $d$ $\rightarrow$ similarity near $0$.


### UMAP Cross-Entropy Objective

UMAP tries to make the low-dimensional fuzzy graph resemble the high-dimensional fuzzy graph.

A conceptual cross-entropy objective is

$$
C
=
\sum_{i,j}
\left[
w_{ij}\log f(d_{ij})
+
(1-w_{ij})
\log(1-f(d_{ij}))
\right].
$$

Equivalently, when written as a loss to minimize, sign conventions may produce

$$
C
=
-\sum_{i,j}
\left[
w_{ij}\log f(d_{ij})
+
(1-w_{ij})
\log(1-f(d_{ij}))
\right].
$$

The exact implementation uses efficient sampling rather than explicitly evaluating every pair.

### Attraction and Repulsion in UMAP

The UMAP objective has two important effects.

### Attraction

If

$$
w_{ij}
$$

is large, points $i$ and $j$ are strongly connected in the high-dimensional graph.

The optimization therefore encourages

$$
z_i
\approx
z_j.
$$

### Repulsion

For pairs that are not strongly connected, the optimization discourages them from becoming artificially close.

Thus UMAP can be understood as balancing

$$
\boxed{
\text{attraction between neighbors}
}
$$

and

$$
\boxed{
\text{repulsion between non-neighbors}.
}
$$

### PCA, Isomap, Laplacian Eigenmaps, t-SNE, and UMAP

We can now compare the main ideas.

| Method | High-dimensional representation | Low-dimensional method | Main objective |
|---|---|---|---|
| PCA | Covariance / Gram matrix | Linear projection | Maximize variance |
| Isomap | Geodesic distances | Classical MDS | Preserve geodesic distances |
| Laplacian Eigenmaps | Local graph weights | Laplacian eigenvectors | Preserve local neighborhoods |
| t-SNE | Pairwise probabilities | Gradient optimization | Minimize $KL(P\|Q)$ |
| UMAP | Fuzzy neighborhood graph | Gradient optimization | Minimize cross-entropy |


### The Eigenvalue Perspective

It is tempting to summarize dimensionality reduction by saying:

> "Construct a matrix and take its eigenvectors."

This is true for some methods but not all.

A better summary is:

### PCA

$$
\text{Covariance}
\rightarrow
\text{eigenvectors}
\rightarrow
\text{largest eigenvalues}.
$$

### Isomap

$$
\text{Geodesic distances}
\rightarrow
\text{MDS Gram matrix}
\rightarrow
\text{largest positive eigenvalues}.
$$

### Laplacian Eigenmaps

$$
\text{Local graph}
\rightarrow
\text{Laplacian}
\rightarrow
\text{smallest non-zero eigenvalues}.
$$

### t-SNE

$$
\text{Probabilities}
\rightarrow
\text{KL divergence}
\rightarrow
\text{gradient optimization}.
$$

### UMAP

$$
\text{Fuzzy graph}
\rightarrow
\text{cross-entropy}
\rightarrow
\text{gradient optimization}.
$$

This distinction is extremely important.


### Local vs. Global Structure

One of the most useful ways to understand these algorithms is to ask:

> **Does the method primarily try to preserve local neighborhoods or global relationships?**

However, the answer should not be treated as an absolute classification.


### Isomap: Strong Global Geometry

Isomap constructs a local graph but then computes shortest paths between all pairs.

Therefore,

$$
\text{local graph}
\rightarrow
\text{global geodesic distances}.
$$

The final MDS step attempts to preserve these global distances.

Therefore, Isomap is commonly described as emphasizing **global manifold geometry**.

### Laplacian Eigenmaps: Local Structure

Laplacian Eigenmaps uses a neighborhood graph and its objective is based on

$$
\sum_{i,j}W_{ij}\|z_i-z_j\|^2.
$$

Only graph-connected pairs have a direct contribution.

Therefore, the method strongly emphasizes local relationships.

It is consequently described as a **local** manifold-learning method.


### t-SNE: Strong Local Neighborhood Preservation

t-SNE converts distances into probability affinities and minimizes

$$
KL(P\|Q).
$$

The KL objective is particularly sensitive to cases where a high-dimensional neighbor has been assigned too little probability in the embedding.

Therefore, t-SNE strongly emphasizes preserving local neighborhoods.

This is one reason t-SNE is very effective for visualizing clusters.

However, an important warning is:

> **Distances between separated t-SNE clusters should generally not be interpreted as reliable global distances.**

Two clusters being far apart in a t-SNE plot does not necessarily mean that they are far apart in the original high-dimensional space.


### UMAP: Local Structure with Broader Layout Information

UMAP starts from a local k-nearest-neighbor graph.

Therefore, local neighborhood relationships are central to the method.

However, UMAP's fuzzy graph representation and attractive/repulsive optimization can produce a broader global organization than t-SNE in many datasets.

It is therefore often described as preserving both local and global structure better than t-SNE.

However, this statement should be interpreted carefully.

> **UMAP does not guarantee that global distances in the embedding are faithful to global distances in the original space.**

Its global organization depends on the data, graph construction, hyperparameters, initialization, and optimization.


### A Better Global-vs-Local Summary

A useful practical summary is:

| Method | Local structure | Global structure |
|---|---:|---:|
| PCA | Moderate | Strong linear/global structure |
| Isomap | Used to construct graph | Strong |
| Laplacian Eigenmaps | Strong | Limited |
| t-SNE | Very strong | Weak / unreliable |
| UMAP | Strong | Often better than t-SNE, but not guaranteed |

The important point is that **local vs. global is not simply determined by whether an $N\times N$ matrix exists**.

Instead, we should ask:

> **Which relationships does the objective function prioritize?**


### Why Does k-NN Create Local Structure?

Suppose we have $n$ points.

A full pairwise distance matrix contains

$$
n^2
$$

entries.

For example,

$$
D=
\begin{bmatrix}
d_{11}&d_{12}&\cdots&d_{1n}\\
d_{21}&d_{22}&\cdots&d_{2n}\\
\vdots&\vdots&\ddots&\vdots\\
d_{n1}&d_{n2}&\cdots&d_{nn}
\end{bmatrix}.
$$

A k-NN graph keeps only approximately $k$ meaningful neighbors per point.

Thus, instead of considering all relationships equally, it focuses on

$$
k\ll n.
$$

This is why k-NN graphs are naturally associated with local structure.


### Similarity vs. Distance

A common source of confusion is that some methods work with **similarity** while others work with **distance**.

A distance behaves like

$$
d_{ij}\uparrow
\quad\Rightarrow\quad
\text{points become less similar}.
$$

A similarity behaves like

$$
s_{ij}\uparrow
\quad\Rightarrow\quad
\text{points become more similar}.
$$

For example, the heat kernel

$$
W_{ij}
=
\exp
\left(
-\frac{\|x_i-x_j\|^2}{2\sigma^2}
\right)
$$

converts a distance into a similarity.

Therefore:

$$
\text{distance}
\rightarrow
\text{similarity}
$$

is a common preprocessing step.


### Why the Exponential Kernel Works

Consider

$$
W_{ij}
=
\exp(-d_{ij}^2/c),
$$

where $c>0$.

If

$$
d_{ij}=0,
$$

then

$$
W_{ij}=1.
$$

If

$$
d_{ij}
$$

increases, then

$$
W_{ij}
$$

decreases rapidly toward zero.

Thus,

$$
d_{ij}\uparrow
\quad\Rightarrow\quad
W_{ij}\downarrow.
$$

This makes the exponential kernel a convenient way to transform distance into a smooth similarity measure.


### Why Different Methods Need Different Similarity Definitions

There is no single universally correct definition of similarity.

For example:

### PCA

Uses covariance and inner products.

### Isomap

Uses geodesic distance.

### Laplacian Eigenmaps

Uses local graph weights.

### t-SNE

Uses probability affinities.

### UMAP

Uses fuzzy graph membership strengths.

Therefore, when comparing dimensionality-reduction methods, we should ask two separate questions:

1. **How is similarity defined?**
2. **What does the algorithm do with that similarity?**


### One $N\times N$ Matrix, Different Meanings

The same matrix shape does not imply the same mathematical meaning.

For example:

$$
S_{ij}
$$

could mean:

- covariance,
- inner product,
- Euclidean distance,
- geodesic distance,
- graph weight,
- probability,
- fuzzy membership.

Therefore:

$$
\boxed{
N\times N
\text{ describes the shape, not the meaning.}
}
$$

This is an important conceptual point.


### A Unified View

We can now describe the methods using the same template.

### PCA

$$
X
\rightarrow
\text{variance/covariance}
\rightarrow
\text{eigenvectors}
\rightarrow
Z.
$$

It asks:

> Which linear directions explain the most variance?


### Isomap

$$
X
\rightarrow
\text{k-NN graph}
\rightarrow
\text{geodesic distances}
\rightarrow
\text{MDS}
\rightarrow
Z.
$$

It asks:

> Can we preserve distances along the manifold?


### Laplacian Eigenmaps

$$
X
\rightarrow
\text{k-NN graph}
\rightarrow
W
\rightarrow
L
\rightarrow
\text{eigenvectors}
\rightarrow
Z.
$$

It asks:

> Can we keep neighboring points close?


### t-SNE

$$
X
\rightarrow
P
\rightarrow
Q(Z)
\rightarrow
KL(P\|Q)
\rightarrow
Z.
$$

It asks:

> Can the low-dimensional space reproduce the neighborhood probabilities?


### UMAP

$$
X
\rightarrow
\text{fuzzy k-NN graph}
\rightarrow
\text{low-dimensional graph}
\rightarrow
\text{cross-entropy}
\rightarrow
Z.
$$

It asks:

> Can the low-dimensional space reproduce the fuzzy neighborhood structure?


### A Useful Mental Model

The following picture is a useful way to remember the main differences.

$$
\boxed{
\begin{array}{c}
\text{High-dimensional data }X\\
\downarrow\\
\text{How are points related?}\\
\downarrow\\
\text{Relationship representation}\\
\downarrow\\
\text{What should be preserved?}\\
\downarrow\\
\text{Low-dimensional space }Z
\end{array}
}
$$

The methods mainly differ in the answers to the two middle questions.


### Comparison of the Five Methods

| Property | PCA | Isomap | Laplacian Eigenmaps | t-SNE | UMAP |
|---|---|---|---|---|---|
| Linear/nonlinear | Linear | Nonlinear | Nonlinear | Nonlinear | Nonlinear |
| Main representation | Covariance / Gram | Geodesic distances | Graph weights | Probabilities | Fuzzy graph |
| Neighborhood graph | No | Yes | Yes | Implicit/local affinities | Yes |
| Main focus | Variance | Global manifold geometry | Local geometry | Local neighborhoods | Local topology + broader layout |
| Eigenvalue decomposition | Yes | Yes, through MDS | Yes | No | No |
| Eigenvalues selected | Largest | Largest positive | Smallest non-zero | N/A | N/A |
| Main optimization | Variance maximization | Distance preservation | Graph smoothness | KL divergence | Cross-entropy |
| Typical visualization use | Yes | Yes | Yes | Excellent | Excellent |


### Important Corrections to Remember

Several statements about dimensionality reduction are easy to get wrong.

### Correction 1: Isomap does not simply use the smallest eigenvalues

Isomap performs classical MDS after constructing the geodesic distance matrix.

The centered Gram matrix is

$$
K=-\frac{1}{2}HD^2H.
$$

The embedding uses the eigenvectors corresponding to the **largest positive eigenvalues**.


### Correction 2: Laplacian Eigenmaps uses the smallest non-zero eigenvalues

For

$$
Ly=\lambda Dy,
$$

the useful embedding coordinates come from the eigenvectors associated with the smallest non-zero eigenvalues.


### Correction 3: t-SNE minimizes $KL(P\|Q)$

The objective is

$$
KL(P\|Q)
=
\sum_{i\neq j}
p_{ij}
\log
\frac{p_{ij}}{q_{ij}}.
$$

It is not generally written as $KL(Q\|P)$.


### Correction 4: UMAP is not simply "Laplacian Eigenmaps with a different kernel"

Both methods use neighborhood graphs, but their mathematical frameworks are different.

Laplacian Eigenmaps solves an eigenvalue problem involving the graph Laplacian.

UMAP constructs a fuzzy topological representation and optimizes a low-dimensional layout using a cross-entropy-based objective.


### Correction 5: UMAP does not guarantee global distance preservation

UMAP often gives a more useful global organization than t-SNE, but this does not mean that distances between arbitrary clusters in UMAP should be interpreted as quantitatively accurate original-space distances.

### Final Summary

The central idea of dimensionality reduction is:

$$
\boxed{
X\rightarrow Z,\qquad
Z\in\mathbb{R}^{n\times d},
\quad d\ll m.
}
$$

The difficult part is deciding what information from $X$ should be preserved.

PCA preserves directions of large variance.

$$
\boxed{\text{PCA}\rightarrow\text{variance}}
$$

Isomap attempts to preserve global geodesic distances.

$$
\boxed{\text{Isomap}\rightarrow\text{geodesic geometry}}
$$

Laplacian Eigenmaps preserves local graph relationships.

$$
\boxed{\text{Laplacian Eigenmaps}\rightarrow\text{local neighborhoods}}
$$

t-SNE preserves local probability-based affinities.

$$
\boxed{\text{t-SNE}\rightarrow\text{local probabilities}}
$$

UMAP preserves a fuzzy representation of local neighborhood structure and optimizes a low-dimensional graph.

$$
\boxed{\text{UMAP}\rightarrow\text{fuzzy neighborhood topology}}
$$

A useful final mental model is:

$$
\boxed{
\begin{array}{ccc}
\text{PCA} &\rightarrow& \text{variance}\\
\text{Isomap} &\rightarrow& \text{geodesic distances}\\
\text{Laplacian Eigenmaps} &\rightarrow& \text{local graph smoothness}\\
\text{t-SNE} &\rightarrow& \text{probability matching}\\
\text{UMAP} &\rightarrow& \text{fuzzy graph matching}
\end{array}
}
$$

Thus, although these methods look very different mathematically, they all attempt to solve the same broad problem:

