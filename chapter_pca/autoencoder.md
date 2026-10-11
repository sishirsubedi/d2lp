# PCA vs. Autoencoder
:label:`sec_pca_autoencoder`

### Overview

**PCA (Principal Component Analysis)** and **autoencoders** are both widely used for dimensionality reduction. Both map high-dimensional data into a lower-dimensional **latent representation**, but they learn that representation in fundamentally different ways.

The central idea is:

$$
\mathbf{x} \longrightarrow \mathbf{z}
$$

where:

- $\mathbf{x}$ is the original high-dimensional input.
- $\mathbf{z}$ is the lower-dimensional latent representation.

PCA learns a **linear transformation**, while an autoencoder can learn a **non-linear transformation**.

### Principal Component Analysis (PCA)

### Overview

PCA is a **linear dimensionality reduction** technique.

Given a data matrix

$$
X \in \mathbb{R}^{n \times m},
$$

PCA finds a set of orthogonal directions along which the data has the largest variance.

The lower-dimensional representation can be written as

$$
Z = XW,
$$

where:

- $X$ is the centered input data.
- $W$ contains the selected principal components.
- $Z$ is the lower-dimensional representation.

For example, if

$$
X \in \mathbb{R}^{n \times 100},
$$

and we want a 2-dimensional representation, PCA finds

$$
W \in \mathbb{R}^{100 \times 2}
$$

and produces

$$
Z = XW \in \mathbb{R}^{n \times 2}.
$$

Thus, every high-dimensional point is mapped to a point in a 2D latent space.

### How PCA Works

### Step 1: Center the Data

PCA first subtracts the mean of every feature.

$$
X_{\text{centered}}
=
X - \mu
$$

where $\mu$ is the vector containing the feature means.

Centering is important because PCA is based on variance and covariance.

### Step 2: Compute the Covariance Matrix

The covariance matrix is

$$
\Sigma
=
\frac{1}{n-1}X_{\text{centered}}^T X_{\text{centered}}.
$$

If there are $m$ features, then

$$
\Sigma \in \mathbb{R}^{m \times m}.
$$

The covariance matrix describes how the different features vary together.

### Step 3: Eigenvalue Decomposition

We solve

$$
\Sigma w_i = \lambda_i w_i.
$$

Here:

- $w_i$ is an eigenvector.
- $\lambda_i$ is the corresponding eigenvalue.

The eigenvalue tells us how much variance is captured by the corresponding eigenvector.

Therefore, PCA sorts the eigenvectors according to their eigenvalues:

$$
\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_m.
$$

The eigenvectors corresponding to the largest eigenvalues become the principal components.

### Step 4: Select the Top Components

Suppose we want a $d$-dimensional representation.

We select

$$
W =
[w_1,w_2,\ldots,w_d].
$$

Then

$$
Z = X_{\text{centered}}W.
$$

The resulting matrix

$$
Z \in \mathbb{R}^{n \times d}
$$

is the PCA latent representation.

### PCA Latent Space

The PCA latent space has several important properties.

### 1. Linear

The transformation is linear:

$$
\mathbf{z} = W^T\mathbf{x}.
$$

Therefore, PCA can only represent the data using a linear subspace.


### 2. Orthogonal Components

The principal components are orthogonal:

$$
w_i^T w_j = 0
\qquad
i \neq j.
$$

This means the PCA coordinates are uncorrelated when computed from centered data.

### 3. Ordered by Variance

The first principal component captures the largest possible variance.

The second captures the largest remaining variance subject to being orthogonal to the first.

Therefore:

$$
\lambda_1 > \lambda_2 > \lambda_3 > \cdots
$$

generally indicates decreasing amounts of explained variance.


### 4. Global Linear Structure

PCA considers the entire dataset when finding its principal directions.

It therefore provides a global **linear** approximation of the data.

However, this does not mean PCA always preserves every aspect of global geometry. It specifically preserves directions of large variance under its linear reconstruction objective.


### PCA as Reconstruction

PCA can also be understood as an encoder-decoder system.

The encoder is

$$
\mathbf{z} = W^T(\mathbf{x}-\mu).
$$

The decoder reconstructs the original point:

$$
\hat{\mathbf{x}}
=
W\mathbf{z}+\mu.
$$

Combining the two:

$$
\hat{\mathbf{x}}
=
WW^T(\mathbf{x}-\mu)+\mu.
$$

Therefore, PCA can be interpreted as finding the best low-dimensional **linear reconstruction** of the data.


### Autoencoder

### Overview

An autoencoder is a neural network designed to learn a compressed representation of its input.

The basic architecture is:

$$
\mathbf{x}
\rightarrow
\text{Encoder}
\rightarrow
\mathbf{z}
\rightarrow
\text{Decoder}
\rightarrow
\hat{\mathbf{x}}
$$

where:

- $\mathbf{x}$ is the original input.
- $\mathbf{z}$ is the latent representation.
- $\hat{\mathbf{x}}$ is the reconstructed input.

The autoencoder is trained so that

$$
\hat{\mathbf{x}} \approx \mathbf{x}.
$$


### Encoder

The encoder maps the input into a lower-dimensional latent space:

$$
\mathbf{z}=f_{\theta}(\mathbf{x}).
$$

For example, a neural network might perform

$$
\mathbf{h}_1
=
\sigma(W_1\mathbf{x}+b_1)
$$

followed by

$$
\mathbf{z}
=
\sigma(W_2\mathbf{h}_1+b_2).
$$

Because neural networks can contain non-linear activation functions, the mapping from $\mathbf{x}$ to $\mathbf{z}$ can be highly non-linear.

### Latent Space

The latent vector

$$
\mathbf{z}\in\mathbb{R}^d
$$

is the compressed representation of the input.

For a 2D autoencoder,

$$
\mathbf{z}
=
\begin{bmatrix}
z_1\\
z_2
\end{bmatrix}.
$$

Every input point is therefore mapped to a point in a two-dimensional latent space.

This makes the latent space useful for visualization.


### Decoder

The decoder maps the latent representation back to the original feature space:

$$
\hat{\mathbf{x}}
=
g_{\phi}(\mathbf{z}).
$$

For example:

$$
\mathbf{h}_2
=
\sigma(W_3\mathbf{z}+b_3)
$$

and

$$
\hat{\mathbf{x}}
=
W_4\mathbf{h}_2+b_4.
$$

The decoder attempts to reconstruct the original input.

### Training Objective

The autoencoder is trained by minimizing reconstruction error.

For continuous data, a common loss is mean squared error:

$$
\mathcal{L}
=
\frac{1}{n}
\sum_{i=1}^{n}
\|\mathbf{x}_i-\hat{\mathbf{x}}_i\|^2.
$$

The training objective is therefore

$$
\min_{\theta,\phi}
\sum_i
\|\mathbf{x}_i-g_{\phi}(f_{\theta}(\mathbf{x}_i))\|^2.
$$

The encoder and decoder parameters are learned jointly.

### PCA vs. Autoencoder

The most important difference is:

> **PCA learns a linear subspace, while a general autoencoder can learn a non-linear representation.**

Consider:

$$
\mathbf{x}
\rightarrow
\mathbf{z}
\rightarrow
\hat{\mathbf{x}}.
$$

For PCA:

$$
\mathbf{z}=W^T(\mathbf{x}-\mu)
$$

and

$$
\hat{\mathbf{x}}=W\mathbf{z}+\mu.
$$

Everything is linear.

For an autoencoder:

$$
\mathbf{z}=f_{\theta}(\mathbf{x})
$$

and

$$
\hat{\mathbf{x}}=g_{\phi}(\mathbf{z}),
$$

where $f_{\theta}$ and $g_{\phi}$ may contain multiple non-linear layers.


### Linear vs. Non-Linear

### PCA

PCA assumes that the important structure can be represented by a linear subspace.

For example, if the data approximately lies on a plane in a high-dimensional space, PCA can represent it effectively.

### Autoencoder

An autoencoder can learn a non-linear manifold.

For example, suppose the data lies on a curved surface:

$$
\text{high-dimensional space}
\rightarrow
\text{curved manifold}.
$$

A sufficiently expressive autoencoder can learn a non-linear transformation that unfolds or represents this structure.

### Important Connection: Linear Autoencoder and PCA

A particularly important result is:

> A linear autoencoder with a single hidden layer, a linear activation function, and an appropriate reconstruction objective can learn the same principal subspace as PCA.

This means PCA and autoencoders are not completely unrelated.

Consider a linear autoencoder:

$$
\mathbf{z}=W_e\mathbf{x}
$$

and

$$
\hat{\mathbf{x}}=W_d\mathbf{z}.
$$

Therefore,

$$
\hat{\mathbf{x}}
=
W_dW_e\mathbf{x}.
$$

The network is performing a linear projection followed by a linear reconstruction.

Under appropriate conditions, minimizing the reconstruction error gives the same principal subspace found by PCA.

Thus:

$$
\boxed{
\text{Linear Autoencoder}
\approx
\text{PCA}
}
$$

in terms of the learned subspace.

The important difference appears when we introduce **non-linear layers**.


### Non-Linear Autoencoder

Suppose the encoder is

$$
\mathbf{z}
=
f_{\theta}(\mathbf{x})
$$

and the decoder is

$$
\hat{\mathbf{x}}
=
g_{\phi}(\mathbf{z}).
$$

If $f_{\theta}$ and $g_{\phi}$ contain non-linear activation functions, then the complete transformation

$$
\mathbf{x}
\rightarrow
\mathbf{z}
\rightarrow
\hat{\mathbf{x}}
$$

is non-linear.

This allows the autoencoder to model curved or complicated manifolds.

Therefore:

$$
\boxed{
\text{PCA: linear latent space}
}
$$

while

$$
\boxed{
\text{Non-linear Autoencoder: potentially non-linear latent space}
}
$$


### Latent Space Geometry

The concept of a **2D latent space** is particularly useful for understanding the difference.

Suppose the original data has 100 features:

$$
\mathbf{x}\in\mathbb{R}^{100}.
$$

We want:

$$
\mathbf{z}\in\mathbb{R}^{2}.
$$

Both PCA and an autoencoder can perform:

$$
\mathbb{R}^{100}
\rightarrow
\mathbb{R}^{2}.
$$

However, the meaning of the 2D space can be very different.

### PCA

The 2D latent space is a linear projection:

$$
\mathbf{z}=W^T(\mathbf{x}-\mu).
$$

The two axes correspond to principal directions in the original feature space.

### Autoencoder

The 2D latent space is learned through a neural network:

$$
\mathbf{z}=f_{\theta}(\mathbf{x}).
$$

The axes do not necessarily correspond to simple combinations of the original features.

They are learned because they allow the decoder to reconstruct the data effectively.


### PCA vs. Autoencoder: Reconstruction View

Both methods can be understood through reconstruction.

### PCA

PCA finds a low-dimensional representation that minimizes linear reconstruction error.

$$
\min_W
\|X-XWW^T\|^2
$$

subject to appropriate orthogonality constraints on $W$.

The resulting representation is linear.

### Autoencoder

An autoencoder minimizes

$$
\min_{\theta,\phi}
\|X-g_{\phi}(f_{\theta}(X))\|^2.
$$

There is no requirement that the encoder or decoder be linear.

Therefore, the model can learn a much more flexible transformation.

### Comparison Table

| Property | PCA | Autoencoder |
|---|---|---|
| Type | Linear | Can be non-linear |
| Main mechanism | Eigenvectors / SVD | Neural network |
| Encoder | Linear projection | Neural network |
| Decoder | Linear reconstruction | Neural network |
| Training | Closed-form / SVD or eigendecomposition | Gradient-based optimization |
| Latent space | Linear subspace | Potentially non-linear manifold |
| Components | Orthogonal principal components | Not necessarily orthogonal |
| Objective | Maximize variance / minimize reconstruction error | Minimize reconstruction error |
| Interpretability | Relatively high | Usually lower |
| Hyperparameters | Mainly number of components | Architecture, latent dimension, learning rate, etc. |
| Computational cost | Usually low | Usually higher |
| Non-linear structure | Cannot directly model it | Can model it |
| Large datasets | Efficient | Can scale well with GPUs |
| Reconstruction | Linear | Potentially non-linear |


### PCA and Autoencoder: A More Important Comparison

It is tempting to say:

> PCA preserves global structure and autoencoders preserve local structure.

This statement is **too strong**.

The real distinction is the **objective and model class**.

PCA finds the best linear low-dimensional reconstruction.

An autoencoder finds a representation that allows a neural network to reconstruct the input.

Depending on the architecture, loss function, regularization, and training procedure, an autoencoder may learn different aspects of the data.

Therefore, we should not automatically classify:

$$
\text{PCA} = \text{global}
$$

and

$$
\text{Autoencoder} = \text{local}.
$$

A better statement is:

- **PCA:** linear representation optimized for variance and linear reconstruction.
- **Autoencoder:** learned representation optimized for reconstruction, potentially using non-linear transformations.


### Relationship to Previous Dimensionality Reduction Methods

The methods discussed earlier can be organized according to how they construct their low-dimensional representation.

### PCA

PCA starts with the data matrix:

$$
X
$$

and finds a linear projection:

$$
X\rightarrow Z.
$$

The important mathematical object is the covariance matrix:

$$
\Sigma=X^TX.
$$

The largest eigenvalues and corresponding eigenvectors are used.


### Isomap

Isomap first constructs a neighborhood graph and estimates geodesic distances:

$$
X
\rightarrow
D_{\text{geodesic}}.
$$

It then performs classical MDS:

$$
D_{\text{geodesic}}
\rightarrow
K
\rightarrow
Z.
$$

Thus, Isomap uses a distance-based representation and attempts to preserve global manifold geometry.


### Laplacian Eigenmaps

Laplacian Eigenmaps constructs a local neighborhood graph:

$$
X
\rightarrow
W
\rightarrow
L.
$$

It then solves an eigenvalue problem involving the graph Laplacian.

The eigenvectors corresponding to the smallest non-zero eigenvalues provide the embedding.


### t-SNE

t-SNE converts high-dimensional distances into probabilities:

$$
X
\rightarrow
P.
$$

It then learns low-dimensional coordinates whose probability distribution $Q$ approximately matches $P$:

$$
P
\rightarrow
Q
\rightarrow
Z.
$$

The objective is based on KL divergence:

$$
\operatorname{KL}(P\|Q).
$$

### UMAP

UMAP constructs a weighted nearest-neighbor graph:

$$
X
\rightarrow
W.
$$

It then optimizes low-dimensional coordinates using a cross-entropy-type objective:

$$
W
\rightarrow
Z.
$$


### Autoencoder

An autoencoder takes a different approach:

$$
X
\rightarrow
\text{Encoder}
\rightarrow
Z
\rightarrow
\text{Decoder}
\rightarrow
\hat{X}.
$$

Instead of explicitly constructing an $N\times N$ similarity matrix and then performing an eigenvalue decomposition or probability matching procedure, the model **learns the encoder directly through reconstruction**.

This is an important conceptual difference.

### The Bigger Picture

Many classical dimensionality reduction methods can be viewed as:

$$
\boxed{
\text{Define relationships}
\rightarrow
\text{Construct mathematical representation}
\rightarrow
\text{Find low-dimensional coordinates}
}
$$

For example:

| Method | High-dimensional representation | How $Z$ is obtained |
|---|---|---|
| PCA | Covariance / Gram structure | Eigenvectors / SVD |
| Isomap | Geodesic distances | MDS |
| Laplacian Eigenmaps | Graph weights / Laplacian | Eigenvectors |
| t-SNE | Pairwise probabilities | KL optimization |
| UMAP | Fuzzy neighborhood graph | Cross-entropy optimization |
| Autoencoder | Reconstruction objective | Neural network training |

This gives us a useful conceptual transition:

$$
\boxed{
\text{Classical DR}
\rightarrow
\text{explicit mathematical structure}
}
$$

while

$$
\boxed{
\text{Autoencoder}
\rightarrow
\text{learn the representation}
}
$$


### Why Use an Autoencoder?

An autoencoder becomes especially useful when the data contains complex non-linear structure.

For example, consider images.

An image may contain:

- edges,
- shapes,
- textures,
- objects,
- lighting variations,
- spatial relationships.

A simple linear projection may not capture these relationships efficiently.

A deep autoencoder can learn hierarchical representations:

$$
\mathbf{x}
\rightarrow
\mathbf{h}_1
\rightarrow
\mathbf{h}_2
\rightarrow
\mathbf{z}.
$$

The decoder then attempts to reconstruct the image:

$$
\mathbf{z}
\rightarrow
\mathbf{h}_3
\rightarrow
\mathbf{h}_4
\rightarrow
\hat{\mathbf{x}}.
$$

The latent vector $\mathbf{z}$ can therefore capture a compact representation of the important information needed for reconstruction.

### Important Limitation of Autoencoders

A smaller latent dimension does **not automatically** guarantee that the latent space is meaningful.

For example, an autoencoder could learn a representation that reconstructs the input well but has:

- poor interpretability,
- irregular geometry,
- discontinuities,
- sensitivity to training initialization,
- entangled latent dimensions.

The quality of the latent space depends on:

- architecture,
- latent dimension,
- training data,
- reconstruction loss,
- regularization,
- activation functions,
- optimization.

This is why specialized autoencoders such as **variational autoencoders (VAEs)** introduce additional constraints on the latent space.


### PCA vs. Autoencoder: Core Intuition

A useful way to remember the difference is:

### PCA asks:

> **What linear directions explain the most variance?**

Mathematically:

$$
X
\rightarrow
\text{principal directions}
\rightarrow
Z.
$$

### Autoencoder asks:

> **What compressed representation allows a neural network to reconstruct the input?**

Mathematically:

$$
X
\rightarrow
\text{encoder}
\rightarrow
Z
\rightarrow
\text{decoder}
\rightarrow
\hat{X}.
$$

Therefore:

$$
\boxed{
\text{PCA: find the best linear subspace}
}
$$

and

$$
\boxed{
\text{Autoencoder: learn a representation useful for reconstruction}
}
$$


### 2D Latent Space Example

Suppose we have a dataset containing images represented by 784 pixels:

$$
\mathbf{x}\in\mathbb{R}^{784}.
$$

We want to visualize the dataset using two dimensions.

We can use PCA:

$$
\mathbb{R}^{784}
\xrightarrow{\text{PCA}}
\mathbb{R}^{2}.
$$

The result is:

$$
\mathbf{x}_i
\rightarrow
(z_{i1},z_{i2}).
$$

We can plot:

$$
z_{i1}
\quad\text{vs.}\quad
z_{i2}.
$$

Alternatively, we can train an autoencoder:

$$
\mathbb{R}^{784}
\xrightarrow{\text{Encoder}}
\mathbb{R}^{2}
\xrightarrow{\text{Decoder}}
\mathbb{R}^{784}.
$$

The 2D latent coordinates can again be plotted.

However, the resulting plots can look quite different because the two methods optimize different objectives.


### Summary

The main differences between PCA and autoencoders are:

1. **PCA is linear.**

   $$
   Z=XW
   $$

2. **Autoencoders can be non-linear.**

   $$
   Z=f_{\theta}(X)
   $$

3. **PCA uses eigenvectors or SVD** to find principal directions.

4. **Autoencoders use neural-network optimization** to learn an encoder and decoder.

5. **PCA components are orthogonal** and ordered by explained variance.

6. **Autoencoder latent dimensions are not necessarily orthogonal** and do not have a fixed interpretation.

7. **PCA provides the optimal linear reconstruction subspace** under the standard squared-error formulation.

8. **A sufficiently simple linear autoencoder can recover the same principal subspace as PCA.**

9. **A non-linear autoencoder can represent non-linear manifolds** that cannot be represented efficiently by a linear PCA subspace.

10. A 2D latent space from PCA and a 2D latent space from an autoencoder may therefore have very different geometry and interpretation.

The key conceptual distinction is:

$$
\boxed{
\text{PCA}
=
\text{linear dimensionality reduction}
}
$$

while

$$
\boxed{
\text{Autoencoder}
=
\text{learned representation through reconstruction}
}
$$

and, when non-linear neural-network layers are used,

$$
\boxed{
\text{Autoencoder}
=
\text{non-linear dimensionality reduction}
}
$$
