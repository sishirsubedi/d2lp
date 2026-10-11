
.. _sec_pca_autoencoder:

PCA vs. Autoencoder
===================


Overview
~~~~~~~~

**PCA (Principal Component Analysis)** and **autoencoders** are both
widely used for dimensionality reduction. Both map high-dimensional data
into a lower-dimensional **latent representation**, but they learn that
representation in fundamentally different ways.

The central idea is:

.. math::


   \mathbf{x} \longrightarrow \mathbf{z}

where:

-  :math:`\mathbf{x}` is the original high-dimensional input.
-  :math:`\mathbf{z}` is the lower-dimensional latent representation.

PCA learns a **linear transformation**, while an autoencoder can learn a
**non-linear transformation**.

.. _overview-1:

Principal Component Analysis (PCA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Overview
~~~~~~~~

PCA is a **linear dimensionality reduction** technique.

Given a data matrix

.. math::


   X \in \mathbb{R}^{n \times m},

PCA finds a set of orthogonal directions along which the data has the
largest variance.

The lower-dimensional representation can be written as

.. math::


   Z = XW,

where:

-  :math:`X` is the centered input data.
-  :math:`W` contains the selected principal components.
-  :math:`Z` is the lower-dimensional representation.

For example, if

.. math::


   X \in \mathbb{R}^{n \times 100},

and we want a 2-dimensional representation, PCA finds

.. math::


   W \in \mathbb{R}^{100 \times 2}

and produces

.. math::


   Z = XW \in \mathbb{R}^{n \times 2}.

Thus, every high-dimensional point is mapped to a point in a 2D latent
space.

How PCA Works
~~~~~~~~~~~~~

Step 1: Center the Data
~~~~~~~~~~~~~~~~~~~~~~~

PCA first subtracts the mean of every feature.

.. math::


   X_{\text{centered}}
   =
   X - \mu

where :math:`\mu` is the vector containing the feature means.

Centering is important because PCA is based on variance and covariance.

Step 2: Compute the Covariance Matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The covariance matrix is

.. math::


   \Sigma
   =
   \frac{1}{n-1}X_{\text{centered}}^T X_{\text{centered}}.

If there are :math:`m` features, then

.. math::


   \Sigma \in \mathbb{R}^{m \times m}.

The covariance matrix describes how the different features vary
together.

Step 3: Eigenvalue Decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We solve

.. math::


   \Sigma w_i = \lambda_i w_i.

Here:

-  :math:`w_i` is an eigenvector.
-  :math:`\lambda_i` is the corresponding eigenvalue.

The eigenvalue tells us how much variance is captured by the
corresponding eigenvector.

Therefore, PCA sorts the eigenvectors according to their eigenvalues:

.. math::


   \lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_m.

The eigenvectors corresponding to the largest eigenvalues become the
principal components.

Step 4: Select the Top Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suppose we want a :math:`d`-dimensional representation.

We select

.. math::


   W =
   [w_1,w_2,\ldots,w_d].

Then

.. math::


   Z = X_{\text{centered}}W.

The resulting matrix

.. math::


   Z \in \mathbb{R}^{n \times d}

is the PCA latent representation.

PCA Latent Space
~~~~~~~~~~~~~~~~

The PCA latent space has several important properties.

1. Linear
~~~~~~~~~

The transformation is linear:

.. math::


   \mathbf{z} = W^T\mathbf{x}.

Therefore, PCA can only represent the data using a linear subspace.

2. Orthogonal Components
~~~~~~~~~~~~~~~~~~~~~~~~

The principal components are orthogonal:

.. math::


   w_i^T w_j = 0
   \qquad
   i \neq j.

This means the PCA coordinates are uncorrelated when computed from
centered data.

3. Ordered by Variance
~~~~~~~~~~~~~~~~~~~~~~

The first principal component captures the largest possible variance.

The second captures the largest remaining variance subject to being
orthogonal to the first.

Therefore:

.. math::


   \lambda_1 > \lambda_2 > \lambda_3 > \cdots

generally indicates decreasing amounts of explained variance.

4. Global Linear Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~

PCA considers the entire dataset when finding its principal directions.

It therefore provides a global **linear** approximation of the data.

However, this does not mean PCA always preserves every aspect of global
geometry. It specifically preserves directions of large variance under
its linear reconstruction objective.

PCA as Reconstruction
~~~~~~~~~~~~~~~~~~~~~

PCA can also be understood as an encoder-decoder system.

The encoder is

.. math::


   \mathbf{z} = W^T(\mathbf{x}-\mu).

The decoder reconstructs the original point:

.. math::


   \hat{\mathbf{x}}
   =
   W\mathbf{z}+\mu.

Combining the two:

.. math::


   \hat{\mathbf{x}}
   =
   WW^T(\mathbf{x}-\mu)+\mu.

Therefore, PCA can be interpreted as finding the best low-dimensional
**linear reconstruction** of the data.

.. _overview-2:

Autoencoder
~~~~~~~~~~~


Overview
~~~~~~~~

An autoencoder is a neural network designed to learn a compressed
representation of its input.

The basic architecture is:

.. math::


   \mathbf{x}
   \rightarrow
   \text{Encoder}
   \rightarrow
   \mathbf{z}
   \rightarrow
   \text{Decoder}
   \rightarrow
   \hat{\mathbf{x}}

where:

-  :math:`\mathbf{x}` is the original input.
-  :math:`\mathbf{z}` is the latent representation.
-  :math:`\hat{\mathbf{x}}` is the reconstructed input.

The autoencoder is trained so that

.. math::


   \hat{\mathbf{x}} \approx \mathbf{x}.

Encoder
~~~~~~~

The encoder maps the input into a lower-dimensional latent space:

.. math::


   \mathbf{z}=f_{\theta}(\mathbf{x}).

For example, a neural network might perform

.. math::


   \mathbf{h}_1
   =
   \sigma(W_1\mathbf{x}+b_1)

followed by

.. math::


   \mathbf{z}
   =
   \sigma(W_2\mathbf{h}_1+b_2).

Because neural networks can contain non-linear activation functions, the
mapping from :math:`\mathbf{x}` to :math:`\mathbf{z}` can be highly
non-linear.

Latent Space
~~~~~~~~~~~~

The latent vector

.. math::


   \mathbf{z}\in\mathbb{R}^d

is the compressed representation of the input.

For a 2D autoencoder,

.. math::


   \mathbf{z}
   =
   \begin{bmatrix}
   z_1\\
   z_2
   \end{bmatrix}.

Every input point is therefore mapped to a point in a two-dimensional
latent space.

This makes the latent space useful for visualization.

Decoder
~~~~~~~

The decoder maps the latent representation back to the original feature
space:

.. math::


   \hat{\mathbf{x}}
   =
   g_{\phi}(\mathbf{z}).

For example:

.. math::


   \mathbf{h}_2
   =
   \sigma(W_3\mathbf{z}+b_3)

and

.. math::


   \hat{\mathbf{x}}
   =
   W_4\mathbf{h}_2+b_4.

The decoder attempts to reconstruct the original input.

.. _pca-vs.-autoencoder-1:

Training Objective
~~~~~~~~~~~~~~~~~~

The autoencoder is trained by minimizing reconstruction error.

For continuous data, a common loss is mean squared error:

.. math::


   \mathcal{L}
   =
   \frac{1}{n}
   \sum_{i=1}^{n}
   \|\mathbf{x}_i-\hat{\mathbf{x}}_i\|^2.

The training objective is therefore

.. math::


   \min_{\theta,\phi}
   \sum_i
   \|\mathbf{x}_i-g_{\phi}(f_{\theta}(\mathbf{x}_i))\|^2.

The encoder and decoder parameters are learned jointly.


PCA vs. Autoencoder
~~~~~~~~~~~~~~~~~~~

The most important difference is:

   **PCA learns a linear subspace, while a general autoencoder can learn
   a non-linear representation.**

Consider:

.. math::


   \mathbf{x}
   \rightarrow
   \mathbf{z}
   \rightarrow
   \hat{\mathbf{x}}.

For PCA:

.. math::


   \mathbf{z}=W^T(\mathbf{x}-\mu)

and

.. math::


   \hat{\mathbf{x}}=W\mathbf{z}+\mu.

Everything is linear.

For an autoencoder:

.. math::


   \mathbf{z}=f_{\theta}(\mathbf{x})

and

.. math::


   \hat{\mathbf{x}}=g_{\phi}(\mathbf{z}),

where :math:`f_{\theta}` and :math:`g_{\phi}` may contain multiple
non-linear layers.

Linear vs. Non-Linear
~~~~~~~~~~~~~~~~~~~~~

.. _autoencoder-1:

PCA
~~~

PCA assumes that the important structure can be represented by a linear
subspace.

For example, if the data approximately lies on a plane in a
high-dimensional space, PCA can represent it effectively.


Autoencoder
~~~~~~~~~~~

An autoencoder can learn a non-linear manifold.

For example, suppose the data lies on a curved surface:

.. math::


   \text{high-dimensional space}
   \rightarrow
   \text{curved manifold}.

A sufficiently expressive autoencoder can learn a non-linear
transformation that unfolds or represents this structure.

Important Connection: Linear Autoencoder and PCA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A particularly important result is:

   A linear autoencoder with a single hidden layer, a linear activation
   function, and an appropriate reconstruction objective can learn the
   same principal subspace as PCA.

This means PCA and autoencoders are not completely unrelated.

Consider a linear autoencoder:

.. math::


   \mathbf{z}=W_e\mathbf{x}

and

.. math::


   \hat{\mathbf{x}}=W_d\mathbf{z}.

Therefore,

.. math::


   \hat{\mathbf{x}}
   =
   W_dW_e\mathbf{x}.

The network is performing a linear projection followed by a linear
reconstruction.

Under appropriate conditions, minimizing the reconstruction error gives
the same principal subspace found by PCA.

Thus:

.. math::


   \boxed{
   \text{Linear Autoencoder}
   \approx
   \text{PCA}
   }

in terms of the learned subspace.

The important difference appears when we introduce **non-linear
layers**.

Non-Linear Autoencoder
~~~~~~~~~~~~~~~~~~~~~~

Suppose the encoder is

.. math::


   \mathbf{z}
   =
   f_{\theta}(\mathbf{x})

and the decoder is

.. math::


   \hat{\mathbf{x}}
   =
   g_{\phi}(\mathbf{z}).

If :math:`f_{\theta}` and :math:`g_{\phi}` contain non-linear activation
functions, then the complete transformation

.. math::


   \mathbf{x}
   \rightarrow
   \mathbf{z}
   \rightarrow
   \hat{\mathbf{x}}

is non-linear.

This allows the autoencoder to model curved or complicated manifolds.

Therefore:

.. math::


   \boxed{
   \text{PCA: linear latent space}
   }

while

.. math::


   \boxed{
   \text{Non-linear Autoencoder: potentially non-linear latent space}
   }

.. _pca-1:

Latent Space Geometry
~~~~~~~~~~~~~~~~~~~~~

The concept of a **2D latent space** is particularly useful for
understanding the difference.

Suppose the original data has 100 features:

.. math::


   \mathbf{x}\in\mathbb{R}^{100}.

We want:

.. math::


   \mathbf{z}\in\mathbb{R}^{2}.

Both PCA and an autoencoder can perform:

.. math::


   \mathbb{R}^{100}
   \rightarrow
   \mathbb{R}^{2}.

However, the meaning of the 2D space can be very different.


.. _autoencoder-2:

PCA
~~~

The 2D latent space is a linear projection:

.. math::


   \mathbf{z}=W^T(\mathbf{x}-\mu).

The two axes correspond to principal directions in the original feature
space.


Autoencoder
~~~~~~~~~~~

The 2D latent space is learned through a neural network:

.. math::


   \mathbf{z}=f_{\theta}(\mathbf{x}).

The axes do not necessarily correspond to simple combinations of the
original features.

They are learned because they allow the decoder to reconstruct the data
effectively.

.. _pca-2:

PCA vs. Autoencoder: Reconstruction View
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both methods can be understood through reconstruction.


.. _autoencoder-3:

PCA
~~~

PCA finds a low-dimensional representation that minimizes linear
reconstruction error.

.. math::


   \min_W
   \|X-XWW^T\|^2

subject to appropriate orthogonality constraints on :math:`W`.

The resulting representation is linear.


Autoencoder
~~~~~~~~~~~

An autoencoder minimizes

.. math::


   \min_{\theta,\phi}
   \|X-g_{\phi}(f_{\theta}(X))\|^2.

There is no requirement that the encoder or decoder be linear.

Therefore, the model can learn a much more flexible transformation.

Comparison Table
~~~~~~~~~~~~~~~~

+-----------------------+-----------------------+-----------------------+
| Property              | PCA                   | Autoencoder           |
+=======================+=======================+=======================+
| Type                  | Linear                | Can be non-linear     |
+-----------------------+-----------------------+-----------------------+
| Main mechanism        | Eigenvectors / SVD    | Neural network        |
+-----------------------+-----------------------+-----------------------+
| Encoder               | Linear projection     | Neural network        |
+-----------------------+-----------------------+-----------------------+
| Decoder               | Linear reconstruction | Neural network        |
+-----------------------+-----------------------+-----------------------+
| Training              | Closed-form / SVD or  | Gradient-based        |
|                       | eigendecomposition    | optimization          |
+-----------------------+-----------------------+-----------------------+
| Latent space          | Linear subspace       | Potentially           |
|                       |                       | non-linear manifold   |
+-----------------------+-----------------------+-----------------------+
| Components            | Orthogonal principal  | Not necessarily       |
|                       | components            | orthogonal            |
+-----------------------+-----------------------+-----------------------+
| Objective             | Maximize variance /   | Minimize              |
|                       | minimize              | reconstruction error  |
|                       | reconstruction error  |                       |
+-----------------------+-----------------------+-----------------------+
| Interpretability      | Relatively high       | Usually lower         |
+-----------------------+-----------------------+-----------------------+
| Hyperparameters       | Mainly number of      | Architecture, latent  |
|                       | components            | dimension, learning   |
|                       |                       | rate, etc.            |
+-----------------------+-----------------------+-----------------------+
| Computational cost    | Usually low           | Usually higher        |
+-----------------------+-----------------------+-----------------------+
| Non-linear structure  | Cannot directly model | Can model it          |
|                       | it                    |                       |
+-----------------------+-----------------------+-----------------------+
| Large datasets        | Efficient             | Can scale well with   |
|                       |                       | GPUs                  |
+-----------------------+-----------------------+-----------------------+
| Reconstruction        | Linear                | Potentially           |
|                       |                       | non-linear            |
+-----------------------+-----------------------+-----------------------+

PCA and Autoencoder: A More Important Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is tempting to say:

   PCA preserves global structure and autoencoders preserve local
   structure.

This statement is **too strong**.

The real distinction is the **objective and model class**.

PCA finds the best linear low-dimensional reconstruction.

An autoencoder finds a representation that allows a neural network to
reconstruct the input.

Depending on the architecture, loss function, regularization, and
training procedure, an autoencoder may learn different aspects of the
data.

Therefore, we should not automatically classify:

.. math::


   \text{PCA} = \text{global}

and

.. math::


   \text{Autoencoder} = \text{local}.

A better statement is:

-  **PCA:** linear representation optimized for variance and linear
   reconstruction.
-  **Autoencoder:** learned representation optimized for reconstruction,
   potentially using non-linear transformations.

.. _pca-3:

Relationship to Previous Dimensionality Reduction Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The methods discussed earlier can be organized according to how they
construct their low-dimensional representation.


PCA
~~~

PCA starts with the data matrix:

.. math::


   X

and finds a linear projection:

.. math::


   X\rightarrow Z.

The important mathematical object is the covariance matrix:

.. math::


   \Sigma=X^TX.

The largest eigenvalues and corresponding eigenvectors are used.

Isomap
~~~~~~

Isomap first constructs a neighborhood graph and estimates geodesic
distances:

.. math::


   X
   \rightarrow
   D_{\text{geodesic}}.

It then performs classical MDS:

.. math::


   D_{\text{geodesic}}
   \rightarrow
   K
   \rightarrow
   Z.

Thus, Isomap uses a distance-based representation and attempts to
preserve global manifold geometry.

Laplacian Eigenmaps
~~~~~~~~~~~~~~~~~~~

Laplacian Eigenmaps constructs a local neighborhood graph:

.. math::


   X
   \rightarrow
   W
   \rightarrow
   L.

It then solves an eigenvalue problem involving the graph Laplacian.

The eigenvectors corresponding to the smallest non-zero eigenvalues
provide the embedding.

t-SNE
~~~~~

t-SNE converts high-dimensional distances into probabilities:

.. math::


   X
   \rightarrow
   P.

It then learns low-dimensional coordinates whose probability
distribution :math:`Q` approximately matches :math:`P`:

.. math::


   P
   \rightarrow
   Q
   \rightarrow
   Z.

The objective is based on KL divergence:

.. math::


   \operatorname{KL}(P\|Q).

.. _autoencoder-4:

UMAP
~~~~

UMAP constructs a weighted nearest-neighbor graph:

.. math::


   X
   \rightarrow
   W.

It then optimizes low-dimensional coordinates using a cross-entropy-type
objective:

.. math::


   W
   \rightarrow
   Z.


Autoencoder
~~~~~~~~~~~

An autoencoder takes a different approach:

.. math::


   X
   \rightarrow
   \text{Encoder}
   \rightarrow
   Z
   \rightarrow
   \text{Decoder}
   \rightarrow
   \hat{X}.

Instead of explicitly constructing an :math:`N\times N` similarity
matrix and then performing an eigenvalue decomposition or probability
matching procedure, the model **learns the encoder directly through
reconstruction**.

This is an important conceptual difference.

The Bigger Picture
~~~~~~~~~~~~~~~~~~

Many classical dimensionality reduction methods can be viewed as:

.. math::


   \boxed{
   \text{Define relationships}
   \rightarrow
   \text{Construct mathematical representation}
   \rightarrow
   \text{Find low-dimensional coordinates}
   }

For example:

+---------------------+----------------------+----------------------+
| Method              | High-dimensional     | How :math:`Z` is     |
|                     | representation       | obtained             |
+=====================+======================+======================+
| PCA                 | Covariance / Gram    | Eigenvectors / SVD   |
|                     | structure            |                      |
+---------------------+----------------------+----------------------+
| Isomap              | Geodesic distances   | MDS                  |
+---------------------+----------------------+----------------------+
| Laplacian Eigenmaps | Graph weights /      | Eigenvectors         |
|                     | Laplacian            |                      |
+---------------------+----------------------+----------------------+
| t-SNE               | Pairwise             | KL optimization      |
|                     | probabilities        |                      |
+---------------------+----------------------+----------------------+
| UMAP                | Fuzzy neighborhood   | Cross-entropy        |
|                     | graph                | optimization         |
+---------------------+----------------------+----------------------+
| Autoencoder         | Reconstruction       | Neural network       |
|                     | objective            | training             |
+---------------------+----------------------+----------------------+

This gives us a useful conceptual transition:

.. math::


   \boxed{
   \text{Classical DR}
   \rightarrow
   \text{explicit mathematical structure}
   }

while

.. math::


   \boxed{
   \text{Autoencoder}
   \rightarrow
   \text{learn the representation}
   }

Why Use an Autoencoder?
~~~~~~~~~~~~~~~~~~~~~~~

An autoencoder becomes especially useful when the data contains complex
non-linear structure.

For example, consider images.

An image may contain:

-  edges,
-  shapes,
-  textures,
-  objects,
-  lighting variations,
-  spatial relationships.

A simple linear projection may not capture these relationships
efficiently.

A deep autoencoder can learn hierarchical representations:

.. math::


   \mathbf{x}
   \rightarrow
   \mathbf{h}_1
   \rightarrow
   \mathbf{h}_2
   \rightarrow
   \mathbf{z}.

The decoder then attempts to reconstruct the image:

.. math::


   \mathbf{z}
   \rightarrow
   \mathbf{h}_3
   \rightarrow
   \mathbf{h}_4
   \rightarrow
   \hat{\mathbf{x}}.

The latent vector :math:`\mathbf{z}` can therefore capture a compact
representation of the important information needed for reconstruction.

Important Limitation of Autoencoders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A smaller latent dimension does **not automatically** guarantee that the
latent space is meaningful.

For example, an autoencoder could learn a representation that
reconstructs the input well but has:

-  poor interpretability,
-  irregular geometry,
-  discontinuities,
-  sensitivity to training initialization,
-  entangled latent dimensions.

The quality of the latent space depends on:

-  architecture,
-  latent dimension,
-  training data,
-  reconstruction loss,
-  regularization,
-  activation functions,
-  optimization.

This is why specialized autoencoders such as **variational autoencoders
(VAEs)** introduce additional constraints on the latent space.

PCA vs. Autoencoder: Core Intuition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A useful way to remember the difference is:

PCA asks:
~~~~~~~~~

   **What linear directions explain the most variance?**

Mathematically:

.. math::


   X
   \rightarrow
   \text{principal directions}
   \rightarrow
   Z.

Autoencoder asks:
~~~~~~~~~~~~~~~~~

   **What compressed representation allows a neural network to
   reconstruct the input?**

Mathematically:

.. math::


   X
   \rightarrow
   \text{encoder}
   \rightarrow
   Z
   \rightarrow
   \text{decoder}
   \rightarrow
   \hat{X}.

Therefore:

.. math::


   \boxed{
   \text{PCA: find the best linear subspace}
   }

and

.. math::


   \boxed{
   \text{Autoencoder: learn a representation useful for reconstruction}
   }

2D Latent Space Example
~~~~~~~~~~~~~~~~~~~~~~~

Suppose we have a dataset containing images represented by 784 pixels:

.. math::


   \mathbf{x}\in\mathbb{R}^{784}.

We want to visualize the dataset using two dimensions.

We can use PCA:

.. math::


   \mathbb{R}^{784}
   \xrightarrow{\text{PCA}}
   \mathbb{R}^{2}.

The result is:

.. math::


   \mathbf{x}_i
   \rightarrow
   (z_{i1},z_{i2}).

We can plot:

.. math::


   z_{i1}
   \quad\text{vs.}\quad
   z_{i2}.

Alternatively, we can train an autoencoder:

.. math::


   \mathbb{R}^{784}
   \xrightarrow{\text{Encoder}}
   \mathbb{R}^{2}
   \xrightarrow{\text{Decoder}}
   \mathbb{R}^{784}.

The 2D latent coordinates can again be plotted.

However, the resulting plots can look quite different because the two
methods optimize different objectives.

Summary
~~~~~~~

The main differences between PCA and autoencoders are:

1.  **PCA is linear.**

    .. math::


       Z=XW

2.  **Autoencoders can be non-linear.**

    .. math::


       Z=f_{\theta}(X)

3.  **PCA uses eigenvectors or SVD** to find principal directions.

4.  **Autoencoders use neural-network optimization** to learn an encoder
    and decoder.

5.  **PCA components are orthogonal** and ordered by explained variance.

6.  **Autoencoder latent dimensions are not necessarily orthogonal** and
    do not have a fixed interpretation.

7.  **PCA provides the optimal linear reconstruction subspace** under
    the standard squared-error formulation.

8.  **A sufficiently simple linear autoencoder can recover the same
    principal subspace as PCA.**

9.  **A non-linear autoencoder can represent non-linear manifolds** that
    cannot be represented efficiently by a linear PCA subspace.

10. A 2D latent space from PCA and a 2D latent space from an autoencoder
    may therefore have very different geometry and interpretation.

The key conceptual distinction is:

.. math::


   \boxed{
   \text{PCA}
   =
   \text{linear dimensionality reduction}
   }

while

.. math::


   \boxed{
   \text{Autoencoder}
   =
   \text{learned representation through reconstruction}
   }

and, when non-linear neural-network layers are used,

.. math::


   \boxed{
   \text{Autoencoder}
   =
   \text{non-linear dimensionality reduction}
   }
