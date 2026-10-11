
.. _chap_pca:

Principal Component Analysis
============================


We can understand Principal Component Analysis (PCA) using Singular
Value Decomposition (SVD) and relationships between the matrices and the
properties of matrix multiplication.

Given the SVD of matrix :math:`X`:

.. math:: X = U \Sigma V^T

where:

-  :math:`X` is an :math:`m \times n` matrix.
-  :math:`U` is an :math:`m \times m` orthogonal matrix (columns are
   left singular vectors).
-  :math:`\Sigma` is an :math:`m \times n` diagonal matrix with
   non-negative real numbers on the diagonal (singular values).
-  :math:`V` is an :math:`n \times n` orthogonal matrix (columns are
   right singular vectors).
-  :math:`V^T` is the transpose of :math:`V`.

Relationship among X,U,and V
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can start with simple cross-product matrices. Here, we can have two
views - one sample x sample or feature x feature:

.. math:: XX^T = U \Sigma V^T (V \Sigma^T U^T) = U \Sigma \Sigma^T U^T

.. math:: X^T X = V \Sigma^T U^T (U \Sigma V^T) = V \Sigma^T \Sigma V^T

From the above, we can write:

.. math:: XX^T U = U \Sigma \Sigma^T

.. math:: X^T X V = V \Sigma^T \Sigma

Given :math:`XX^T U = U \Sigma \Sigma^T`, we can rearrange this
equation:

.. math:: XX^T U = U \Sigma \Sigma^T

Multiplying both sides by :math:`U^T`:

.. math:: U^T XX^T U = \Sigma \Sigma^T

Thus:

.. math:: U^T X = \Sigma V^T

Taking the transpose:

.. math:: X^T U = V \Sigma^T

Now, multiplying both sides by :math:`\left(\Sigma^T\right)^{-1}` from
the right:

.. math:: X^T U \left(\Sigma^T\right)^{-1} = V

Therefore:

.. math:: V = X^T U \left(\Sigma^T\right)^{-1}

Similarly, given :math:`X^T X V = V \Sigma^T \Sigma`, we can rearrange
this equation:

.. math:: X^T X V = V \Sigma^T \Sigma

Multiplying both sides by :math:`V^T`:

.. math:: V^T X^T X V = \Sigma^T \Sigma

Thus:

.. math:: X V = U \Sigma

Now, multiplying both sides by :math:`\Sigma^{-1}` from the right:

.. math:: X V \Sigma^{-1} = U

Therefore:

.. math:: U = X V \Sigma^{-1}

In summary, we have derived the following relationships:

.. math:: V = X^T U \left(\Sigma^T\right)^{-1}

.. math:: U = X V \Sigma^{-1}

These equations show how the right singular vectors (:math:`V`) can be
expressed in terms of the left singular vectors (:math:`U`) and vice
versa, using the original matrix :math:`X` and the diagonal matrix of
singular values (:math:`\Sigma`).

**DUALITY OF PCA** Both U and V can be learned from other components.

.. math:: X = U \Sigma V^T

.. math:: V = X^T U \left(\Sigma^T\right)^{-1}

.. math:: U = X V \Sigma^{-1}

Interpretation
~~~~~~~~~~~~~~

-  **Projection to Latent Space**: When we project the original data
   :math:`X` onto the latent space defined by :math:`V`, we are
   essentially expressing :math:`X` in terms of its principal directions
   (eigenvectors of :math:`X^T X`). This is the projection :math:`X V`.

-  **Normalization**: The matrix :math:`\Sigma^{-1}` normalizes this
   projection. The resulting product :math:`X V \Sigma^{-1}` gives us
   the matrix :math:`U`, which contains the left singular vectors of
   :math:`X`, or equivalently, the eigenvectors of :math:`XX^T`.

The equation :math:`U = X V \Sigma^{-1}` encapsulates the idea that the
left singular vectors (eigenvectors of :math:`XX^T`) can be expressed as
a normalized projection of the original data :math:`X` onto the right
singular vectors (eigenvectors of :math:`X^T X`). This provides a bridge
between the data space and the latent space, showing how the singular
vectors and the original data matrix are interconnected through the SVD.

Singular Values vs Eigenvalues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  | **Singular Values (:math:`\sigma_i`)**: Obtained from the **SVD**
     of data matrix :math:`X`: :math:`X = U \Sigma V^T`. Represents the
     **scale/stretching** of :math:`X` along principal directions.
   | *Intuition:* Lengths of axes when a unit sphere is transformed by
     :math:`X`.

-  | **Eigenvalues (:math:`\lambda_i`)**: Obtained from the
     **eigendecomposition** of cross-product matrix :math:`X^T X`:
     :math:`X^T X = V \Lambda V^T`. Represents the **squared scale** of
     data along principal components.
   | *Intuition:* Squared lengths of the transformed axes (spread of
     data).

Relationship
~~~~~~~~~~~~

.. math:: \lambda_i = \sigma_i^2

Eigenvalues of :math:`X^T X` are the **squares** of singular values.

For a centered data matrix :math:`X`, if the covariance matrix is

.. math:: C = \frac{1}{m-1}X^T X,

then its eigenvalues are

.. math:: \lambda_i = \frac{\sigma_i^2}{m-1}.

Thus, the eigenvalues of :math:`X^T X` are proportional to the variance
of the data along the principal components.

+-----------------+-----------------------+---------------+-----------+
| Quantity        | From matrix           | Meaning       | Units     |
+=================+=======================+===============+===========+
| Singular value  | :math:`X` (SVD)       | Sca           | Linear    |
|                 |                       | le/stretching | scale     |
+-----------------+-----------------------+---------------+-----------+
| Eigenvalue      | :math:`X^T X`         | Squared scale | Squared   |
|                 |                       |               | linear    |
|                 |                       |               | scale     |
+-----------------+-----------------------+---------------+-----------+
| Covariance      | :math                 | Variance      | Squared   |
| eigenvalue      | :`\frac{1}{m-1}X^T X` | (spread)      | feature   |
|                 |                       |               | units     |
+-----------------+-----------------------+---------------+-----------+

Eigenvector is a direction in feature space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the case where the **data matrix** is :math:`X` and the
**cross-product matrix** is :math:`A = X^T X`, the **eigenvector
:math:`v`** in the equation :math:`A v = \lambda v` is a **direction in
feature space**, not a column of :math:`A`.

**Interpretation as a Feature Combination: Each element in an
eigenvector corresponds to the contribution of each original feature in
that particular direction. Therefore, you can think of an eigenvector as
a “recipe” for combining features in a specific proportion, representing
a dominant pattern in the data.**

Let’s break this down:
~~~~~~~~~~~~~~~~~~~~~~

1. **Cross-Product Matrix :math:`A = X^T X`**:

   -  Here, :math:`X` is your data matrix with dimensions
      :math:`m \times n` (where :math:`m` is the number of samples and
      :math:`n` is the number of features).
   -  The matrix :math:`A = X^T X` is an :math:`n \times n` matrix,
      representing the inner products between features (it’s the **Gram
      matrix** of :math:`X`).

2. **Eigenvectors of :math:`A`**:

   -  The matrix :math:`A` is symmetric and positive semi-definite, so
      it has real, non-negative eigenvalues and corresponding
      eigenvectors.
   -  The eigenvectors :math:`v` of :math:`A` satisfy the equation
      :math:`A v = \lambda v`, where :math:`\lambda` is the eigenvalue
      corresponding to :math:`v`.

3. **Direction in Feature Space**:

   -  :math:`v` is an eigenvector of the matrix :math:`A = X^T X`. Since
      :math:`A` is the product of :math:`X^T` and :math:`X`, the
      eigenvector :math:`v` lies in the **feature space**, meaning it is
      a direction in the space spanned by the features.
   -  The eigenvectors of :math:`A` represent directions in the feature
      space, where the elements of :math:`v` give the coefficients for
      combining the original features. Thus, :math:`v` indicates a
      direction that corresponds to some combination of the original
      features in :math:`X`.

Key Point:
~~~~~~~~~~

-  **Eigenvectors of :math:`A = X^T X`** are **directions in the
   original feature space**. Each element of the eigenvector gives the
   coefficient of the corresponding feature.

Why Not Columns of :math:`A`?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  :math:`A` is the **cross-product matrix** of :math:`X`, and its
   columns represent relationships between features, not the original
   data features themselves. The eigenvectors :math:`v` are calculated
   from the relationships between the features, so they describe
   directions in the feature space of :math:`X`.

-  The eigenvector :math:`v` of :math:`A = X^T X` represents a direction
   in the original feature space, with each component corresponding to
   one of the original features.

ROW space vs COLUMN space
~~~~~~~~~~~~~~~~~~~~~~~~~

The main idea is eigen vectors in feature space i.e. Vs are unit length
and :math:`X` transforms these vectors to another coordinate system
where eigen vectors are sample space i.e. Us and here vectors are scaled
by sigma.

The right singular vectors :math:`V` correspond to directions in the
**row space of :math:`X`**, which is a subspace of the feature space.
The left singular vectors :math:`U` correspond to directions in the
**column space of :math:`X`**, which is a subspace of the sample space.

Understanding SVD: Column Space vs. Row Space, and the Role of Singular Values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the Singular Value Decomposition (SVD), a data matrix
:math:`X \in \mathbb{R}^{n \times p}` is factorized as:

.. math:: X = U \Sigma V^T

where: - :math:`U \in \mathbb{R}^{n \times n}` contains the **left
singular vectors**, which form an orthonormal basis for the **column
space** (sample space) of :math:`X`, -
:math:`V \in \mathbb{R}^{p \times p}` contains the **right singular
vectors**, which form an orthonormal basis for the **row space**
(feature space) of :math:`X`, - :math:`\Sigma` is a diagonal matrix with
**singular values** :math:`\sigma_i`, indicating the strength/scale
along each singular direction.

The **row space** of :math:`X` is related to the **eigenvectors of
:math:`X^T X`** and spans directions in the **feature space**. The
**column space** of :math:`X` corresponds to the **eigenvectors of
:math:`X X^T`** and spans directions in the **sample space**. Although
these spaces live in different dimensions, they are intimately connected
through the SVD — they share the same rank, and their bases are linked
via singular values.

A key insight is that each right singular vector :math:`v_i` (from the
feature space) is a **unit vector**, and when it is transformed by
:math:`X`, the result is a vector :math:`\sigma_i u_i` in the sample
space. That is:

.. math:: X v_i = \sigma_i u_i

So, while both :math:`v_i` and :math:`u_i` are unit vectors in their
respective spaces, the transformation from feature to sample space
scales the magnitude by the singular value :math:`\sigma_i`. This shows
that singular values encode how much scale/energy is present in each
direction. For centered data, :math:`\sigma_i^2` is proportional to the
variance of the data along that direction. The mapping preserves
directionality but stretches the magnitude, effectively turning unit
spheres in feature space into ellipsoids in sample space.

In this way, SVD provides a geometrically meaningful link between the
structure of data in feature space and how it manifests in sample space.

Main Idea
~~~~~~~~~

In SVD, the “core” directions live in feature space. Singular values
determine how important those directions are. Applying the data matrix
:math:`X` to each feature-space direction :math:`v` gives you the
corresponding sample-space direction :math:`u`, scaled by how much
scale/energy is present along :math:`v`.

.. toctree::
   :maxdepth: 2

   laplacian
   twodim
   autoencoder
   factor
