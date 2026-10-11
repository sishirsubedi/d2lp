
.. _sec_laplacian:

Laplacian
=========


Step-by-Step Isomap Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Step 1: Construct the Neighborhood Graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Compute the pairwise distances between data points.
-  Identify the k-nearest neighbors for each point.
-  Construct the graph with nodes representing data points and edges
   representing distances to the k-nearest neighbors.

Step 2: Compute Geodesic Distances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Use the shortest path algorithm (e.g., Floyd-Warshall) to compute the
   shortest paths between all pairs of points.

Step 3: Apply Classical MDS i.e. eigenvectors of the centered Gram matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Center the geodesic distance matrix.
-  Compute the eigenvalues and eigenvectors of the centered Gram matrix.
-  Select the top eigenvectors corresponding to the largest positive
   eigenvalues to get the lower-dimensional embedding.

Centering the Data in MDS
~~~~~~~~~~~~~~~~~~~~~~~~~

In the context of MDS, we start with a distance matrix :math:`D`
containing the pairwise distances between points. We aim to convert this
distance matrix into an inner product matrix :math:`K` (also known as
the Gram matrix), which is centered. This process involves double
centering, which is achieved using the centering matrix :math:`H`.

The centering matrix :math:`H` is defined as:

.. math:: H = I - \frac{1}{n} \mathbf{1}\mathbf{1}^T

where :math:`I` is the :math:`n \times n` identity matrix,
:math:`\mathbf{1}` is an :math:`n \times 1` vector of ones, and
:math:`n` is the number of data points.

Double Centering Formula
~~~~~~~~~~~~~~~~~~~~~~~~

Given the distance matrix :math:`D`, the goal is to obtain the centered
inner product matrix :math:`K` using the formula:

.. math:: K = -\frac{1}{2} H D^2 H

where :math:`D^2` is the element-wise squared distance matrix.

The centering matrix :math:`H` ensures that the mean of each row and
each column of the resulting Gram matrix is zero. This is necessary for
recovering the coordinates in classical MDS.

Centering vs. Standardizing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  **Centering**: Centering involves subtracting the mean of each column
   from the data. In matrix form, for a data matrix :math:`X`, centering
   can be expressed as:

   .. math:: X_{\text{centered}} = HX

   where

   .. math:: H = I - \frac{1}{n}\mathbf{1}\mathbf{1}^T.

-  **Standardizing**: Standardizing involves subtracting the mean and
   dividing by the standard deviation of each column. This transforms
   the data to have zero mean and unit variance:

   .. math:: X_{\text{standardized}} = \frac{X - \text{mean}(X)}{\text{std}(X)}

Why Use Centering in MDS
~~~~~~~~~~~~~~~~~~~~~~~~

In MDS, we are concerned with distances and the resulting geometry of
the data. Double centering converts the squared distance matrix into a
centered Gram matrix. This is essential for recovering the coordinates
in lower-dimensional space.

Standardizing is not part of classical MDS because it changes the scale
of the original features and therefore changes the distances.

When using dimension reduction techniques like PCA (Principal Component
Analysis) and Isomap, both methods involve the use of similarity or Gram
matrices, but they do so in different ways and for different purposes.
Here’s a comparison:

ISOMAP comparison with PCA (Principal Component Analysis)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similarity Matrix / Gram Matrix in PCA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Formation**:

   -  The data matrix :math:`X` is centered by subtracting the mean of
      each feature.
   -  The covariance matrix :math:`\Sigma` is then calculated as
      :math:`\Sigma = \frac{1}{n-1} X^T X`.
   -  Alternatively, the Gram matrix :math:`K` can be used, which is
      given by :math:`K = X X^T`.

2. **Purpose**:

   -  The covariance matrix captures the linear relationships between
      features.
   -  Eigenvalue decomposition of the covariance matrix is performed to
      find the principal components, which are the eigenvectors
      corresponding to the largest eigenvalues.

3. **Interpretation**:

   -  The Gram matrix in PCA reflects the inner products of the centered
      data points.
   -  It provides information about the variance and linear structure of
      the data.

Isomap (Isometric Mapping)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Similarity Matrix / Gram Matrix in Isomap
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Formation**:

   -  Compute the pairwise Euclidean distances between all data points.
   -  Construct a neighborhood graph where each data point is connected
      to its k-nearest neighbors.
   -  Use the shortest path algorithm (e.g., Floyd-Warshall) to compute
      the geodesic distances between all pairs of points on this graph.
   -  Form the geodesic distance matrix :math:`D`.
   -  Double-center the squared geodesic distance matrix to obtain the
      Gram matrix :math:`K = -\frac{1}{2} H D^2 H`, where :math:`H` is
      the centering matrix defined as
      :math:`H = I - \frac{1}{n} \mathbf{1} \mathbf{1}^T`.

2. **Purpose**:

   -  The Gram matrix in Isomap captures the intrinsic geometric
      relationships of the data manifold.
   -  Eigenvalue decomposition of this Gram matrix is used to embed the
      data in a lower-dimensional space while preserving the geodesic
      distances.

3. **Interpretation**:

   -  The Gram matrix in Isomap reflects the intrinsic geometry of the
      data, focusing on preserving the manifold structure.
   -  It provides a measure of similarity based on geodesic distances,
      rather than direct Euclidean distances.

Comparison
~~~~~~~~~~

-  **Linear vs. Non-linear**:

   -  **PCA**: Assumes that the data can be well-approximated by a
      linear subspace and captures directions of maximum variance.
   -  **Isomap**: Captures non-linear relationships by preserving
      geodesic distances on a manifold.

-  **Type of Similarity**:

   -  **PCA**: Similarity is based on covariance or inner products in
      the feature space.
   -  **Isomap**: Similarity is based on geodesic distances, which
      reflect the shortest paths on the manifold.

-  **Data Structure**:

   -  **PCA**: Suitable for data that can be well-represented in a
      linear subspace.
   -  **Isomap**: Suitable for data with complex, non-linear manifold
      structures.

-  **Matrix Computation**:

   -  **PCA**: Computes the covariance matrix or Gram matrix directly
      from the centered data matrix.
   -  **Isomap**: Computes a geodesic distance matrix from a
      neighborhood graph and then converts it into a Gram matrix using
      double-centering.

Summary
~~~~~~~

Both PCA and Isomap use Gram matrices, but PCA’s matrix is based on
direct pairwise inner products (or covariance), focusing on linear
relationships and variance. In contrast, Isomap’s matrix is derived from
geodesic distances, emphasizing non-linear relationships and the
manifold’s intrinsic geometry. These differences make PCA more suitable
for data that is well-approximated by a linear subspace, while Isomap is
useful for capturing and preserving the structure of non-linear
manifolds.

Isomap and Laplacian Eigenmaps are both non-linear dimension reduction
techniques that preserve the intrinsic geometry of the data, but they do
so in different ways. Here is a detailed comparison and contrast between
the two methods:

.. _summary-1:

ISOMAP comparison with Laplacian Eigenmaps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Objective**:

   -  **Isomap**: Focuses on preserving global structure by
      approximating geodesic distances.
   -  **Laplacian Eigenmaps**: Focuses on preserving local structure by
      maintaining local neighborhood relationships.

2. **Methodology**:

   -  **Isomap**: Utilizes geodesic distances and classical MDS.
   -  **Laplacian Eigenmaps**: Utilizes the graph Laplacian and
      eigenvalue decomposition.

3. **Distance Metric**:

   -  **Isomap**: Geodesic distances on the manifold.
   -  **Laplacian Eigenmaps**: Local neighborhood similarities
      represented by the graph weights and graph Laplacian.

4. **Computational Complexity**:

   -  **Isomap**: Higher computational complexity due to shortest path
      calculations.
   -  **Laplacian Eigenmaps**: Generally more computationally efficient
      for sparse neighborhood graphs, since it focuses on local
      structure and the graph Laplacian.

5. **Strengths**:

   -  **Isomap**: Better at capturing global manifold structure when
      geodesic distances are estimated accurately.
   -  **Laplacian Eigenmaps**: Better at preserving local neighborhood
      structure.

6. **Weaknesses**:

   -  **Isomap**: Sensitive to the choice of neighborhood size and the
      accuracy of the estimated geodesic distances.
   -  **Laplacian Eigenmaps**: May not preserve global distances or the
      global manifold structure as effectively as Isomap.


Summary
~~~~~~~

-  **Isomap** is preferable when the goal is to capture and preserve the
   global structure of a complex, non-linear manifold. It uses geodesic
   distances followed by classical MDS.
-  **Laplacian Eigenmaps** are more suitable when the focus is on
   preserving local relationships. They construct a neighborhood graph
   and find an embedding that keeps nearby points close together.

Both methods have their unique strengths and are chosen based on the
specific requirements of the data and the analysis objectives.

Laplacian Eigenmaps Explanation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Step-by-Step Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Laplacian Eigenmaps is a popular technique for non-linear dimensionality
reduction that focuses on preserving local geometric relationships
between data points. Here is a detailed explanation of the Laplacian
Eigenmaps algorithm:

Objective
~~~~~~~~~

The main objective of Laplacian Eigenmaps is to preserve the local
neighborhood structure of the data when projecting it to a
lower-dimensional space. It does this by constructing a graph where the
nodes represent data points and the edges represent the similarities
between the points. The method then uses the properties of the graph to
find a low-dimensional embedding.

Steps of the Algorithm
~~~~~~~~~~~~~~~~~~~~~~

1. Construct the Neighborhood Graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Input**: Data matrix :math:`X` with :math:`n` data points and
   :math:`m` features.
2. **Neighborhood Definition**: Determine the neighbors for each data
   point using either **k-nearest neighbors** (each point is connected
   to its :math:`k` nearest neighbors) or **:math:`\epsilon`-ball**
   (each point is connected to all points within a distance
   :math:`\epsilon`).
3. **Weight Matrix :math:`W`**: Define a weight matrix :math:`W` where
   :math:`W_{ij}` represents the weight (similarity) between points
   :math:`i` and :math:`j`. Common choices include:

   -  **Binary weights**: :math:`W_{ij} = 1` if :math:`j` is a neighbor
      of :math:`i`, otherwise :math:`W_{ij} = 0`. *(Note: Unlike PCA and
      Isomap where pairwise values for all points are calculated, here
      only k-neighbor pairwise values are calculated).*
   -  **Heat kernel weights**:
      :math:`W_{ij} = \exp\left(-\frac{\Vert{}x_i - x_j\Vert{}^2}{2\sigma^2}\right)`
      where :math:`\sigma` controls the width of the Gaussian kernel.

In the context of the heat kernel used in Laplacian Eigenmaps, the
weight :math:`w_{ij}` is calculated as:

.. math:: w_{ij} = \exp\left(-\frac{\Vert{}x_i - x_j\Vert{}^2}{2\sigma^2}\right)

Here is how the weights behave in terms of distance:

1. **Small Distance (:math:`\Vert{}x_i - x_j\Vert{}` is small):** When
   the distance between :math:`x_i` and :math:`x_j` is very small,
   :math:`w_{ij} \approx \exp(0) = 1`. This indicates that the points
   are very similar or close.
2. **Large Distance (:math:`\Vert{}x_i - x_j\Vert{}` is large):** When
   the distance between :math:`x_i` and :math:`x_j` is large,
   :math:`w_{ij}` approaches zero. Therefore, distant points have very
   small weights.

2. Compute the Graph Laplacian
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Degree Matrix :math:`D`**: Construct the degree matrix :math:`D`,
   which is a diagonal matrix where :math:`D_{ii} = \sum_j W_{ij}`.
2. **Laplacian Matrix :math:`L`**: Compute the unnormalized Laplacian
   matrix :math:`L` as :math:`L = D - W`. Alternatively, the normalized
   Laplacian matrices are
   :math:`L_{\text{sym}} = I - D^{-\frac{1}{2}} W D^{-\frac{1}{2}}` or
   :math:`L_{\text{rw}} = I - D^{-1} W`.

3. Solve the Eigenvalue Problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Eigenvalue Decomposition**: Solve the generalized eigenvalue
   problem :math:`L y = \lambda D y`.
2. **Select Eigenvectors**: Choose the eigenvectors corresponding to the
   smallest non-zero eigenvalues. The smallest eigenvalue
   :math:`\lambda_0 = 0` corresponds to the trivial solution and is
   ignored. The next :math:`d` smallest eigenvalues provide the
   coordinates of the lower-dimensional embedding.

4. Output the Embedding
^^^^^^^^^^^^^^^^^^^^^^^

The lower-dimensional representation of the data is given by the matrix
:math:`Y = [y_1, y_2, \ldots, y_d]` formed from the selected
eigenvectors.

Synthesis: Largest vs. Smallest Eigenvalues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  **PCA**: High eigenvalues indicate directions of significant variance
   (important structure).
-  **ISOMAP**: The largest positive eigenvalues of the centered Gram
   matrix correspond to the principal embedding directions. Small or
   negative eigenvalues may indicate that the estimated geodesic
   distances do not correspond exactly to a Euclidean geometry.
-  **Laplacian Eigenmaps**: Small eigenvalues correspond to smooth
   variations over the graph and therefore preserve local neighborhood
   structure, while large eigenvalues correspond to higher-frequency
   variations across the graph.

Similarity/Gram Matrix in Laplacian Eigenmaps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In Laplacian Eigenmaps, the weight matrix :math:`W` plays a role
analogous to the similarity matrix, and the graph Laplacian :math:`L` is
derived from :math:`W`. This contrasts with Isomap, which first
calculates geodesic distances and then uses these to form a Gram matrix.

Objective Function Derivation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The original objective function in Laplacian Eigenmaps is:

.. math:: \sum_{i,j} W_{ij} \Vert{} \mathbf{z}_i - \mathbf{z}_j \Vert{}^2

Expanding the squared Euclidean distance:

.. math:: \Vert{} \mathbf{z}_i - \mathbf{z}_j \Vert{}^2 = \mathbf{z}_i^T \mathbf{z}_i + \mathbf{z}_j^T \mathbf{z}_j - 2 \mathbf{z}_i^T \mathbf{z}_j

Summing over all pairs :math:`(i,j)` and combining terms using the
degree matrix :math:`D_{ii} = \sum_j W_{ij}`:

.. math:: \sum_{i,j} W_{ij} \Vert{} \mathbf{z}_i - \mathbf{z}_j \Vert{}^2 = 2 \left( \sum_i D_{ii} \mathbf{z}_i^T \mathbf{z}_i - \sum_{i,j} W_{ij} \mathbf{z}_i^T \mathbf{z}_j \right)

.. math:: = 2 \operatorname{tr}(Z^T L Z)

where :math:`Z` is the matrix whose rows are the embedded points
:math:`\mathbf{z}_i^T`.

By minimizing :math:`\operatorname{tr}(Z^T L Z)` subject to the
normalization constraint

.. math:: Z^T D Z = I,

we obtain the generalized eigenvalue problem

.. math:: L y = \lambda D y.

The eigenvectors corresponding to the smallest non-zero eigenvalues
provide the low-dimensional embedding.
