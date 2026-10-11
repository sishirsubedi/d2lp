# Matrix Factorization
:label:`sec_pca`

Matrix factorization sits at the heart of modern data science, high-dimensional statistics, and machine learning. At its core, the technique addresses a fundamental curse of dimensionality: how to take a massive, complex, and often sparse matrix of data and decompose it into simpler, lower-dimensional components without losing the essential signals hidden inside.

Whether you are compressing images, recommending movies to millions of users, or unspooling complex genetic networks, matrix factorization methods transform entangled observations into meaningful latent factors.


- Linear & Spectral Methods: Techniques like PCA, SVD, and Factor Analysis rely on linear projections and eigenvalue decompositions to find orthogonal axes of maximum variance or shared latent drivers. Laplacian Eigenmaps extends this philosophy into manifold learning by preserving local graph connectivity.

- Non-Linear & Manifold Techniques: t-SNE and UMAP tackle the limits of linear dimensionality reduction, using probabilistic neighborhoods and fuzzy simplicial sets to preserve complex, non-linear structures in low-dimensional visualization spaces.

- Neural Architectures: Autoencoders generalize matrix factorization into deep learning, allowing neural networks to learn flexible, non-linear compressed representations through bottleneck layers.

```toc
:maxdepth: 2

pca
laplacian
twodim
autoencoder
factor
```
