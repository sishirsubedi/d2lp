# Introduction
:label:`chap_introduction`

### Deep Learning Paradigms: Supervised vs. Unsupervised Learning
Deep learning architectures generally operate under two core paradigms depending on the availability and nature of data labels:

- **Supervised Learning**: Models are trained on paired data consisting of inputs and corresponding target labels (e.g., classification or regression). The network learns a mapping function by minimizing the error between its predictions and the ground-truth labels using optimization techniques like backpropagation.

- **Unsupervised Learning**: Models work with unlabeled data to discover hidden structures, patterns, or underlying distributions (e.g., clustering, dimensionality reduction, or generative modeling like the topic models discussed previously). Instead of optimizing against explicit targets, these models maximize data likelihood or reconstruct input representations.

### Roadmap: Exploring Unsupervised Learning via Latent Space Representations
This book begins its exploration of unsupervised learning through a central lens: latent space representations—how complex, high-dimensional data can be compressed into lower-dimensional, interpretable spaces.

To understand how modern representations evolved, the journey follows a deliberate trajectory from linear algebra to probabilistic and neural text modeling:

- **Linear Matrix Factorization (PCA & SVD)**: We establish the geometric foundations using Principal Component Analysis and Singular Value Decomposition to capture global variance in continuous data.

- **Algebraic Text Retrieval (LSA)**: We adapt matrix factorization to text by applying SVD to term-document matrices, introducing Latent Semantic Analysis.

- **Probabilistic Topic Modeling (pLSA & LDA)**: We transition from deterministic geometry to generative probabilistic models, viewing documents as mixtures of latent topics using Latent Dirichlet Allocation.

- **Advanced Inference and Neural Extensions (LDA-VI & ETM)**: We tackle complex posterior estimation using Variational Inference (VI) and modernize topic modeling by integrating deep learning word embeddings through the Embedded Topic Model (ETM).