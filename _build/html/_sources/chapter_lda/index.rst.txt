
.. _chap_pmf:

Probabilistic Matrix Factorization
==================================


Earlier, we explored linear dimensionality reduction techniques like PCA
and SVD, which project data into lower-dimensional spaces by capturing
global geometric variance and linear correlations. While powerful for
continuous data, these algebraic methods treat documents as simple
vectors and often struggle to capture the polysemy (multiple meanings)
and probabilistic nature of natural language.This chapter transitions
from linear algebra to probabilistic and neural text modeling. We begin
with LSA (Latent Semantic Analysis), which applies SVD to term-document
matrices, and then move into generative probabilistic models—pLSA and
LDA (Latent Dirichlet Allocation)—which view documents as mixtures of
latent topics using Bayesian inference and Variational Inference (VI).
Finally, we bridge traditional topic modeling with modern deep learning
by exploring ETM (Embedded Topic Models), which combine word embeddings
with probabilistic topic structures to handle sparse and large-scale
vocabularies.

.. toctree::
   :maxdepth: 2

   lsa
   plsa
   lda_1
   lda_2
   lda_vi
   lda_etm
