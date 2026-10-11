
.. _sec_lda_vi:

LDA VI
======


Latent Dirichlet Allocation (LDA) and Variational Inference (VI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Latent Dirichlet Allocation (LDA) is a generative probabilistic model
used to discover topics within a collection of documents. The central
idea of LDA is that documents are mixtures of topics, and each topic is
a distribution over words. Variational Inference (VI) is an approach
used to approximate the posterior distributions of the latent variables
in LDA, given the observed data.

Core Concepts in LDA
~~~~~~~~~~~~~~~~~~~~

Document-Topic Distribution (:math:`\theta_i`):

For each document :math:`i`, :math:`\theta_i` is a distribution over
:math:`K` topics, typically drawn from a Dirichlet distribution
parameterized by :math:`\gamma_i`.

Topic-Word Distribution (:math:`\beta_k`):

For each topic :math:`k`, :math:`\beta_k` is a distribution over the
vocabulary, typically drawn from a Dirichlet distribution parameterized
by :math:`\lambda_k`.

Topic Assignment (:math:`z_{ij}`):

For each word :math:`w_{ij}` in document :math:`i`, :math:`z_{ij}` is a
latent variable indicating the topic assigned to that word.

Word Generation (:math:`w_{ij}`):

Given the topic assignment :math:`z_{ij}`, the word :math:`w_{ij}` is
drawn from the topic-word distribution :math:`\beta_{z_{ij}}`.

Variational Inference (VI) in LDA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In VI, we approximate the intractable posterior
:math:`p(z, \theta, \beta \mid w)` with a variational distribution
:math:`q(z, \theta, \beta)` that factorizes as follows:

.. math::


   q(z, \theta, \beta)
   =
   \prod_{i=1}^{D} q(\theta_i \mid \gamma_i)
   \prod_{k=1}^{K} q(\beta_k \mid \lambda_k)
   \prod_{i=1}^{D}
   \prod_{j=1}^{N_i}
   q(z_{ij} \mid \phi_{ij})

Here:

:math:`\gamma_i` are the variational parameters for the Dirichlet
distribution over topics for document :math:`i`.

:math:`\lambda_k` are the variational parameters for the Dirichlet
distribution over words for topic :math:`k`.

:math:`\phi_{ijk}` are the variational parameters for the categorical
distribution over topics for word :math:`w_{ij}`.

Evidence Lower Bound (ELBO)

The ELBO is the objective function that VI seeks to maximize. It is
given by:

.. math::


   \mathrm{ELBO}
   =
   \mathbb{E}_q[\log p(w \mid z, \beta)]
   +
   \mathbb{E}_q[\log p(z \mid \theta)]
   +
   \mathbb{E}_q[\log p(\theta \mid \alpha)]
   +
   \mathbb{E}_q[\log p(\beta \mid \eta)]
   -
   \mathbb{E}_q[\log q(z, \theta, \beta)]

This can be broken down into:

Likelihood Term: :math:`\mathbb{E}_q[\log p(w \mid z, \beta)]` captures
how well the model explains the observed words given the topic
assignments and topic-word distributions.

Topic Assignment Prior: :math:`\mathbb{E}_q[\log p(z \mid \theta)]`
represents the prior belief about topic assignments given the
document-topic distributions.

Document-Topic Prior: :math:`\mathbb{E}_q[\log p(\theta \mid \alpha)]`
represents the prior belief about document-topic distributions.

Topic-Word Prior: :math:`\mathbb{E}_q[\log p(\beta \mid \eta)]`
represents the prior belief about topic-word distributions.

Entropy Term: :math:`-\mathbb{E}_q[\log q(z, \theta, \beta)]` encourages
the variational distribution :math:`q` to remain close to the true
posterior.

Update Formulas in VI

Update for :math:`\phi_{ijk}` (Soft Topic Assignment)

.. math::


   \phi_{ijk}
   \propto
   \exp\left\{
   \mathbb{E}_q[\log \theta_{ik}]
   +
   \mathbb{E}_q[\log \beta_{k w_{ij}}]
   \right\}

Normalization:

.. math::


   \phi_{ijk}
   =
   \frac{
   \exp\left\{
   \mathbb{E}_q[\log \theta_{ik}]
   +
   \mathbb{E}_q[\log \beta_{k w_{ij}}]
   \right\}
   }{
   \displaystyle
   \sum_{k'=1}^{K}
   \exp\left\{
   \mathbb{E}_q[\log \theta_{ik'}]
   +
   \mathbb{E}_q[\log \beta_{k' w_{ij}}]
   \right\}
   }

This ensures that :math:`\phi_{ijk}` sums to 1 across all topics for
each word.

Update for :math:`\gamma_{ik}` (Document-Topic Distribution)

.. math::


   \gamma_{ik}
   =
   \alpha_k
   +
   \sum_{j=1}^{N_i}
   \phi_{ijk}

Here, :math:`\gamma_{ik}` is updated by adding the Dirichlet prior
:math:`\alpha_k` and the expected number of times topic :math:`k` is
assigned to words in document :math:`i`.

Update for :math:`\lambda_{kw}` (Topic-Word Distribution)

.. math::


   \lambda_{kw}
   =
   \eta
   +
   \sum_{i=1}^{D}
   \sum_{j=1}^{N_i}
   \phi_{ijk}
   \delta(w_{ij}=w)

Here, :math:`\lambda_{kw}` is updated by adding the Dirichlet prior
:math:`\eta` and the expected number of times word :math:`w` is
associated with topic :math:`k` across all documents.

No need to normalize for :math:`\theta_{ik}` and :math:`\beta_{wk}`
because the Dirichlet distribution will normalize them when used as
parameters.

Final Inference

After variational parameters :math:`\phi_{ijk}`, :math:`\gamma_{ik}`,
and :math:`\lambda_{kw}` have been updated iteratively to convergence,
they represent the approximate posterior distributions over the latent
variables. The :math:`\gamma_{ik}` values give the inferred
document-topic distribution for each document, while
:math:`\lambda_{kw}` informs the inferred word distributions for each
topic.

For predicting the topic distribution for a new document:

Initialize :math:`\theta_{\text{new}}` randomly or uniformly.

Iteratively update :math:`\theta_{\text{new}}` using the same
variational approach, but with the previously learned :math:`\beta_{kw}`
values.

Conclusion
~~~~~~~~~~

LDA, combined with VI, provides a powerful framework for uncovering
latent topic structures within a corpus. The ELBO serves as the
objective function that guides the optimization process, ensuring that
the variational parameters converge to values that approximate the true
posterior distributions as closely as possible. Through the iterative
update of variational parameters, LDA can effectively model the complex,
overlapping themes present in large text datasets, making it an
essential tool in natural language processing.

Certainly! Let’s walk through Latent Dirichlet Allocation (LDA),
explaining it in a similar way to Gaussian Mixture Models (GMM) and
focusing on the soft assignment of words to topics, along with the
update equations used in Variational Inference (VI).

Overview of LDA and Soft Assignments

Just like a GMM, where data points (e.g., samples) are assigned
probabilistically to clusters, LDA assigns words within documents to
topics softly. In LDA, each document is a mixture of topics, and each
word is probabilistically assigned to a topic. The assignment of words
to topics is not hard (one-hot), but soft, meaning that each word has
some probability of belonging to each topic.

LDA Generative Process (Like GMM Generative Process)

For each document in LDA:

A distribution over topics :math:`\theta_i` is drawn from a Dirichlet
distribution with hyperparameters :math:`\alpha`:

.. math::


   \theta_i \sim \operatorname{Dirichlet}(\alpha)

:math:`\theta_i` is a :math:`K`-dimensional vector that represents the
proportion of each topic in document :math:`i`.

For each word :math:`j` in document :math:`i`:

A topic :math:`z_{ij}` is drawn from a categorical distribution
parameterized by the topic proportions :math:`\theta_i`:

.. math::


   z_{ij} \sim \operatorname{Categorical}(\theta_i)

:math:`z_{ij}` is the topic assigned to word :math:`j` in document
:math:`i`.

The word :math:`w_{ij}` is then drawn from the topic’s word distribution
:math:`\phi_k`, where :math:`z_{ij}=k` is the topic:

.. math::


   w_{ij} \sim \operatorname{Categorical}(\phi_{z_{ij}})

:math:`\phi_k` is the distribution of words for topic :math:`k`, which
comes from a Dirichlet distribution parameterized by :math:`\beta`:

.. math::


   \phi_k \sim \operatorname{Dirichlet}(\beta)

Key Components in LDA (like GMM)

:math:`\theta_i`: Distribution over topics for document :math:`i`
(analogous to mixture weights :math:`\pi` in GMM).

:math:`z_{ij}`: Latent variable representing the topic assignment for
word :math:`j` in document :math:`i` (analogous to :math:`z_i`, the
component assignment in GMM).

:math:`\phi_k`: Word distribution for topic :math:`k`, where each word
has some probability of being generated from the topic (analogous to the
Gaussian component parameters in GMM).

Inference in LDA: Variational Inference and Soft Assignments

In Variational Inference (VI), just like in GMM, we don’t sample the
latent variables directly. Instead, we compute the posterior
distribution of the latent variables by optimizing a variational
approximation to the true posterior.

Mean-Field Factorization

We approximate the posterior distribution of the latent variables using
a mean-field approximation, which factorizes the joint posterior
distribution into independent variational distributions:

.. math::


   q(\theta_i, z_{ij}, \phi_k)
   =
   q(\theta_i)
   q(z_{ij})
   q(\phi_k)

Where:

:math:`q(\theta_i)` is the variational distribution over the topic
proportions for document :math:`i`.

:math:`q(z_{ij})` is the variational distribution over the topic
assignment for word :math:`j` in document :math:`i`.

:math:`q(\phi_k)` is the variational distribution over the word
distributions for each topic.

Soft Assignments in LDA (Like Soft Assignments in GMM)

1. Variational Distribution for :math:`z_{ij}` (Soft Assignment)

In LDA, each word :math:`w_{ij}` has a soft assignment to each of the
:math:`K` topics. Instead of sampling :math:`z_{ij}`, we compute the
variational responsibilities :math:`\phi_{ijk}`, which represent the
probability that word :math:`j` in document :math:`i` belongs to topic
:math:`k`. These are analogous to the soft assignments
:math:`\gamma_{z_{ik}}` in GMM.

The variational distribution :math:`q(z_{ij})` is a categorical
distribution with parameters :math:`\phi_{ijk}`, which represent the
soft assignments:

.. math::


   q(z_{ij}=k)=\phi_{ijk}

Where :math:`\phi_{ijk}` represents the probability that word :math:`j`
in document :math:`i` is assigned to topic :math:`k`.

2. Variational Distribution for :math:`\theta_i` (Document-Topic
   Proportions)

For each document, the topic proportions :math:`\theta_i` are drawn from
a Dirichlet distribution. In VI, we approximate the posterior
distribution over :math:`\theta_i` with a variational distribution
:math:`q(\theta_i)`, which is also a Dirichlet distribution:

.. math::


   q(\theta_i)
   =
   \operatorname{Dirichlet}(\gamma_i)

Where :math:`\gamma_i` is a :math:`K`-dimensional vector representing
the variational parameters for the topic proportions in document
:math:`i`.

3. Variational Distribution for :math:`\phi_k` (Topic-Word Distribution)

For each topic :math:`k`, the word distribution :math:`\phi_k` is drawn
from a Dirichlet distribution:

.. math::


   q(\phi_k)
   =
   \operatorname{Dirichlet}(\lambda_k)

Where :math:`\lambda_k` is a vector of length :math:`V` (vocabulary
size), representing the variational parameters for the word distribution
for topic :math:`k`.

Update Equations in Variational Inference for LDA

Update for :math:`q(z_{ij})` (Soft Assignment)

The soft assignment of word :math:`w_{ij}` to each topic :math:`k` is
updated based on the following equation:

.. math::


   \phi_{ijk}
   \propto
   \exp\left(
   \mathbb{E}_{q(\theta_i)}[\log \theta_{ik}]
   +
   \mathbb{E}_{q(\phi_k)}[\log \phi_{k w_{ij}}]
   \right)

Where:

:math:`\mathbb{E}_{q(\theta_i)}[\log \theta_{ik}]` is the expected log
probability of topic :math:`k` in document :math:`i`.

:math:`\mathbb{E}_{q(\phi_k)}[\log \phi_{k w_{ij}}]` is the expected log
probability of word :math:`w_{ij}` in topic :math:`k`.

This is analogous to how responsibilities :math:`r_{ik}` are updated in
GMMs.

Update for :math:`q(\theta_i)` (Document-Topic Proportions)

The variational parameters :math:`\gamma_i` for the topic proportions
:math:`\theta_i` in document :math:`i` are updated as follows:

.. math::


   \gamma_{ik}
   =
   \alpha_k
   +
   \sum_{j=1}^{N_i}
   \phi_{ijk}

Where:

:math:`\alpha_k` is the prior parameter for topic :math:`k`.

:math:`\phi_{ijk}` is the soft assignment of word :math:`w_{ij}` to
topic :math:`k`.

:math:`N_i` is the number of words in document :math:`i`.

This is analogous to the way the mixture weights :math:`\pi_k` are
updated in GMMs.

Update for :math:`q(\phi_k)` (Topic-Word Distribution)

The variational parameters :math:`\lambda_k` for the word distribution
:math:`\phi_k` over topic :math:`k` are updated as follows:

.. math::


   \lambda_{kv}
   =
   \beta_v
   +
   \sum_{i=1}^{D}
   \sum_{j=1}^{N_i}
   \mathbb{1}(w_{ij}=v)
   \phi_{ijk}

Where:

:math:`\beta_v` is the prior parameter for word :math:`v`.

:math:`\mathbb{1}(w_{ij}=v)` is an indicator function that is 1 if
:math:`w_{ij}` is word :math:`v`.

:math:`\phi_{ijk}` is the soft assignment of word :math:`w_{ij}` to
topic :math:`k`.

:math:`D` is the number of documents, and :math:`N_i` is the number of
words in document :math:`i`.

Summary: LDA and Soft Assignments Like GMM

In LDA, the latent variable :math:`z_{ij}` represents the topic
assignment for word :math:`w_{ij}` in document :math:`i`, analogous to
the cluster assignment :math:`z_i` in GMM.

Instead of directly sampling :math:`z_{ij}`, soft assignments
:math:`\phi_{ijk}` (variational responsibilities) are used to represent
the probability that word :math:`w_{ij}` belongs to topic :math:`k`,
similar to the soft assignments in GMM.

The update equations in VI for LDA involve updating these soft
assignments as well as the variational distributions over the
document-topic proportions.
