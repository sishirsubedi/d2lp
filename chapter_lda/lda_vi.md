# LDA VI
:label:`sec_lda_vi`

### Latent Dirichlet Allocation (LDA) and Variational Inference (VI)

Latent Dirichlet Allocation (LDA) is a generative probabilistic model used to discover topics within a collection of documents. The central idea of LDA is that documents are mixtures of topics, and each topic is a distribution over words. Variational Inference (VI) is an approach used to approximate the posterior distributions of the latent variables in LDA, given the observed data.

### Core Concepts in LDA

Document-Topic Distribution ($\theta_i$):

For each document $i$, $\theta_i$ is a distribution over $K$ topics, typically drawn from a Dirichlet distribution parameterized by $\gamma_i$.

Topic-Word Distribution ($\beta_k$):

For each topic $k$, $\beta_k$ is a distribution over the vocabulary, typically drawn from a Dirichlet distribution parameterized by $\lambda_k$.

Topic Assignment ($z_{ij}$):

For each word $w_{ij}$ in document $i$, $z_{ij}$ is a latent variable indicating the topic assigned to that word.

Word Generation ($w_{ij}$):

Given the topic assignment $z_{ij}$, the word $w_{ij}$ is drawn from the topic-word distribution $\beta_{z_{ij}}$.

### Variational Inference (VI) in LDA

In VI, we approximate the intractable posterior $p(z, \theta, \beta \mid w)$ with a variational distribution $q(z, \theta, \beta)$ that factorizes as follows:

$$
q(z, \theta, \beta)
=
\prod_{i=1}^{D} q(\theta_i \mid \gamma_i)
\prod_{k=1}^{K} q(\beta_k \mid \lambda_k)
\prod_{i=1}^{D}
\prod_{j=1}^{N_i}
q(z_{ij} \mid \phi_{ij})
$$

Here:

$\gamma_i$ are the variational parameters for the Dirichlet distribution over topics for document $i$.

$\lambda_k$ are the variational parameters for the Dirichlet distribution over words for topic $k$.

$\phi_{ijk}$ are the variational parameters for the categorical distribution over topics for word $w_{ij}$.

Evidence Lower Bound (ELBO)

The ELBO is the objective function that VI seeks to maximize. It is given by:

$$
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
$$

This can be broken down into:

Likelihood Term: $\mathbb{E}_q[\log p(w \mid z, \beta)]$ captures how well the model explains the observed words given the topic assignments and topic-word distributions.

Topic Assignment Prior: $\mathbb{E}_q[\log p(z \mid \theta)]$ represents the prior belief about topic assignments given the document-topic distributions.

Document-Topic Prior: $\mathbb{E}_q[\log p(\theta \mid \alpha)]$ represents the prior belief about document-topic distributions.

Topic-Word Prior: $\mathbb{E}_q[\log p(\beta \mid \eta)]$ represents the prior belief about topic-word distributions.

Entropy Term: $-\mathbb{E}_q[\log q(z, \theta, \beta)]$ encourages the variational distribution $q$ to remain close to the true posterior.

Update Formulas in VI

Update for $\phi_{ijk}$ (Soft Topic Assignment)

$$
\phi_{ijk}
\propto
\exp\left\{
\mathbb{E}_q[\log \theta_{ik}]
+
\mathbb{E}_q[\log \beta_{k w_{ij}}]
\right\}
$$

Normalization:

$$
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
$$

This ensures that $\phi_{ijk}$ sums to 1 across all topics for each word.

Update for $\gamma_{ik}$ (Document-Topic Distribution)

$$
\gamma_{ik}
=
\alpha_k
+
\sum_{j=1}^{N_i}
\phi_{ijk}
$$

Here, $\gamma_{ik}$ is updated by adding the Dirichlet prior $\alpha_k$ and the expected number of times topic $k$ is assigned to words in document $i$.

Update for $\lambda_{kw}$ (Topic-Word Distribution)

$$
\lambda_{kw}
=
\eta
+
\sum_{i=1}^{D}
\sum_{j=1}^{N_i}
\phi_{ijk}
\delta(w_{ij}=w)
$$

Here, $\lambda_{kw}$ is updated by adding the Dirichlet prior $\eta$ and the expected number of times word $w$ is associated with topic $k$ across all documents.

No need to normalize for $\theta_{ik}$ and $\beta_{wk}$ because the Dirichlet distribution will normalize them when used as parameters.

Final Inference

After variational parameters $\phi_{ijk}$, $\gamma_{ik}$, and $\lambda_{kw}$ have been updated iteratively to convergence, they represent the approximate posterior distributions over the latent variables. The $\gamma_{ik}$ values give the inferred document-topic distribution for each document, while $\lambda_{kw}$ informs the inferred word distributions for each topic.

For predicting the topic distribution for a new document:

Initialize $\theta_{\text{new}}$ randomly or uniformly.

Iteratively update $\theta_{\text{new}}$ using the same variational approach, but with the previously learned $\beta_{kw}$ values.

### Conclusion

LDA, combined with VI, provides a powerful framework for uncovering latent topic structures within a corpus. The ELBO serves as the objective function that guides the optimization process, ensuring that the variational parameters converge to values that approximate the true posterior distributions as closely as possible. Through the iterative update of variational parameters, LDA can effectively model the complex, overlapping themes present in large text datasets, making it an essential tool in natural language processing.

Certainly! Let’s walk through Latent Dirichlet Allocation (LDA), explaining it in a similar way to Gaussian Mixture Models (GMM) and focusing on the soft assignment of words to topics, along with the update equations used in Variational Inference (VI).

Overview of LDA and Soft Assignments

Just like a GMM, where data points (e.g., samples) are assigned probabilistically to clusters, LDA assigns words within documents to topics softly. In LDA, each document is a mixture of topics, and each word is probabilistically assigned to a topic. The assignment of words to topics is not hard (one-hot), but soft, meaning that each word has some probability of belonging to each topic.

LDA Generative Process (Like GMM Generative Process)

For each document in LDA:

A distribution over topics $\theta_i$ is drawn from a Dirichlet distribution with hyperparameters $\alpha$:

$$
\theta_i \sim \operatorname{Dirichlet}(\alpha)
$$

$\theta_i$ is a $K$-dimensional vector that represents the proportion of each topic in document $i$.

For each word $j$ in document $i$:

A topic $z_{ij}$ is drawn from a categorical distribution parameterized by the topic proportions $\theta_i$:

$$
z_{ij} \sim \operatorname{Categorical}(\theta_i)
$$

$z_{ij}$ is the topic assigned to word $j$ in document $i$.

The word $w_{ij}$ is then drawn from the topic's word distribution $\phi_k$, where $z_{ij}=k$ is the topic:

$$
w_{ij} \sim \operatorname{Categorical}(\phi_{z_{ij}})
$$

$\phi_k$ is the distribution of words for topic $k$, which comes from a Dirichlet distribution parameterized by $\beta$:

$$
\phi_k \sim \operatorname{Dirichlet}(\beta)
$$

Key Components in LDA (like GMM)

$\theta_i$: Distribution over topics for document $i$ (analogous to mixture weights $\pi$ in GMM).

$z_{ij}$: Latent variable representing the topic assignment for word $j$ in document $i$ (analogous to $z_i$, the component assignment in GMM).

$\phi_k$: Word distribution for topic $k$, where each word has some probability of being generated from the topic (analogous to the Gaussian component parameters in GMM).

Inference in LDA: Variational Inference and Soft Assignments

In Variational Inference (VI), just like in GMM, we don't sample the latent variables directly. Instead, we compute the posterior distribution of the latent variables by optimizing a variational approximation to the true posterior.

Mean-Field Factorization

We approximate the posterior distribution of the latent variables using a mean-field approximation, which factorizes the joint posterior distribution into independent variational distributions:

$$
q(\theta_i, z_{ij}, \phi_k)
=
q(\theta_i)
q(z_{ij})
q(\phi_k)
$$

Where:

$q(\theta_i)$ is the variational distribution over the topic proportions for document $i$.

$q(z_{ij})$ is the variational distribution over the topic assignment for word $j$ in document $i$.

$q(\phi_k)$ is the variational distribution over the word distributions for each topic.

Soft Assignments in LDA (Like Soft Assignments in GMM)

1. Variational Distribution for $z_{ij}$ (Soft Assignment)

In LDA, each word $w_{ij}$ has a soft assignment to each of the $K$ topics. Instead of sampling $z_{ij}$, we compute the variational responsibilities $\phi_{ijk}$, which represent the probability that word $j$ in document $i$ belongs to topic $k$. These are analogous to the soft assignments $\gamma_{z_{ik}}$ in GMM.

The variational distribution $q(z_{ij})$ is a categorical distribution with parameters $\phi_{ijk}$, which represent the soft assignments:

$$
q(z_{ij}=k)=\phi_{ijk}
$$

Where $\phi_{ijk}$ represents the probability that word $j$ in document $i$ is assigned to topic $k$.

2. Variational Distribution for $\theta_i$ (Document-Topic Proportions)

For each document, the topic proportions $\theta_i$ are drawn from a Dirichlet distribution. In VI, we approximate the posterior distribution over $\theta_i$ with a variational distribution $q(\theta_i)$, which is also a Dirichlet distribution:

$$
q(\theta_i)
=
\operatorname{Dirichlet}(\gamma_i)
$$

Where $\gamma_i$ is a $K$-dimensional vector representing the variational parameters for the topic proportions in document $i$.

3. Variational Distribution for $\phi_k$ (Topic-Word Distribution)

For each topic $k$, the word distribution $\phi_k$ is drawn from a Dirichlet distribution:

$$
q(\phi_k)
=
\operatorname{Dirichlet}(\lambda_k)
$$

Where $\lambda_k$ is a vector of length $V$ (vocabulary size), representing the variational parameters for the word distribution for topic $k$.

Update Equations in Variational Inference for LDA

Update for $q(z_{ij})$ (Soft Assignment)

The soft assignment of word $w_{ij}$ to each topic $k$ is updated based on the following equation:

$$
\phi_{ijk}
\propto
\exp\left(
\mathbb{E}_{q(\theta_i)}[\log \theta_{ik}]
+
\mathbb{E}_{q(\phi_k)}[\log \phi_{k w_{ij}}]
\right)
$$

Where:

$\mathbb{E}_{q(\theta_i)}[\log \theta_{ik}]$ is the expected log probability of topic $k$ in document $i$.

$\mathbb{E}_{q(\phi_k)}[\log \phi_{k w_{ij}}]$ is the expected log probability of word $w_{ij}$ in topic $k$.

This is analogous to how responsibilities $r_{ik}$ are updated in GMMs.

Update for $q(\theta_i)$ (Document-Topic Proportions)

The variational parameters $\gamma_i$ for the topic proportions $\theta_i$ in document $i$ are updated as follows:

$$
\gamma_{ik}
=
\alpha_k
+
\sum_{j=1}^{N_i}
\phi_{ijk}
$$

Where:

$\alpha_k$ is the prior parameter for topic $k$.

$\phi_{ijk}$ is the soft assignment of word $w_{ij}$ to topic $k$.

$N_i$ is the number of words in document $i$.

This is analogous to the way the mixture weights $\pi_k$ are updated in GMMs.

Update for $q(\phi_k)$ (Topic-Word Distribution)

The variational parameters $\lambda_k$ for the word distribution $\phi_k$ over topic $k$ are updated as follows:

$$
\lambda_{kv}
=
\beta_v
+
\sum_{i=1}^{D}
\sum_{j=1}^{N_i}
\mathbb{1}(w_{ij}=v)
\phi_{ijk}
$$

Where:

$\beta_v$ is the prior parameter for word $v$.

$\mathbb{1}(w_{ij}=v)$ is an indicator function that is 1 if $w_{ij}$ is word $v$.

$\phi_{ijk}$ is the soft assignment of word $w_{ij}$ to topic $k$.

$D$ is the number of documents, and $N_i$ is the number of words in document $i$.

Summary: LDA and Soft Assignments Like GMM

In LDA, the latent variable $z_{ij}$ represents the topic assignment for word $w_{ij}$ in document $i$, analogous to the cluster assignment $z_i$ in GMM.

Instead of directly sampling $z_{ij}$, soft assignments $\phi_{ijk}$ (variational responsibilities) are used to represent the probability that word $w_{ij}$ belongs to topic $k$, similar to the soft assignments in GMM.

The update equations in VI for LDA involve updating these soft assignments as well as the variational distributions over the document-topic proportions.
