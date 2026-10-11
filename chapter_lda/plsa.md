# Probabilistic Latent Semantic Analysis (PLSA)
:label:`sec_plsa`

**Probabilistic Latent Semantic Analysis (PLSA)**, also known as Probabilistic Latent Semantic Indexing (PLSI), is a statistical technique used in natural language processing and information retrieval to uncover latent topics within a collection of text documents. PLSA extends the ideas of Latent Semantic Analysis (LSA) by incorporating a probabilistic model, which offers a more rigorous foundation and can handle uncertainty in the data.


### Latent Variables (Topics)
PLSA assumes that each document can be represented as a mixture of latent topics, and each topic is characterized by a probability distribution over words. These latent topics are not directly observable but can be inferred from the data.

### Generative Model
PLSA is based on a generative model where each word in a document is generated through the following process:
* **Select a Document**: A document $d$ is chosen with probability $P(d)$.
* **Select a Topic**: Given the document, a latent topic $z$ is chosen with probability $P(z\vert{}d)$.
* **Generate a Word**: Given the topic, a word $w$ is generated with probability $P(w\vert{}z)$.

### Probabilities and Parameters
* $P(w\vert{}z)$: Probability of a word given a topic, representing the distribution of words in a topic.
* $P(z\vert{}d)$: Probability of a topic given a document, representing the distribution of topics in a document.
* $P(d)$: Probability of selecting a particular document.

### Expectation-Maximization (EM) Algorithm
The parameters of the model are estimated using the Expectation-Maximization (EM) algorithm. The EM algorithm iteratively adjusts the parameters to maximize the likelihood of the observed data (documents and words). It consists of two steps:
* **E-step (Expectation step):** Calculate the expected value of the latent variables (topics) given the current estimates of the parameters.
* **M-step (Maximization step):** Update the parameters to maximize the likelihood of the observed data given the expected values computed in the E-step.

### Probabilistic Framework
The probabilistic nature of PLSA allows it to model uncertainty and variability in the data, making it more robust than traditional LSA. It can also provide probabilities for different interpretations, which is useful in various applications like information retrieval and topic modeling.

### Applications

* **Information Retrieval**: PLSA can improve the relevance of search results by considering the latent topics in documents and queries.
* **Document Clustering**: Documents can be clustered based on their topic distributions.
* **Topic Modeling**: Identifying underlying topics in a corpus, useful for summarization and analysis.

### Limitations

* **Overfitting**: PLSA can suffer from overfitting because it does not include a prior distribution over topics. This can lead to a model that fits the training data well but generalizes poorly to new data.
* **Scalability**: The computational cost of training PLSA increases with the size of the corpus and the number of topics.

PLSA is a precursor to more advanced topic models like Latent Dirichlet Allocation (LDA), which address some of PLSA's limitations by incorporating Bayesian priors, thus providing better generalization and avoiding overfitting.


### Derivation of the Expectation-Maximization (EM) Algorithm

To derive the Expectation-Maximization (EM) algorithm for Probabilistic Latent Semantic Analysis (PLSA), we follow a process that involves defining the model, formulating the complete data likelihood, and then deriving the E-step and M-step.

### Model Setup

In PLSA, the goal is to model the probability distribution of words and documents using latent topics. Let:
* $D$ be the set of documents,
* $W$ be the set of words (vocabulary),
* $K$ be the number of latent topics.

The probability of observing a word $w$ in document $d$ is given by:

$$P(w\vert{}d) = \sum_{z=1}^{K} P(w\vert{}z) P(z\vert{}d)$$

where:
* $P(w\vert{}z)$ is the probability of word $w$ given topic $z$,
* $P(z\vert{}d)$ is the probability of topic $z$ given document $d$.

### Complete Data Likelihood

The complete data includes not only the observed words and documents but also the latent topic assignments. Let $z$ be a latent variable representing the topic. The joint probability of a word, document, and topic is:

$$P(w, z, d) = P(d) P(z\vert{}d) P(w\vert{}z)$$

The complete data likelihood, given a set of document-word pairs $(d, w)$ and the corresponding topic assignments $z$, is:

$$L = \prod_{d \in D} \prod_{w \in W} P(w, z, d)^{n(d, w)}$$

where $n(d, w)$ is the number of times word $w$ appears in document $d$.

### EM Algorithm Steps

The EM algorithm iteratively optimizes the parameters by alternating between the E-step and the M-step.

#### E-Step (Expectation)

In the E-step, we compute the posterior distribution of the latent variables given the current parameters. The posterior probability of a topic $z$ given a word $w$ and a document $d$ is:

$$P(z\vert{}d, w) = \frac{P(w\vert{}z) P(z\vert{}d)}{\sum_{z'=1}^{K} P(w\vert{}z') P(z'\vert{}d)}$$

#### M-Step (Maximization)

In the M-step, we update the parameters $P(w\vert{}z)$ and $P(z\vert{}d)$ to maximize the expected complete data log-likelihood:

1. **Updating $P(w\vert{}z)$**:

   $$P(w\vert{}z) = \frac{\sum_{d \in D} n(d, w) P(z\vert{}d, w)}{\sum_{w' \in W} \sum_{d \in D} n(d, w') P(z\vert{}d, w')}$$

2. **Updating $P(z\vert{}d)$**:

   $$P(z\vert{}d) = \frac{\sum_{w \in W} n(d, w) P(z\vert{}d, w)}{\sum_{z'=1}^{K} \sum_{w \in W} n(d, w) P(z'\vert{}d, w)}$$

### Iterative Optimization

The E-step and M-step are iteratively applied until convergence. Convergence can be determined by checking if the change in the log-likelihood is below a specified threshold.


### Variant: PLSA with Topic Prior $P(z)$

The basic formulation of PLSA presented above does not include a prior distribution over the latent topics $P(z)$. This is one of the key differences between standard PLSA and its Bayesian extension, Latent Dirichlet Allocation (LDA).

In standard PLSA, the topic distribution $P(z\vert{}d)$ is directly parameterized for each document, which can lead to overfitting when the number of topics is large or the dataset is small.

Below is the EM formulation for PLSA with the inclusion of a prior distribution $P(z)$ over the topics.

### E-Step (Expectation Step)
The E-step calculates the posterior probabilities of the latent topics $z$ given the observed data (word $w$ in document $d$) using the current parameters:

$$P(z\vert{}d, w) = \frac{P(z) P(d\vert{}z) P(w\vert{}z)}{\sum_{z'} P(z') P(d\vert{}z') P(w\vert{}z')}$$

In this formula:
* $P(z)$ represents the prior probability of topic $z$,
* $P(d\vert{}z)$ is the probability of the document given topic $z$,
* $P(w\vert{}z)$ is the probability of the word given topic $z$.

### M-Step (Maximization Step)
The M-step updates the model parameters to maximize the expected complete data log-likelihood given the current posterior probabilities:

1. **Updating $P(w\vert{}z)$**:
   $$P(w\vert{}z) = \frac{\sum_d n(d, w) P(z\vert{}d, w)}{\sum_{d, w'} n(d, w') P(z\vert{}d, w')}$$

2. **Updating $P(d\vert{}z)$**:
   $$P(d\vert{}z) = \frac{\sum_w n(d, w) P(z\vert{}d, w)}{\sum_{d', w} n(d', w) P(z\vert{}d', w)}$$

3. **Updating $P(z)$**:
   $$P(z) = \frac{1}{R} \sum_{d, w} n(d, w) P(z\vert{}d, w)$$

   where $R = \sum_{d, w} n(d, w)$ is the total number of word occurrences in the corpus.

### Benefits of Including Prior $P(z)$
The prior $P(z)$ allows for a more structured model by influencing topic distributions, aiding in regularization and preventing overfitting. This approach serves as an intermediate step between standard PLSA and Bayesian models like LDA.

