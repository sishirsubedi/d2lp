
.. _sec_lda_etm:

LDA ETM
=======


To update the Latent Dirichlet Allocation (LDA) model to incorporate a
neural network structure, where the prior over topics :math:`p(z)`
follows a standard normal distribution :math:`N(0,1)`, and
:math:`p(w|z)` is parameterized by a decoder network, we essentially
move towards a Variational Autoencoder (VAE) framework for topic
modeling. This approach is known as a **Neural Variational Document
Model (NVDM)** or **Variational Autoencoder for Topic Modeling**.

Here’s a high-level outline of the updated model:

**Latent Variables and Priors**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  **Prior on Topics :math:`z`**:

   .. math::


      p(z) = \mathcal{N}(0, I)

   Here, :math:`z` is a continuous latent variable with a multivariate
   Gaussian prior with mean 0 and identity covariance matrix.

**Encoder Network (Inference Network)**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The encoder network maps a document :math:`d` to a latent representation
:math:`z`. The encoder network outputs the mean :math:`\mu` and the
diagonal covariance :math:`\sigma^2` of the Gaussian distribution over
:math:`z`:

.. math::


   q(z|d) = \mathcal{N}(\mu_d, \text{diag}(\sigma^2_d))

The parameters :math:`\mu_d` and :math:`\sigma_d` are learned from the
data using a neural network that takes the document :math:`d` (typically
represented as a bag-of-words vector) as input.

**Decoder Network**
~~~~~~~~~~~~~~~~~~~

The decoder network parameterizes the distribution :math:`p(w|z)`. Given
a latent topic representation :math:`z`, the decoder outputs a
probability distribution over words:

.. math::


   p(w|z) = \text{Softmax}(f(z))

Here, :math:`f(z)` is a neural network (decoder) that maps the latent
variable :math:`z` to the logits of a softmax distribution over the
vocabulary.

**Objective Function**
~~~~~~~~~~~~~~~~~~~~~~

The objective function is the Evidence Lower Bound (ELBO) which combines
the reconstruction loss and the KL divergence between the approximate
posterior :math:`q(z|d)` and the prior :math:`p(z)`:

.. math::


   \mathcal{L} = \mathbb{E}_{q(z|d)}[\log p(d|z)] - \text{KL}(q(z|d) || p(z))

**Reconstruction Term**:
^^^^^^^^^^^^^^^^^^^^^^^^

.. math::


   \mathbb{E}_{q(z|d)}[\log p(d|z)] = \mathbb{E}_{q(z|d)}\left[\sum_{w \in d} \log p(w|z)\right]

**KL Divergence Term**:
^^^^^^^^^^^^^^^^^^^^^^^

.. math::


   \text{KL}(q(z|d) || p(z)) = \frac{1}{2} \sum_{j=1}^K \left(1 + \log(\sigma^2_{d,j}) - \mu^2_{d,j} - \sigma^2_{d,j}\right)
