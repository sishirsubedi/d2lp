# Principal Component Analysis
:label:`chap_pca`


We can understand Principal Component Analysis (PCA) using Singular Value Decomposition (SVD) and relationships between the matrices and the properties of matrix multiplication. 


Given the SVD of matrix $X$:

$$X = U \Sigma V^T$$

where:

- $X$ is an $m \times n$ matrix.
- $U$ is an $m \times m$ orthogonal matrix (columns are left singular vectors).
- $\Sigma$ is an $m \times n$ diagonal matrix with non-negative real numbers on the diagonal (singular values).
- $V$ is an $n \times n$ orthogonal matrix (columns are right singular vectors).
- $V^T$ is the transpose of $V$.

### Relationship among X,U,and V

We can start with simple cross-product matrices. Here, we can have two views - one sample x sample or feature x feature:

$$XX^T = U \Sigma V^T (V \Sigma^T U^T) = U \Sigma \Sigma^T U^T$$

$$X^T X = V \Sigma^T U^T (U \Sigma V^T) = V \Sigma^T \Sigma V^T$$


From the above, we can write:

$$XX^T U = U \Sigma \Sigma^T$$

$$X^T X V = V \Sigma^T \Sigma$$

Given $XX^T U = U \Sigma \Sigma^T$, we can rearrange this equation:

$$XX^T U = U \Sigma \Sigma^T$$

Multiplying both sides by $U^T$:

$$U^T XX^T U = \Sigma \Sigma^T$$

Thus:

$$U^T X = \Sigma V^T$$

Taking the transpose:

$$X^T U = V \Sigma^T$$

Now, multiplying both sides by $\left(\Sigma^T\right)^{-1}$ from the right:

$$X^T U \left(\Sigma^T\right)^{-1} = V$$

Therefore:

$$V = X^T U \left(\Sigma^T\right)^{-1}$$

Similarly, given $X^T X V = V \Sigma^T \Sigma$, we can rearrange this equation:

$$X^T X V = V \Sigma^T \Sigma$$

Multiplying both sides by $V^T$:

$$V^T X^T X V = \Sigma^T \Sigma$$

Thus:

$$X V = U \Sigma$$

Now, multiplying both sides by $\Sigma^{-1}$ from the right:

$$X V \Sigma^{-1} = U$$

Therefore:

$$U = X V \Sigma^{-1}$$

In summary, we have derived the following relationships:

$$V = X^T U \left(\Sigma^T\right)^{-1}$$

$$U = X V \Sigma^{-1}$$

These equations show how the right singular vectors ($V$) can be expressed in terms of the left singular vectors ($U$) and vice versa, using the original matrix $X$ and the diagonal matrix of singular values ($\Sigma$).


**DUALITY OF PCA** Both U and V can be learned from other components.

$$X = U \Sigma V^T$$

$$V = X^T U \left(\Sigma^T\right)^{-1}$$

$$U = X V \Sigma^{-1}$$

### Interpretation

- **Projection to Latent Space**:
  When we project the original data $X$ onto the latent space defined by $V$, we are essentially expressing $X$ in terms of its principal directions (eigenvectors of $X^T X$). This is the projection $X V$.

- **Normalization**:
  The matrix $\Sigma^{-1}$ normalizes this projection. The resulting product $X V \Sigma^{-1}$ gives us the matrix $U$, which contains the left singular vectors of $X$, or equivalently, the eigenvectors of $XX^T$.

The equation $U = X V \Sigma^{-1}$ encapsulates the idea that the left singular vectors (eigenvectors of $XX^T$) can be expressed as a normalized projection of the original data $X$ onto the right singular vectors (eigenvectors of $X^T X$). This provides a bridge between the data space and the latent space, showing how the singular vectors and the original data matrix are interconnected through the SVD.

### Singular Values vs Eigenvalues

- **Singular Values ($\sigma_i$)**: Obtained from the **SVD** of data matrix $X$: $X = U \Sigma V^T$. Represents the **scale/stretching** of $X$ along principal directions.  
  *Intuition:* Lengths of axes when a unit sphere is transformed by $X$.

- **Eigenvalues ($\lambda_i$)**: Obtained from the **eigendecomposition** of cross-product matrix $X^T X$: $X^T X = V \Lambda V^T$. Represents the **squared scale** of data along principal components.  
  *Intuition:* Squared lengths of the transformed axes (spread of data).

### Relationship

$$\lambda_i = \sigma_i^2$$

Eigenvalues of $X^T X$ are the **squares** of singular values.

For a centered data matrix $X$, if the covariance matrix is

$$C = \frac{1}{m-1}X^T X,$$

then its eigenvalues are

$$\lambda_i = \frac{\sigma_i^2}{m-1}.$$

Thus, the eigenvalues of $X^T X$ are proportional to the variance of the data along the principal components.


| Quantity | From matrix | Meaning | Units |
|----------|-------------|---------|-------|
| Singular value | $X$ (SVD) | Scale/stretching | Linear scale |
| Eigenvalue | $X^T X$ | Squared scale | Squared linear scale |
| Covariance eigenvalue | $\frac{1}{m-1}X^T X$ | Variance (spread) | Squared feature units |


### Eigenvector is a direction in feature space

In the case where the **data matrix** is $X$ and the **cross-product matrix** is $A = X^T X$, the **eigenvector $v$** in the equation $A v = \lambda v$ is a **direction in feature space**, not a column of $A$.

**Interpretation as a Feature Combination: Each element in an eigenvector corresponds to the contribution of each original feature in that particular direction. Therefore, you can think of an eigenvector as a "recipe" for combining features in a specific proportion, representing a dominant pattern in the data.**

### Let's break this down:

1. **Cross-Product Matrix $A = X^T X$**:
   - Here, $X$ is your data matrix with dimensions $m \times n$ (where $m$ is the number of samples and $n$ is the number of features).
   - The matrix $A = X^T X$ is an $n \times n$ matrix, representing the inner products between features (it's the **Gram matrix** of $X$).

2. **Eigenvectors of $A$**:
   - The matrix $A$ is symmetric and positive semi-definite, so it has real, non-negative eigenvalues and corresponding eigenvectors.
   - The eigenvectors $v$ of $A$ satisfy the equation $A v = \lambda v$, where $\lambda$ is the eigenvalue corresponding to $v$.

3. **Direction in Feature Space**:
   - $v$ is an eigenvector of the matrix $A = X^T X$. Since $A$ is the product of $X^T$ and $X$, the eigenvector $v$ lies in the **feature space**, meaning it is a direction in the space spanned by the features.
   - The eigenvectors of $A$ represent directions in the feature space, where the elements of $v$ give the coefficients for combining the original features. Thus, $v$ indicates a direction that corresponds to some combination of the original features in $X$.

### Key Point:
- **Eigenvectors of $A = X^T X$** are **directions in the original feature space**. Each element of the eigenvector gives the coefficient of the corresponding feature.

#### Why Not Columns of $A$?
- $A$ is the **cross-product matrix** of $X$, and its columns represent relationships between features, not the original data features themselves. The eigenvectors $v$ are calculated from the relationships between the features, so they describe directions in the feature space of $X$.

- The eigenvector $v$ of $A = X^T X$ represents a direction in the original feature space, with each component corresponding to one of the original features.


### ROW space vs COLUMN space

The main idea is eigen vectors in feature space i.e. Vs are unit length and $X$ transforms these vectors to another coordinate system where eigen vectors are sample space i.e. Us and here vectors are scaled by sigma.

The right singular vectors $V$ correspond to directions in the **row space of $X$**, which is a subspace of the feature space. The left singular vectors $U$ correspond to directions in the **column space of $X$**, which is a subspace of the sample space.


### Understanding SVD: Column Space vs. Row Space, and the Role of Singular Values

In the Singular Value Decomposition (SVD), a data matrix $X \in \mathbb{R}^{n \times p}$ is factorized as:

$$X = U \Sigma V^T$$

where:
- $U \in \mathbb{R}^{n \times n}$ contains the **left singular vectors**, which form an orthonormal basis for the **column space** (sample space) of $X$,
- $V \in \mathbb{R}^{p \times p}$ contains the **right singular vectors**, which form an orthonormal basis for the **row space** (feature space) of $X$,
- $\Sigma$ is a diagonal matrix with **singular values** $\sigma_i$, indicating the strength/scale along each singular direction.

The **row space** of $X$ is related to the **eigenvectors of $X^T X$** and spans directions in the **feature space**. The **column space** of $X$ corresponds to the **eigenvectors of $X X^T$** and spans directions in the **sample space**. Although these spaces live in different dimensions, they are intimately connected through the SVD — they share the same rank, and their bases are linked via singular values.

A key insight is that each right singular vector $v_i$ (from the feature space) is a **unit vector**, and when it is transformed by $X$, the result is a vector $\sigma_i u_i$ in the sample space. That is:

$$X v_i = \sigma_i u_i$$

So, while both $v_i$ and $u_i$ are unit vectors in their respective spaces, the transformation from feature to sample space scales the magnitude by the singular value $\sigma_i$. This shows that singular values encode how much scale/energy is present in each direction. For centered data, $\sigma_i^2$ is proportional to the variance of the data along that direction. The mapping preserves directionality but stretches the magnitude, effectively turning unit spheres in feature space into ellipsoids in sample space.

In this way, SVD provides a geometrically meaningful link between the structure of data in feature space and how it manifests in sample space.


### Main Idea

In SVD, the "core" directions live in feature space. Singular values determine how important those directions are. Applying the data matrix $X$ to each feature-space direction $v$ gives you the corresponding sample-space direction $u$, scaled by how much scale/energy is present along $v$.


```toc
:maxdepth: 2

laplacian
twodim
autoencoder
factor
```
