# SMCS Mathematical Formulas

This document provides detailed mathematical formulations for all algorithms implemented in the smcs library.

## Table of Contents

1. [Core Concepts](#1-core-concepts)
2. [Resampling Algorithms](#2-resampling-algorithms)
3. [Sequential Monte Carlo (SMC)](#3-sequential-monte-carlo-smc)
4. [Parameter Learning Algorithms](#4-parameter-learning-algorithms)
5. [Advanced SMC Methods](#5-advanced-smc-methods)
6. [MCMC Methods](#6-mcmc-methods)
7. [Bootstrap Methods](#7-bootstrap-methods)
8. [Importance Sampling](#8-importance-sampling)
9. [Quasi-Monte Carlo](#9-quasi-monte-carlo)
10. [State-Space Models](#10-state-space-models)

---

## 1. Core Concepts

### 1.1 State-Space Model

A general state-space model is defined by:

**State Equation (Hidden Markov Process):**

$$x_t \sim p(x_t \mid x_{t-1}, \theta)$$

**Observation Equation:**

$$y_t \sim p(y_t \mid x_t, \theta)$$

**Initial Distribution:**

$$x_0 \sim p(x_0 \mid \theta)$$

where:
- $x_t \in \mathbb{R}^{d_x}$: latent state at time $t$
- $y_t \in \mathbb{R}^{d_y}$: observation at time $t$
- $\theta$: model parameters

### 1.2 Filtering Problem

**Goal:** Compute the filtering distribution:

$$p(x_t \mid y_{1:t}) = \frac{p(y_t \mid x_t) p(x_t \mid y_{1:t-1})}{p(y_t \mid y_{1:t-1})}$$

**Prediction Step:**

$$p(x_t \mid y_{1:t-1}) = \int p(x_t \mid x_{t-1}) p(x_{t-1} \mid y_{1:t-1}) dx_{t-1}$$

**Update Step:**

$$p(x_t \mid y_{1:t}) \propto p(y_t \mid x_t) p(x_t \mid y_{1:t-1})$$

### 1.3 Marginal Likelihood

$$p(y_{1:T}) = \prod_{t=1}^{T} p(y_t \mid y_{1:t-1}) = \prod_{t=1}^{T} \int p(y_t \mid x_t) p(x_t \mid y_{1:t-1}) dx_t$$

### 1.4 Effective Sample Size (ESS)

For normalized weights $\{w^{(i)}\}_{i=1}^{N}$ where $\sum_i w^{(i)} = 1$:

$$\text{ESS} = \frac{1}{\sum_{i=1}^{N} (w^{(i)})^2}$$

For log-weights:

$$\text{ESS} = \exp\left(-\text{logsumexp}(2 \cdot \tilde{w}) + 2 \cdot \text{logsumexp}(\tilde{w})\right)$$

where $\tilde{w}^{(i)} = \log w^{(i)} - \text{logsumexp}(\log w)$ are normalized log-weights.

### 1.5 Log-Sum-Exp Trick

For numerical stability:

$$\text{logsumexp}(x_1, \ldots, x_n) = m + \log\left(\sum_{i=1}^{n} e^{x_i - m}\right)$$

where $m = \max_i x_i$.

---

## 2. Resampling Algorithms

### 2.1 Multinomial Resampling

Draw $N$ indices independently:

$$A^{(i)} \sim \text{Categorical}(w^{(1)}, \ldots, w^{(N)}), \quad i = 1, \ldots, N$$

**Variance:** $\text{Var}[N^{(j)}] = N w^{(j)}(1 - w^{(j)})$

### 2.2 Systematic Resampling

1. Draw single uniform: $U \sim \text{Uniform}(0, 1)$
2. Compute positions: $u_i = \frac{i - 1 + U}{N}$ for $i = 1, \ldots, N$
3. Select ancestor: $A^{(i)} = F^{-1}(u_i)$

where $F(j) = \sum_{k=1}^{j} w^{(k)}$ is the cumulative distribution.

**Property:** Lower variance than multinomial, $O(N)$ complexity.

### 2.3 Stratified Resampling

1. Draw $N$ independent uniforms: $U_i \sim \text{Uniform}(0, 1)$
2. Compute stratified positions: $u_i = \frac{i - 1 + U_i}{N}$
3. Select ancestor: $A^{(i)} = F^{-1}(u_i)$

### 2.4 Residual Resampling

1. **Deterministic part:** Set $n^{(j)} = \lfloor N w^{(j)} \rfloor$
2. **Residual weights:** $\tilde{w}^{(j)} = N w^{(j)} - n^{(j)}$
3. **Stochastic part:** Sample remaining $N - \sum_j n^{(j)}$ particles using normalized residual weights

### 2.5 Killing Resampling

For each particle $i$:
1. Compute expected copies: $\lambda_i = N w^{(i)}$
2. Deterministic copies: $n_i = \lfloor \lambda_i \rfloor$
3. Random extra copy with probability $\lambda_i - n_i$

### 2.6 SSP (Srinivasan Sampling Process)

Ensures the number of copies differs from expectation by at most 1:

$$n^{(i)} \in \{\lfloor N w^{(i)} \rfloor, \lceil N w^{(i)} \rceil\}$$

with appropriate randomization to maintain unbiasedness.

### 2.7 Optimal Transport Resampling

Minimizes expected transport distance using quantile coupling:

$$A^{(i)} = F_w^{-1}\left(\frac{i - 0.5}{N}\right)$$

Couples uniform grid to empirical CDF for minimal expected squared distance.

---

## 3. Sequential Monte Carlo (SMC)

### 3.1 Bootstrap Particle Filter (Gordon et al., 1993)

**Algorithm:**

1. **Initialize:** Sample $x_0^{(i)} \sim p(x_0)$, set $w_0^{(i)} = 1/N$

2. **For** $t = 1, \ldots, T$:

   a. **Resample** if $\text{ESS} < \tau \cdot N$:
   $$A_{t-1}^{(i)} \sim \text{Categorical}(w_{t-1}^{(1)}, \ldots, w_{t-1}^{(N)})$$

   b. **Propagate** using transition as proposal:
   $$\tilde{x}_t^{(i)} \sim p(x_t \mid x_{t-1}^{(A_{t-1}^{(i)})})$$

   c. **Weight** using likelihood:
   $$w_t^{(i)} \propto p(y_t \mid \tilde{x}_t^{(i)})$$

**Log-likelihood estimate:**

$$\log \hat{p}(y_{1:T}) = \sum_{t=1}^{T} \log\left(\frac{1}{N} \sum_{i=1}^{N} w_t^{(i)}\right)$$

### 3.2 Auxiliary Particle Filter (Pitt & Shephard, 1999)

**Key Idea:** Use predictive likelihood for first-stage resampling.

**Algorithm:**

1. **First-stage weights** (predictive):
   $$\lambda_t^{(i)} \propto w_{t-1}^{(i)} \cdot p(y_t \mid \mu_t^{(i)})$$
   where $\mu_t^{(i)} = \mathbb{E}[x_t \mid x_{t-1}^{(i)}]$

2. **Resample:** $A_{t-1}^{(i)} \sim \text{Categorical}(\lambda_t^{(1)}, \ldots, \lambda_t^{(N)})$

3. **Propagate:** $\tilde{x}_t^{(i)} \sim p(x_t \mid x_{t-1}^{(A_{t-1}^{(i)})})$

4. **Second-stage weights** (adjustment):
   $$w_t^{(i)} \propto \frac{p(y_t \mid \tilde{x}_t^{(i)})}{p(y_t \mid \mu_t^{(A_{t-1}^{(i)})})}$$

### 3.3 Guided (Optimal) Particle Filter

**Proposal:** Use observation-dependent proposal $q(x_t \mid x_{t-1}, y_t)$

**Importance weights:**

$$w_t^{(i)} \propto \frac{p(y_t \mid x_t^{(i)}) \cdot p(x_t^{(i)} \mid x_{t-1}^{(i)})}{q(x_t^{(i)} \mid x_{t-1}^{(i)}, y_t)}$$

**Optimal proposal** (when tractable):

$$q^*(x_t \mid x_{t-1}, y_t) = p(x_t \mid x_{t-1}, y_t) = \frac{p(y_t \mid x_t) p(x_t \mid x_{t-1})}{p(y_t \mid x_{t-1})}$$

**Optimal weights:**

$$w_t^{(i)} = p(y_t \mid x_{t-1}^{(i)}) = \int p(y_t \mid x_t) p(x_t \mid x_{t-1}^{(i)}) dx_t$$

### 3.4 Regularized Particle Filter

**Kernel smoothing** after resampling to maintain diversity:

$$\tilde{x}_t^{(i)} = x_t^{(A_t^{(i)})} + h \cdot \epsilon^{(i)}, \quad \epsilon^{(i)} \sim \mathcal{N}(0, I)$$

**Bandwidth** (Silverman's rule):

$$h = \left(\frac{4}{d+2}\right)^{\frac{1}{d+4}} \cdot N^{-\frac{1}{d+4}} \cdot \hat{\sigma}$$

where $d$ is the state dimension and $\hat{\sigma}$ is the empirical standard deviation.

### 3.5 Unscented Particle Filter

**Combines Unscented Kalman Filter with particle filtering.**

**Sigma points** for state $x$ with mean $\mu$ and covariance $P$:

$$\mathcal{X}_0 = \mu$$
$$\mathcal{X}_i = \mu + \sqrt{(d + \lambda) P}_i, \quad i = 1, \ldots, d$$
$$\mathcal{X}_{i+d} = \mu - \sqrt{(d + \lambda) P}_i, \quad i = 1, \ldots, d$$

**Weights:**

$$W_0^{(m)} = \frac{\lambda}{d + \lambda}, \quad W_0^{(c)} = \frac{\lambda}{d + \lambda} + (1 - \alpha^2 + \beta)$$
$$W_i^{(m)} = W_i^{(c)} = \frac{1}{2(d + \lambda)}, \quad i = 1, \ldots, 2d$$

where $\lambda = \alpha^2(d + \kappa) - d$.

**Prediction:**

$$\hat{\mu}_{t|t-1} = \sum_{i=0}^{2d} W_i^{(m)} f(\mathcal{X}_i)$$
$$\hat{P}_{t|t-1} = \sum_{i=0}^{2d} W_i^{(c)} (f(\mathcal{X}_i) - \hat{\mu}_{t|t-1})(f(\mathcal{X}_i) - \hat{\mu}_{t|t-1})^\top + Q$$

### 3.6 Ensemble Kalman Filter (EnKF)

**Forecast step:**

$$x_t^{f,(i)} = M(x_{t-1}^{a,(i)}) + \eta^{(i)}, \quad \eta^{(i)} \sim \mathcal{N}(0, Q)$$

**Covariance inflation:**

$$x_t^{f,(i)} \leftarrow \bar{x}_t^f + \kappa (x_t^{f,(i)} - \bar{x}_t^f)$$

where $\kappa > 1$ is the inflation factor.

**Analysis step (Stochastic EnKF):**

$$x_t^{a,(i)} = x_t^{f,(i)} + K_t (y_t + \epsilon^{(i)} - H x_t^{f,(i)})$$

where $\epsilon^{(i)} \sim \mathcal{N}(0, R)$ and:

$$K_t = P_{xy}^f (P_{yy}^f + R)^{-1}$$

**Ensemble covariances:**

$$P_{xy}^f = \frac{1}{N-1} \sum_{i=1}^{N} (x_t^{f,(i)} - \bar{x}_t^f)(H x_t^{f,(i)} - \overline{Hx}_t^f)^\top$$

$$P_{yy}^f = \frac{1}{N-1} \sum_{i=1}^{N} (H x_t^{f,(i)} - \overline{Hx}_t^f)(H x_t^{f,(i)} - \overline{Hx}_t^f)^\top$$

---

## 4. Parameter Learning Algorithms

### 4.1 Liu-West Filter

**Kernel shrinkage** for parameter particles:

$$\theta_t^{(i)} = a \cdot \theta_{t-1}^{(i)} + (1-a) \cdot \bar{\theta}_{t-1} + \sqrt{1 - a^2} \cdot V^{1/2} \epsilon^{(i)}$$

where:
- $\bar{\theta}_{t-1} = \sum_i w_{t-1}^{(i)} \theta_{t-1}^{(i)}$ is the weighted mean
- $V = \sum_i w_{t-1}^{(i)} (\theta_{t-1}^{(i)} - \bar{\theta}_{t-1})(\theta_{t-1}^{(i)} - \bar{\theta}_{t-1})^\top$
- $a = \frac{3\delta - 1}{2\delta}$ with $\delta \in (0.95, 0.99)$ (shrinkage parameter)
- $\epsilon^{(i)} \sim \mathcal{N}(0, I)$

**Properties:**
- Maintains approximate posterior variance
- Prevents particle degeneracy in parameter space

### 4.2 Storvik Filter

**For models with sufficient statistics:**

1. **State propagation:** $x_t^{(i)} \sim p(x_t \mid x_{t-1}^{(i)}, \theta_{t-1}^{(i)})$

2. **Sufficient statistics update:**
   $$s_t^{(i)} = u(s_{t-1}^{(i)}, x_t^{(i)}, y_t)$$

3. **Parameter sampling from closed-form posterior:**
   $$\theta_t^{(i)} \sim p(\theta \mid s_t^{(i)})$$

**Example (Normal-Inverse-Gamma):**

For $y_t \mid \mu, \sigma^2 \sim \mathcal{N}(\mu, \sigma^2)$ with conjugate prior:

$$s_t = (n_t, \bar{y}_t, S_t) = \left(n_{t-1} + 1, \frac{n_{t-1}\bar{y}_{t-1} + y_t}{n_t}, S_{t-1} + \frac{n_{t-1}(y_t - \bar{y}_{t-1})^2}{n_t}\right)$$

### 4.3 SMC-Squared (SMC²)

**Nested SMC structure:**

- Outer SMC: $N_\theta$ parameter particles $\{\theta^{(j)}\}_{j=1}^{N_\theta}$
- Inner SMC: $N_x$ state particles per parameter $\{x^{(j,i)}\}_{i=1}^{N_x}$

**Algorithm:**

1. **Inner PF step** for each $\theta^{(j)}$:
   - Propagate state particles: $x_t^{(j,i)} \sim p(x_t \mid x_{t-1}^{(j,i)}, \theta^{(j)})$
   - Update state weights: $w_t^{(j,i)} \propto p(y_t \mid x_t^{(j,i)}, \theta^{(j)})$
   - Estimate likelihood: $\hat{p}(y_t \mid y_{1:t-1}, \theta^{(j)}) = \frac{1}{N_x} \sum_i w_t^{(j,i)}$

2. **Outer weight update:**
   $$W_t^{(j)} \propto W_{t-1}^{(j)} \cdot \hat{p}(y_t \mid y_{1:t-1}, \theta^{(j)})$$

3. **Outer resampling** when ESS low, followed by MCMC rejuvenation

### 4.4 Particle Marginal Metropolis-Hastings (PMMH)

**Target:** $p(\theta \mid y_{1:T})$

**Algorithm:**

1. **Propose:** $\theta' \sim q(\theta' \mid \theta)$

2. **Run particle filter** to estimate $\hat{p}(y_{1:T} \mid \theta')$

3. **Accept with probability:**
   $$\alpha = \min\left(1, \frac{\hat{p}(y_{1:T} \mid \theta') p(\theta')}{\hat{p}(y_{1:T} \mid \theta) p(\theta)} \cdot \frac{q(\theta \mid \theta')}{q(\theta' \mid \theta)}\right)$$

**Key property:** The particle filter likelihood estimate makes this a valid MCMC kernel targeting the exact posterior (pseudo-marginal approach).

---

## 5. Advanced SMC Methods

### 5.1 Waste-Free SMC (Dau & Chopin, 2022)

**Standard SMC discards intermediate MCMC samples. Waste-Free keeps all.**

**Algorithm:**

1. From $N$ particles, resample to get $M = N/k$ "mother" particles
2. Run $k$ MCMC steps from each mother
3. Keep all $M \times k = N$ particles

**Weights:**

$$w^{(i,j)} = \frac{1}{k} \cdot \frac{\pi_n(x^{(i,j)})}{\pi_{n-1}(x^{(i,j)})}$$

where $x^{(i,j)}$ is the $j$-th MCMC sample from mother $i$.

### 5.2 Adaptive SMC

**Automatically adapts temperature schedule based on ESS.**

**Temperature sequence:** $0 = \beta_0 < \beta_1 < \cdots < \beta_T = 1$

**Target at step $n$:**

$$\pi_n(\theta) \propto p(\theta) \cdot p(y_{1:T} \mid \theta)^{\beta_n}$$

**Adaptive selection of $\beta_{n+1}$:**

Find $\beta_{n+1}$ such that:

$$\text{ESS}\left(\{w^{(i)}(\beta_{n+1})\}\right) = \alpha \cdot N$$

where $w^{(i)}(\beta) \propto p(y_{1:T} \mid \theta^{(i)})^{\beta - \beta_n}$ and $\alpha \in (0.5, 0.9)$.

**Solved via bisection.**

### 5.3 SMC Samplers

**For static parameter inference via tempering:**

**Sequence of distributions:**

$$\pi_n(\theta) \propto \pi_0(\theta)^{1-\beta_n} \cdot \pi_T(\theta)^{\beta_n}$$

or

$$\pi_n(\theta) \propto p(\theta) \cdot \ell(\theta)^{\beta_n}$$

**Incremental weights:**

$$w_n^{(i)} = \frac{\pi_n(\theta^{(i)})}{\pi_{n-1}(\theta^{(i)})} = \ell(\theta^{(i)})^{\beta_n - \beta_{n-1}}$$

**Normalizing constant estimate:**

$$\log \hat{Z} = \sum_{n=1}^{T} \log\left(\frac{1}{N} \sum_{i=1}^{N} w_n^{(i)}\right)$$

### 5.4 Annealed Importance Sampling (AIS)

**Bridges from prior to posterior without resampling:**

**Sequence:** $\pi_0 = p(\theta)$ (prior) to $\pi_T = p(\theta \mid y)$ (posterior)

**Intermediate:**

$$\pi_n(\theta) \propto p(\theta) \cdot p(y \mid \theta)^{\beta_n}$$

**Weight accumulation:**

$$w_T = \prod_{n=1}^{T} \frac{\pi_n(\theta_n)}{\pi_{n-1}(\theta_n)} = \prod_{n=1}^{T} p(y \mid \theta_n)^{\beta_n - \beta_{n-1}}$$

**Normalizing constant:**

$$\hat{Z} = \frac{1}{N} \sum_{i=1}^{N} w_T^{(i)}$$

### 5.5 ABC-SMC (Approximate Bayesian Computation)

**For likelihood-free inference with simulator-based models.**

**Accept if:**

$$\rho(S(y^{\text{sim}}), S(y^{\text{obs}})) \leq \epsilon$$

where $S(\cdot)$ is a summary statistic and $\rho$ is a distance metric.

**Adaptive threshold:**

$$\epsilon_t = \text{quantile}_p(\{\rho^{(i)}\}_{i=1}^{N})$$

typically $p \in (0.5, 0.75)$.

**SMC kernel:**

1. Resample based on weights
2. Perturb: $\theta' = \theta + \sigma \cdot \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$
3. Simulate: $y^{\text{sim}} \sim p(y \mid \theta')$
4. Accept if $\rho(S(y^{\text{sim}}), S(y^{\text{obs}})) \leq \epsilon_t$

### 5.6 Particle Gibbs / PGAS

**Conditional SMC for batch inference.**

**Particle Gibbs:**

1. Fix reference trajectory $x_{1:T}^*$
2. Run conditional SMC: One particle follows $x_{1:T}^*$
3. Sample new trajectory from smoothing distribution
4. Update parameters given trajectory

**Ancestor Sampling (PGAS):**

At time $t$, sample ancestor for reference:

$$A_{t-1}^* \sim \text{Categorical}\left(w_{t-1}^{(i)} \cdot p(x_t^* \mid x_{t-1}^{(i)})\right)$$

This maintains proper targeting with lower variance.

### 5.7 Smoothing: Forward Filtering Backward Sampling

**Goal:** Sample from $p(x_{1:T} \mid y_{1:T})$

**Forward pass:** Run particle filter, store all particles and weights

**Backward pass:** For $t = T-1, \ldots, 1$:

$$p(x_t \mid x_{t+1}, y_{1:T}) \propto p(x_{t+1} \mid x_t) \cdot p(x_t \mid y_{1:t})$$

**Backward weights:**

$$\tilde{w}_t^{(i)} \propto w_t^{(i)} \cdot p(x_{t+1}^{\text{selected}} \mid x_t^{(i)})$$

Sample ancestor: $A_t \sim \text{Categorical}(\tilde{w}_t^{(1)}, \ldots, \tilde{w}_t^{(N)})$

---

## 6. MCMC Methods

### 6.1 Metropolis-Hastings

**Target:** $\pi(x)$

**Proposal:** $q(x' \mid x)$

**Algorithm:**

1. Propose: $x' \sim q(x' \mid x)$
2. Acceptance probability:
   $$\alpha = \min\left(1, \frac{\pi(x') q(x \mid x')}{\pi(x) q(x' \mid x)}\right)$$
3. Accept with probability $\alpha$, else stay at $x$

**Random Walk MH:**

$$q(x' \mid x) = \mathcal{N}(x, \sigma^2 I)$$

Symmetric proposal, so $\alpha = \min(1, \pi(x')/\pi(x))$.

### 6.2 Slice Sampling (Neal, 2003)

**Auxiliary variable method for automatic step size.**

**Algorithm:**

1. **Sample slice level:** $u \sim \text{Uniform}(0, \pi(x))$
2. **Find slice:** $S = \{x' : \pi(x') > u\}$
3. **Sample uniformly:** $x' \sim \text{Uniform}(S)$

**Stepping-out procedure:**

1. Initial interval: $[L, R] = [x - w \cdot U, x + w \cdot (1-U)]$ where $U \sim \text{Uniform}(0,1)$
2. Expand left while $\pi(L) > u$: $L \leftarrow L - w$
3. Expand right while $\pi(R) > u$: $R \leftarrow R + w$
4. Sample $x' \sim \text{Uniform}(L, R)$, shrink if $\pi(x') < u$

### 6.3 Gibbs Sampling

**For multivariate distributions with tractable conditionals.**

**Full conditional:** $p(x_j \mid x_{-j}) = p(x_j \mid x_1, \ldots, x_{j-1}, x_{j+1}, \ldots, x_d)$

**Algorithm (Systematic Scan):**

For $j = 1, \ldots, d$:
$$x_j^{(t+1)} \sim p(x_j \mid x_1^{(t+1)}, \ldots, x_{j-1}^{(t+1)}, x_{j+1}^{(t)}, \ldots, x_d^{(t)})$$

**Random Scan:**

- Select dimension $j$ uniformly at random
- Update $x_j$ from full conditional

### 6.4 Hamiltonian Monte Carlo (HMC)

**Augments state with momentum for efficient exploration.**

**Hamiltonian:**

$$H(x, p) = U(x) + K(p) = -\log \pi(x) + \frac{1}{2} p^\top M^{-1} p$$

**Hamilton's equations:**

$$\frac{dx}{dt} = \frac{\partial H}{\partial p} = M^{-1} p$$

$$\frac{dp}{dt} = -\frac{\partial H}{\partial x} = \nabla \log \pi(x)$$

**Leapfrog integrator:**

$$p_{t+\epsilon/2} = p_t + \frac{\epsilon}{2} \nabla \log \pi(x_t)$$
$$x_{t+\epsilon} = x_t + \epsilon M^{-1} p_{t+\epsilon/2}$$
$$p_{t+\epsilon} = p_{t+\epsilon/2} + \frac{\epsilon}{2} \nabla \log \pi(x_{t+\epsilon})$$

**Algorithm:**

1. Sample momentum: $p \sim \mathcal{N}(0, M)$
2. Run $L$ leapfrog steps with step size $\epsilon$
3. Accept with probability:
   $$\alpha = \min(1, \exp(-H(x', p') + H(x, p)))$$

### 6.5 No-U-Turn Sampler (NUTS)

**Automatically tunes trajectory length in HMC.**

**U-turn criterion:**

Stop when trajectory starts returning:

$$(x^+ - x^-) \cdot p^+ < 0 \quad \text{or} \quad (x^+ - x^-) \cdot p^- < 0$$

where $(x^-, p^-)$ and $(x^+, p^+)$ are endpoints of the trajectory.

**Algorithm (Simplified):**

1. Build tree by doubling: repeatedly double trajectory in random direction
2. Stop when U-turn detected or max depth reached
3. Sample uniformly from valid states in trajectory

**Multinomial sampling from trajectory:**

Weight each state by $\exp(-H(x, p))$ and sample proportionally.

---

## 7. Bootstrap Methods

### 7.1 Ordinary Bootstrap (Efron, 1979)

**Given data:** $\mathbf{x} = (x_1, \ldots, x_n)$

**Bootstrap sample:**

$$\mathbf{x}^* = (x_{i_1}^*, \ldots, x_{i_n}^*)$$

where $i_j \sim \text{Uniform}\{1, \ldots, n\}$ i.i.d.

**Bootstrap estimate of statistic $\theta = T(\mathbf{x})$:**

$$\hat{\theta}^{*b} = T(\mathbf{x}^{*b}), \quad b = 1, \ldots, B$$

**Bootstrap variance:**

$$\widehat{\text{Var}}(\hat{\theta}) = \frac{1}{B-1} \sum_{b=1}^{B} (\hat{\theta}^{*b} - \bar{\theta}^*)^2$$

### 7.2 Block Bootstrap (for Time Series)

**Non-overlapping Block Bootstrap:**

1. Divide series into blocks of length $\ell$: $B_k = (x_{(k-1)\ell+1}, \ldots, x_{k\ell})$
2. Sample $\lceil n/\ell \rceil$ blocks with replacement
3. Concatenate to form bootstrap series

**Moving Block Bootstrap (Kunsch, 1989):**

- Overlapping blocks: $B_i = (x_i, x_{i+1}, \ldots, x_{i+\ell-1})$ for $i = 1, \ldots, n-\ell+1$
- Sample blocks with replacement

**Circular Block Bootstrap:**

- Wrap data: $x_{n+i} = x_i$
- All $n$ blocks of length $\ell$ are available

### 7.3 Stationary Bootstrap (Politis & Romano, 1994)

**Random block lengths with geometric distribution.**

**Block length:** $L \sim \text{Geometric}(p)$ where $\mathbb{E}[L] = 1/p$

**Algorithm:**

1. Sample starting point $i \sim \text{Uniform}\{1, \ldots, n\}$
2. Sample block length $L \sim \text{Geometric}(p)$
3. Take block $(x_i, x_{i+1}, \ldots, x_{i+L-1})$ (circular)
4. Repeat until bootstrap series has length $n$

**Property:** Produces stationary bootstrap series.

### 7.4 Wild Bootstrap (Wu, 1986)

**For heteroscedastic regression residuals.**

**Given residuals** $\hat{\varepsilon}_i$:

$$\varepsilon_i^* = \hat{\varepsilon}_i \cdot v_i$$

where $v_i$ are i.i.d. with $\mathbb{E}[v] = 0$, $\mathbb{E}[v^2] = 1$.

**Common choices for $v$:**

- **Rademacher:** $P(v = 1) = P(v = -1) = 0.5$
- **Mammen:** $P(v = \frac{1-\sqrt{5}}{2}) = \frac{1+\sqrt{5}}{2\sqrt{5}}$, $P(v = \frac{1+\sqrt{5}}{2}) = 1 - \frac{1+\sqrt{5}}{2\sqrt{5}}$

### 7.5 Residual Bootstrap

**For regression:** $y_i = f(x_i; \hat{\beta}) + \hat{\varepsilon}_i$

**Algorithm:**

1. **Center residuals:** $\tilde{\varepsilon}_i = \hat{\varepsilon}_i - \bar{\varepsilon}$
2. **Resample residuals:** $\varepsilon_i^* \sim \text{EmpiricalDist}(\tilde{\varepsilon}_1, \ldots, \tilde{\varepsilon}_n)$
3. **Construct bootstrap data:** $y_i^* = f(x_i; \hat{\beta}) + \varepsilon_i^*$
4. **Refit model:** $\hat{\beta}^* = \arg\min \sum_i (y_i^* - f(x_i; \beta))^2$

### 7.6 Parametric Bootstrap

**Assume parametric model:** $x_i \sim F_\theta$

**Algorithm:**

1. **Estimate parameters:** $\hat{\theta} = \hat{\theta}(\mathbf{x})$
2. **Generate bootstrap sample:** $x_i^* \sim F_{\hat{\theta}}$
3. **Compute statistic:** $\hat{\theta}^* = T(\mathbf{x}^*)$

**Example (Normal):**

$$\hat{\mu} = \bar{x}, \quad \hat{\sigma}^2 = s^2$$
$$x_i^* \sim \mathcal{N}(\hat{\mu}, \hat{\sigma}^2)$$

### 7.7 Bootstrap Confidence Intervals

**Percentile method:**

$$CI_{1-\alpha} = [\hat{\theta}^*_{(\alpha/2)}, \hat{\theta}^*_{(1-\alpha/2)}]$$

**Basic (reverse percentile) method:**

$$CI_{1-\alpha} = [2\hat{\theta} - \hat{\theta}^*_{(1-\alpha/2)}, 2\hat{\theta} - \hat{\theta}^*_{(\alpha/2)}]$$

**BCa (Bias-Corrected and Accelerated):**

$$CI_{1-\alpha} = [\hat{\theta}^*_{(\alpha_1)}, \hat{\theta}^*_{(\alpha_2)}]$$

where:

$$\alpha_1 = \Phi\left(\hat{z}_0 + \frac{\hat{z}_0 + z_{\alpha/2}}{1 - \hat{a}(\hat{z}_0 + z_{\alpha/2})}\right)$$

$$\alpha_2 = \Phi\left(\hat{z}_0 + \frac{\hat{z}_0 + z_{1-\alpha/2}}{1 - \hat{a}(\hat{z}_0 + z_{1-\alpha/2})}\right)$$

- $\hat{z}_0 = \Phi^{-1}(\text{proportion of } \hat{\theta}^* < \hat{\theta})$ (bias correction)
- $\hat{a} = \frac{\sum_i (\bar{\theta}_{(\cdot)} - \hat{\theta}_{(-i)})^3}{6[\sum_i (\bar{\theta}_{(\cdot)} - \hat{\theta}_{(-i)})^2]^{3/2}}$ (acceleration)

### 7.8 Jackknife

**Leave-one-out resampling.**

**Jackknife estimate:**

$$\hat{\theta}_{(-i)} = T(x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_n)$$

**Jackknife variance:**

$$\widehat{\text{Var}}_{\text{jack}}(\hat{\theta}) = \frac{n-1}{n} \sum_{i=1}^{n} (\hat{\theta}_{(-i)} - \bar{\theta}_{(\cdot)})^2$$

where $\bar{\theta}_{(\cdot)} = \frac{1}{n} \sum_i \hat{\theta}_{(-i)}$.

**Bias estimation:**

$$\widehat{\text{Bias}} = (n-1)(\bar{\theta}_{(\cdot)} - \hat{\theta})$$

---

## 8. Importance Sampling

### 8.1 Basic Importance Sampling

**Goal:** Estimate $\mathbb{E}_\pi[f(X)]$ when sampling from $\pi$ is difficult.

**Key identity:**

$$\mathbb{E}_\pi[f(X)] = \int f(x) \pi(x) dx = \int f(x) \frac{\pi(x)}{q(x)} q(x) dx = \mathbb{E}_q\left[f(X) \frac{\pi(X)}{q(X)}\right]$$

**Importance weights:**

$$w(x) = \frac{\pi(x)}{q(x)}$$

**Estimator:**

$$\hat{\mu}_{\text{IS}} = \frac{1}{N} \sum_{i=1}^{N} w(x^{(i)}) f(x^{(i)}), \quad x^{(i)} \sim q$$

### 8.2 Self-Normalized Importance Sampling

**When $\pi$ is known only up to normalizing constant:** $\pi(x) = \tilde{\pi}(x) / Z$

**Self-normalized weights:**

$$\tilde{w}^{(i)} = \frac{\tilde{\pi}(x^{(i)})}{q(x^{(i)})}, \quad W^{(i)} = \frac{\tilde{w}^{(i)}}{\sum_j \tilde{w}^{(j)}}$$

**Estimator:**

$$\hat{\mu}_{\text{SNIS}} = \sum_{i=1}^{N} W^{(i)} f(x^{(i)})$$

**Properties:**
- Biased but consistent
- Does not require knowing $Z$
- Lower variance when $q \approx \pi$

### 8.3 Multiple Importance Sampling (MIS)

**Multiple proposal distributions:** $q_1, \ldots, q_M$

**Balance heuristic (Veach & Guibas, 1995):**

$$w_j(x) = \frac{\pi(x)}{\sum_{k=1}^{M} n_k q_k(x)}$$

**Estimator:**

$$\hat{\mu}_{\text{MIS}} = \sum_{j=1}^{M} \frac{1}{n_j} \sum_{i=1}^{n_j} w_j(x_j^{(i)}) f(x_j^{(i)})$$

**Provably good:** minimizes variance among a broad class of estimators.

### 8.4 Adaptive Importance Sampling

**Iteratively adapt proposal to target.**

**Cross-entropy method:**

1. Initialize proposal $q_0$ (e.g., prior)
2. Sample: $x^{(i)} \sim q_t$
3. Compute weights: $w^{(i)} \propto \pi(x^{(i)}) / q_t(x^{(i)})$
4. Update proposal: $q_{t+1} = \arg\min_{q \in \mathcal{Q}} \sum_i w^{(i)} \log q(x^{(i)})$

**For Gaussian proposals:**

$$\mu_{t+1} = \frac{\sum_i w^{(i)} x^{(i)}}{\sum_i w^{(i)}}$$

$$\Sigma_{t+1} = \frac{\sum_i w^{(i)} (x^{(i)} - \mu_{t+1})(x^{(i)} - \mu_{t+1})^\top}{\sum_i w^{(i)}}$$

### 8.5 Effective Sample Size for IS

$$\text{ESS}_{\text{IS}} = \frac{(\sum_i w^{(i)})^2}{\sum_i (w^{(i)})^2} = \frac{1}{\sum_i (W^{(i)})^2}$$

For self-normalized weights, this measures how many i.i.d. samples from $\pi$ would give equivalent information.

### 8.6 IS Diagnostics

**Coefficient of Variation:**

$$\text{CV} = \frac{\text{Std}(w)}{\text{Mean}(w)} = \sqrt{\frac{\sum_i (w^{(i)} - \bar{w})^2 / N}{\bar{w}}}$$

**Kurtosis of weights:**

$$\text{Kurt} = \frac{\mathbb{E}[(w - \bar{w})^4]}{(\mathbb{E}[(w - \bar{w})^2])^2}$$

High kurtosis indicates heavy-tailed weights (poor proposal).

---

## 9. Quasi-Monte Carlo

### 9.1 Discrepancy

**Star discrepancy** measures uniformity:

$$D_N^*(\mathbf{x}_1, \ldots, \mathbf{x}_N) = \sup_{\mathbf{u} \in [0,1]^d} \left| \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}_{[0,\mathbf{u})}(\mathbf{x}_i) - \prod_{j=1}^{d} u_j \right|$$

**Koksma-Hlawka inequality:**

$$\left| \frac{1}{N} \sum_{i=1}^{N} f(\mathbf{x}_i) - \int_{[0,1]^d} f(\mathbf{x}) d\mathbf{x} \right| \leq V_{\text{HK}}(f) \cdot D_N^*$$

where $V_{\text{HK}}(f)$ is the Hardy-Krause variation.

### 9.2 Halton Sequence

**Van der Corput sequence** in base $b$:

For integer $n$ with base-$b$ representation $n = \sum_{j=0}^{k} a_j b^j$:

$$\phi_b(n) = \sum_{j=0}^{k} a_j b^{-(j+1)}$$

**Halton sequence in $d$ dimensions:**

$$\mathbf{x}_n = (\phi_{p_1}(n), \phi_{p_2}(n), \ldots, \phi_{p_d}(n))$$

where $p_1, p_2, \ldots, p_d$ are the first $d$ primes.

**Discrepancy:** $D_N^* = O((\log N)^d / N)$

### 9.3 Sobol Sequence

**Uses direction numbers and Gray code.**

**Direction numbers:** $v_1, v_2, \ldots$ (dimension-specific)

**Gray code:** $G(n) = n \oplus (n >> 1)$

**Sobol point:**

$$x_n = \bigoplus_{j: \text{bit } j \text{ of } G(n) = 1} v_j$$

(XOR of direction numbers corresponding to 1-bits in Gray code)

**Properties:**
- Better uniformity than Halton for moderate $d$
- Discrepancy: $D_N^* = O((\log N)^d / N)$

### 9.4 Latin Hypercube Sampling (LHS)

**Stratified sampling ensuring marginal uniformity.**

**Algorithm:**

1. Divide $[0,1]$ into $N$ equal strata for each dimension
2. For each dimension $j$:
   - Generate random permutation $\pi_j$ of $\{1, \ldots, N\}$
   - Set $x_{ij} = (\pi_j(i) - U_{ij}) / N$ where $U_{ij} \sim \text{Uniform}(0,1)$

**Properties:**
- Each marginal is uniformly distributed
- Each stratum contains exactly one point per dimension
- Variance reduction for additive functions

### 9.5 Randomized Quasi-Monte Carlo

**Cranley-Patterson rotation:**

$$\tilde{\mathbf{x}}_i = (\mathbf{x}_i + \mathbf{u}) \mod 1$$

where $\mathbf{u} \sim \text{Uniform}([0,1]^d)$.

**Properties:**
- Preserves low discrepancy
- Enables unbiased estimation
- Allows variance estimation via independent shifts

---

## 10. State-Space Models

### 10.1 Local Level Model (Random Walk + Noise)

**State equation:**

$$x_t = x_{t-1} + \eta_t, \quad \eta_t \sim \mathcal{N}(0, \sigma_\eta^2)$$

**Observation equation:**

$$y_t = x_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma_\varepsilon^2)$$

**Signal-to-noise ratio:** $q = \sigma_\eta^2 / \sigma_\varepsilon^2$

### 10.2 Local Linear Trend Model

**State:** $\mathbf{x}_t = (\mu_t, \nu_t)^\top$ (level and trend)

**State equations:**

$$\mu_t = \mu_{t-1} + \nu_{t-1} + \eta_t$$
$$\nu_t = \nu_{t-1} + \zeta_t$$

**Observation:**

$$y_t = \mu_t + \varepsilon_t$$

**State-space form:**

$$\mathbf{x}_t = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix} \mathbf{x}_{t-1} + \mathbf{w}_t$$

$$y_t = \begin{pmatrix} 1 & 0 \end{pmatrix} \mathbf{x}_t + \varepsilon_t$$

### 10.3 ARIMA(p,d,q)

**ARIMA process:**

$$\phi(B)(1-B)^d y_t = \theta(B) \varepsilon_t$$

where:
- $\phi(B) = 1 - \phi_1 B - \cdots - \phi_p B^p$ (AR polynomial)
- $\theta(B) = 1 + \theta_1 B + \cdots + \theta_q B^q$ (MA polynomial)
- $B$ is the backshift operator: $B y_t = y_{t-1}$

**State-space representation** (dimension $r = \max(p, q+1)$):

$$\mathbf{x}_t = F \mathbf{x}_{t-1} + \mathbf{g} \varepsilon_t$$

$$y_t = \mathbf{h}^\top \mathbf{x}_t$$

### 10.4 Stochastic Volatility Model

**Log-volatility:**

$$h_t = \mu + \phi(h_{t-1} - \mu) + \sigma_\eta \eta_t, \quad \eta_t \sim \mathcal{N}(0, 1)$$

**Returns:**

$$r_t = \exp(h_t / 2) \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, 1)$$

**Alternative (log-squared returns):**

$$\log r_t^2 = h_t + \log \varepsilon_t^2$$

where $\log \varepsilon_t^2$ follows a $\log \chi^2_1$ distribution (approximated by mixture of normals).

### 10.5 GARCH(1,1)

**Conditional variance:**

$$\sigma_t^2 = \omega + \alpha r_{t-1}^2 + \beta \sigma_{t-1}^2$$

**Returns:**

$$r_t = \sigma_t \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, 1)$$

**Constraints:**
- $\omega > 0$, $\alpha \geq 0$, $\beta \geq 0$
- $\alpha + \beta < 1$ (stationarity)

**Unconditional variance:** $\mathbb{E}[\sigma_t^2] = \omega / (1 - \alpha - \beta)$

### 10.6 GJR-GARCH (Asymmetric)

**Conditional variance with leverage:**

$$\sigma_t^2 = \omega + (\alpha + \gamma \mathbf{1}_{r_{t-1} < 0}) r_{t-1}^2 + \beta \sigma_{t-1}^2$$

**Leverage effect:** $\gamma > 0$ means negative returns increase volatility more than positive returns.

### 10.7 Dynamic Factor Model

**Observations:**

$$\mathbf{y}_t = \Lambda \mathbf{f}_t + \boldsymbol{\varepsilon}_t, \quad \boldsymbol{\varepsilon}_t \sim \mathcal{N}(\mathbf{0}, \Sigma_\varepsilon)$$

**Factor dynamics:**

$$\mathbf{f}_t = A \mathbf{f}_{t-1} + \boldsymbol{\eta}_t, \quad \boldsymbol{\eta}_t \sim \mathcal{N}(\mathbf{0}, \Sigma_\eta)$$

where:
- $\mathbf{y}_t \in \mathbb{R}^{d_y}$: observed variables
- $\mathbf{f}_t \in \mathbb{R}^{k}$: latent factors ($k \ll d_y$)
- $\Lambda \in \mathbb{R}^{d_y \times k}$: factor loadings

### 10.8 Regime-Switching Model

**Regime variable:** $s_t \in \{1, \ldots, K\}$

**Transition probabilities:**

$$P(s_t = j \mid s_{t-1} = i) = p_{ij}$$

**Regime-dependent dynamics:**

$$y_t \mid s_t = k \sim p(y_t \mid x_t, \theta_k)$$
$$x_t \mid s_t = k \sim p(x_t \mid x_{t-1}, \theta_k)$$

**Hamilton filter** for regime inference:

$$P(s_t = j \mid y_{1:t}) \propto p(y_t \mid s_t = j) \sum_{i=1}^{K} p_{ij} P(s_{t-1} = i \mid y_{1:t-1})$$

---

## References

1. Gordon, N. J., Salmond, D. J., & Smith, A. F. (1993). Novel approach to nonlinear/non-Gaussian Bayesian state estimation. *IEE Proceedings F*, 140(2), 107-113.

2. Pitt, M. K., & Shephard, N. (1999). Filtering via simulation: Auxiliary particle filters. *Journal of the American Statistical Association*, 94(446), 590-599.

3. Liu, J., & West, M. (2001). Combined parameter and state estimation in simulation-based filtering. In *Sequential Monte Carlo Methods in Practice* (pp. 197-223). Springer.

4. Chopin, N., Jacob, P. E., & Papaspiliopoulos, O. (2013). SMC2: an efficient algorithm for sequential analysis of state space models. *Journal of the Royal Statistical Society: Series B*, 75(3), 397-426.

5. Andrieu, C., Doucet, A., & Holenstein, R. (2010). Particle Markov chain Monte Carlo methods. *Journal of the Royal Statistical Society: Series B*, 72(3), 269-342.

6. Dau, H. D., & Chopin, N. (2022). Waste-free sequential Monte Carlo. *Journal of the Royal Statistical Society: Series B*, 84(1), 114-148.

7. Neal, R. M. (2003). Slice sampling. *Annals of Statistics*, 31(3), 705-767.

8. Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo. *Journal of Machine Learning Research*, 15(1), 1593-1623.

9. Efron, B. (1979). Bootstrap methods: another look at the jackknife. *Annals of Statistics*, 7(1), 1-26.

10. Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428), 1303-1313.

11. Owen, A. B. (2013). *Monte Carlo theory, methods and examples*. Stanford University.

12. Chopin, N., & Papaspiliopoulos, O. (2020). *An Introduction to Sequential Monte Carlo*. Springer.
