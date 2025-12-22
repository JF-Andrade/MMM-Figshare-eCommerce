# Mathematical Foundations

## Overview

This section covers the mathematical foundations required to understand Bayesian Marketing Mix Models, including probability theory, Bayesian inference, and Markov Chain Monte Carlo sampling.

---

## 1. Probability Fundamentals

### Random Variables and Distributions

A random variable X maps outcomes to real numbers. Its behavior is characterized by a probability distribution.

**Discrete:** P(X = x) = probability mass function (PMF)

**Continuous:** f(x) = probability density function (PDF)

### Common Distributions in MMM

| Distribution | Use in MMM | Parameters |
|--------------|------------|------------|
| Normal | Noise, priors | μ (mean), σ (std) |
| Half-Normal | Positive parameters | σ |
| Beta | Decay rates (0-1) | α, β |
| Gamma | Scale parameters | α, β |

---

## 2. Bayesian Inference

### Bayes' Theorem

The foundation of Bayesian statistics:

```text
P(θ|D) = P(D|θ) × P(θ) / P(D)
```

Where:

- `P(θ|D)` = **Posterior**: Belief about parameters after seeing data
- `P(D|θ)` = **Likelihood**: Probability of data given parameters
- `P(θ)` = **Prior**: Belief about parameters before seeing data
- `P(D)` = **Evidence**: Normalizing constant

### Posterior Proportionality

Since P(D) is constant with respect to θ:

```text
Posterior ∝ Likelihood × Prior
```

### Example: Normal-Normal Model

**Prior:** θ ~ Normal(μ₀, σ₀²)

**Likelihood:** y|θ ~ Normal(θ, σ²)

**Posterior:** θ|y ~ Normal(μₙ, σₙ²)

Where:

```text
μₙ = (σ² × μ₀ + σ₀² × ȳ × n) / (σ² + n × σ₀²)
σₙ² = (σ² × σ₀²) / (σ² + n × σ₀²)
```

---

## 3. Prior Selection

### Informative vs Weakly Informative Priors

| Type | Description | When to Use |
|------|-------------|-------------|
| Informative | Strong belief encoded | Domain expertise available |
| Weakly Informative | Regularizes but allows data to dominate | Default choice |
| Non-informative | Flat/uniform | Rare, can cause issues |

### Priors in MMM

**Adstock decay (λ):** Beta(2, 2) - Centered around 0.5, symmetric

**Saturation parameters:** Half-Normal - Positive, concentrated near zero

**Regression coefficients:** Normal(0, σ) - Centered at zero, σ controls regularization

---

## 4. Linear Regression as Bayesian Model

### Standard Form

```text
y = Xβ + ε
ε ~ Normal(0, σ²)
```

### Bayesian Formulation

**Likelihood:**

```text
y|β, σ ~ Normal(Xβ, σ²I)
```

**Priors:**

```text
β ~ Normal(0, τ²I)
σ ~ HalfNormal(s)
```

### Posterior

The posterior distribution over β and σ given the data y and X.

---

## 5. Markov Chain Monte Carlo (MCMC)

### The Problem

For most Bayesian models, the posterior cannot be computed analytically. We need numerical methods to approximate it.

### Solution: Sampling

Draw samples from the posterior distribution. With enough samples, we can approximate any posterior quantity.

### Markov Chain

A sequence of random variables where each depends only on the previous:

```text
P(θₜ₊₁|θ₁, θ₂, ..., θₜ) = P(θₜ₊₁|θₜ)
```

### Metropolis-Hastings Algorithm

1. Start at initial point θ₀
2. Propose new point θ*from proposal distribution q(θ*|θₜ)
3. Compute acceptance probability:

```text
α = min(1, [P(θ*|D) × q(θₜ|θ*)] / [P(θₜ|D) × q(θ*|θₜ)])
```

4. Accept with probability α: if accepted, θₜ₊₁ = θ*; else θₜ₊₁ = θₜ
5. Repeat

> **Note**: For symmetric proposals where q(θ*|θₜ) = q(θₜ|θ*), the ratio simplifies to:
>
> ```text
> α = min(1, P(θ*|D) / P(θₜ|D))
> ```
>
> This special case is called the **Metropolis algorithm**.

### Hamiltonian Monte Carlo (HMC)

Uses gradient information to propose better moves. More efficient than Metropolis-Hastings for high-dimensional problems.

### No-U-Turn Sampler (NUTS)

Adaptive variant of HMC that automatically tunes step size and trajectory length. Used by PyMC and Stan.

---

## 6. Convergence Diagnostics

### R-hat (Gelman-Rubin Statistic)

Compares variance within chains to variance between chains. For M chains of length N:

**Between-chain variance:**

```text
B = (N / (M-1)) × Σₘ (θ̅ₘ - θ̅)²
```

**Within-chain variance:**

```text
W = (1/M) × Σₘ sₘ²
```

**R-hat formula:**

```text
R̂ = √[((N-1)/N) × W + (1/N) × B) / W]
   = √[(N-1)/N + B/(N×W)]
```

**Interpretation:**

| R-hat | Status |
|-------|--------|
| 1.00 | Perfect convergence |
| < 1.01 | Excellent |
| < 1.05 | Acceptable |
| > 1.05 | Potential issues |
| > 1.1 | Major problems |

### Effective Sample Size (ESS)

Accounts for autocorrelation in MCMC samples.

```text
ESS = n / (1 + 2 × Σρₖ)
```

Where ρₖ is autocorrelation at lag k.

**Rule of thumb:** ESS > 400 per parameter

### Divergences

Numerical instabilities during sampling. Indicate:

- Model misspecification
- Pathological posterior geometry
- Need for reparameterization

**Target:** Zero divergences

---

## 7. The MMM Model in Bayesian Form

### Generative Model

```text
# Priors
β_channels ~ Normal(0, 1)
β_controls ~ Normal(0, 1)
σ ~ HalfNormal(1)
α_adstock ~ Beta(2, 2)
λ_saturation ~ HalfNormal(1)

# Transformations
adstocked = adstock(spend, α)
saturated = saturation(adstocked, λ)

# Likelihood
μ = intercept + saturated @ β_channels + controls @ β_controls
y ~ Normal(μ, σ)
```

### Posterior Inference

MCMC samples from:

```text
P(β, σ, α, λ | y, X)
```

---

## 8. Key Mathematical Insights

1. **Regularization through priors**: Priors on β prevent overfitting, similar to Ridge/LASSO

2. **Uncertainty propagation**: Posterior distributions capture uncertainty in all parameters

3. **Hierarchical structure**: Group-level priors enable partial pooling across regions

4. **Transformation interpretability**: Adstock and saturation parameters have direct business meaning

---

## References

- Gelman, A., et al. (2013). Bayesian Data Analysis. CRC Press.
- McElreath, R. (2020). Statistical Rethinking. CRC Press.
- Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo.
