# MCMC Sampling and Diagnostics

## Overview

Markov Chain Monte Carlo (MCMC) is the computational engine behind Bayesian MMM. Understanding MCMC diagnostics is essential for reliable inference.

---

## MCMC Basics

### What MCMC Does

MCMC generates samples from the posterior distribution:

```
θ₁, θ₂, ..., θₙ ~ P(θ|Data)
```

With enough samples, we can approximate:

- Posterior means
- Credible intervals
- Any function of parameters

### Why MCMC is Necessary

For most Bayesian models, the posterior cannot be computed analytically. MCMC provides a numerical approximation.

---

## The NUTS Sampler

### No-U-Turn Sampler

NUTS is an adaptive variant of Hamiltonian Monte Carlo (HMC) that automatically tunes:

- Step size
- Number of leapfrog steps

### Advantages

- No manual tuning required
- Efficient for high-dimensional problems
- Used by default in PyMC

---

## GPU Acceleration with Numpyro

### Why GPU?

MCMC is computationally intensive. GPU acceleration can provide 10-100x speedups for large models.

### Numpyro Backend

```python
mmm.fit(
    ...,
    nuts_sampler="numpyro",
)
```

Numpyro uses JAX for:

- Just-in-time compilation
- Automatic differentiation
- GPU/TPU execution

### Google Colab Setup (Recommended)

**Step 1**: Enable GPU runtime

- Runtime → Change runtime type → T4 GPU

**Step 2**: Install JAX with CUDA support

```python
# Install JAX with GPU support (must be first!)
!pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install PyMC ecosystem
!pip install numpyro pymc-marketing arviz
```

**Step 3**: Verify GPU access

```python
import jax
print(f"JAX devices: {jax.devices()}")
# Should show: [CudaDevice(id=0)]

# Configure numpyro for multi-chain sampling
import numpyro
numpyro.set_host_device_count(4)  # Must be before model fitting
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| `jax.devices()` shows CPU | Restart runtime after GPU install |
| Out of memory | Reduce `draws` or `chains` |
| Slow first iteration | JAX JIT compilation (normal) |
| `XlaRuntimeError` | Restart runtime, reduce model size |

---

## Convergence Diagnostics

### R-hat (Gelman-Rubin Statistic)

Compares variance within and between chains.

**Formula:**

```
R-hat = sqrt((n-1)/n + B/(n×W))
```

Where:

- B = between-chain variance
- W = within-chain variance
- n = number of samples

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

**Types:**

- `ess_bulk`: Efficiency for central tendency
- `ess_tail`: Efficiency for tail quantiles

**Interpretation:**

| ESS | Status |
|-----|--------|
| > 400 | Acceptable |
| > 1000 | Good |
| > 4000 | Excellent |
| < 100 | Problematic |

### Divergences

Numerical instabilities during sampling that indicate:

- Difficult posterior geometry
- Model misspecification
- Need for reparameterization

**Target:** Zero divergences

---

## Checking Convergence in PyMC-Marketing

### Summary Statistics

```python
import arviz as az

summary = az.summary(mmm.idata, var_names=["~lam"], filter_vars="like")

print(f"Max R-hat: {summary['r_hat'].max():.4f}")
print(f"Min ESS: {summary['ess_bulk'].min():.0f}")
```

### Divergence Count

```python
divergences = int(mmm.idata.sample_stats["diverging"].sum())
print(f"Divergences: {divergences}")
```

### Full Diagnostic Check

```python
def check_convergence(mmm):
    summary = az.summary(mmm.idata, var_names=["~lam"], filter_vars="like")
    
    max_rhat = summary["r_hat"].max()
    min_ess = summary["ess_bulk"].min()
    divergences = int(mmm.idata.sample_stats["diverging"].sum())
    
    print(f"R-hat max: {max_rhat:.4f} (< 1.05)")
    print(f"ESS min: {min_ess:.0f} (> 400)")
    print(f"Divergences: {divergences} (= 0)")
    
    if max_rhat > 1.05:
        print("WARNING: R-hat too high")
    if min_ess < 400:
        print("WARNING: ESS too low")
    if divergences > 0:
        print("WARNING: Divergences detected")
```

---

## Trace Plots

### Visual Inspection

```python
az.plot_trace(mmm.idata, var_names=["intercept", "sigma"])
```

**Good trace:**

- Chains mix well (caterpillar pattern)
- No trends or drifts
- Consistent mean across chains

**Bad trace:**

- Chains stuck in different regions
- Trending behavior
- High autocorrelation

---

## Troubleshooting

### High R-hat

**Causes:**

- Not enough samples
- Poor initialization
- Model misspecification

**Solutions:**

1. Increase `draws` and `tune`
2. Increase `target_accept` to 0.99
3. Check model specification

### Low ESS

**Causes:**

- High autocorrelation
- Inefficient proposal

**Solutions:**

1. Increase number of samples
2. Reparameterize model
3. Use more informative priors

### Divergences

**Causes:**

- Steep gradients in posterior
- Funnel geometry
- Numerical instability

**Solutions:**

1. Increase `target_accept` to 0.95-0.99
2. Reparameterize (non-centered parameterization)
3. Use more informative priors

---

## Production MCMC Settings

### Recommended Configuration

```python
mmm.fit(
    X=X_train,
    y=y_train,
    chains=4,           # Multiple chains for R-hat
    draws=2000,         # Samples per chain
    tune=1500,          # Warmup samples
    target_accept=0.95, # Acceptance rate
    random_seed=42,     # Reproducibility
    nuts_sampler="numpyro",  # GPU acceleration
)
```

### Time Estimates

| Configuration | CPU | GPU (T4) |
|---------------|-----|----------|
| chains=2, draws=500 | 30 min | 5 min |
| chains=4, draws=2000 | 2-4 hours | 30-60 min |
| Hierarchical (18 regions) | 12+ hours | 1-2 hours |

---

## ArviZ Visualization

### Posterior Distributions

```python
az.plot_posterior(mmm.idata, var_names=["channel_effects"])
```

### Rank Plots

```python
az.plot_rank(mmm.idata, var_names=["sigma"])
```

### Pair Plot

```python
az.plot_pair(mmm.idata, var_names=["alpha", "lam"])
```

---

## References

- Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo.
- Vehtari, A., et al. (2021). Rank-Normalization, Folding, and Localization: An Improved R-hat.
- ArviZ Documentation: <https://python.arviz.org/>
