# Adstock Transformations

## The Carryover Effect

Advertising effects do not occur instantaneously. When a consumer sees an ad today, they may not purchase until days or weeks later. This delayed and persistent effect is called **carryover** or **adstock**.

---

## Why Adstock Matters

Without adstock transformation:

- Attribution goes entirely to the week of purchase
- Delayed response is ignored
- ROI estimates are biased

With adstock:

- Effects are spread across time
- Delayed conversions are attributed correctly
- More accurate channel comparison

---

## Geometric Adstock

The most common adstock formulation.

### Formula

```
Adstock(t) = Spend(t) + λ × Adstock(t-1)
```

Or equivalently:

```
Adstock(t) = Σ_{i=0}^{L} λⁱ × Spend(t-i)
```

Where:

- `λ` = decay rate (0 to 1)
- `L` = maximum lag (l_max)

### Properties

| λ Value | Half-Life | Interpretation |
|---------|-----------|----------------|
| 0.9 | 6.6 weeks | Strong carryover (TV, brand) |
| 0.7 | 2.0 weeks | Moderate carryover |
| 0.5 | 1.0 week | Short carryover (search) |
| 0.3 | 0.5 weeks | Minimal carryover |

### Half-Life Calculation

The **half-life** H is the time at which the effect decays to 50%.

**Derivation:** For geometric decay, the effect at lag i is λᶦ. We solve for H where:

```text
λᴴ = 0.5
H × log(λ) = log(0.5)
H = log(0.5) / log(λ)
```

**Verification:**

| λ | H = log(0.5)/log(λ) | Check: λᴴ |
|-----|---------------------|----------|
| 0.9 | 6.58 | 0.9^6.58 ≈ 0.5 ✓ |
| 0.7 | 1.94 | 0.7^1.94 ≈ 0.5 ✓ |
| 0.5 | 1.00 | 0.5^1.00 = 0.5 ✓ |

---

### Delayed Adstock (Weibull-Based)

For channels with delayed peak effects, weights are derived from the Weibull distribution:

```text
Adstock(t) = Σ_{i=0}^{L} w(i) × Spend(t-i)
```

where the weights w(i) are proportional to the Weibull PDF:

```text
w(i) ∝ (k/λ) × (i/λ)^(k-1) × exp(-(i/λ)^k)  for i > 0
```

**Parameters:**

| Parameter | Name | Effect |
|-----------|------|--------|
| λ (lambda) | Scale | Controls when peak occurs |
| k (shape) | Shape | k < 1: peak at 0; k > 1: delayed peak |

**Normalization:** Weights are normalized so Σw(i) = 1.

> **Note on PyMC-Marketing:** Uses `DelayedAdstock` with parameters `alpha` (decay) and `theta` (delay). The implementation differs slightly from the standard Weibull PDF form.

---

## Choosing Adstock Parameters

### Prior Knowledge by Channel

| Channel | Expected Decay | Rationale |
|---------|----------------|-----------|
| TV | 0.7 - 0.9 | Brand building, long memory |
| Social | 0.5 - 0.7 | Content remains visible |
| Search | 0.2 - 0.5 | Intent-based, immediate |
| Display | 0.4 - 0.6 | Awareness, moderate decay |
| Email | 0.3 - 0.5 | Short attention span |

### Learning from Data

In Bayesian MMM, λ is a parameter with a prior:

```
λ ~ Beta(2, 2)
```

The posterior provides data-driven estimates with uncertainty.

---

## Implementation in PyMC-Marketing

```python
from pymc_marketing.mmm import GeometricAdstock

adstock = GeometricAdstock(l_max=8)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `l_max` | 8 | Maximum lag in time periods |
| Prior on α | Beta(1, 3) | Favors lower decay values |

> **Note**: PyMC-Marketing uses Beta(1, 3) by default, which favors faster decay (λ ≈ 0.25). The literature sometimes uses Beta(2, 2) for symmetric priors. Adjust based on domain knowledge.

---

## Visualizing Adstock

### Decay Curve Code

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_adstock_decay(lambdas, l_max=8):
    """Visualize adstock decay for different lambda values."""
    weeks = np.arange(l_max + 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for lam in lambdas:
        decay = lam ** weeks
        ax.plot(weeks, decay * 100, marker='o', label=f'λ = {lam}')
    
    ax.set_xlabel('Weeks After Spend')
    ax.set_ylabel('Effect Remaining (%)')
    ax.set_title('Adstock Decay Curves')
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    return fig

# Example usage
plot_adstock_decay([0.3, 0.5, 0.7, 0.9])
plt.show()
```

### Decay Examples

For λ = 0.7:

| Week | Effect Remaining |
|------|------------------|
| 0 | 100% |
| 1 | 70% |
| 2 | 49% |
| 3 | 34% |
| 4 | 24% |

### Cumulative Effect

Total effect of $1 spent over infinite time:

```text
Total = 1 + λ + λ² + λ³ + ... = 1 / (1 - λ)
```

| λ | Cumulative Multiplier |
|-----|----------------------|
| 0.3 | 1.43× |
| 0.5 | 2.00× |
| 0.7 | 3.33× |
| 0.9 | 10.00× |

---

## Common Pitfalls

### 1. Ignoring Adstock

Leads to underestimation of long-term channels (TV, brand).

### 2. Same Decay for All Channels

Different channels have different decay patterns. Allow per-channel parameters.

### 3. Too Short l_max

If l_max = 4 but true effects last 12 weeks, effects are truncated.

### 4. Confusing Decay with Noise

High decay estimates may indicate poor model fit, not true carryover.

---

## Mathematical Properties

### Linearity

Adstock is a linear transformation:

```
Adstock(a × Spend₁ + b × Spend₂) = a × Adstock(Spend₁) + b × Adstock(Spend₂)
```

### Stationarity

For geometric adstock with constant λ, the transformation is time-invariant.

### Invertibility

Given adstock series and λ, original spend can be recovered:

```
Spend(t) = Adstock(t) - λ × Adstock(t-1)
```

---

## References

- Broadbent, S. (1979). One Way TV Advertisements Work. Journal of the Market Research Society.
- Jin, Y., et al. (2017). Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects.
