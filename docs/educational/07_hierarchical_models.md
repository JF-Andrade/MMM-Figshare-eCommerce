# Hierarchical Models

## The Multi-Region Challenge

When analyzing marketing data across multiple regions, we face a choice:

1. **Separate models**: Fit independent model per region
2. **Pooled model**: One model ignoring regional differences
3. **Hierarchical model**: Partial pooling across regions

---

## Understanding Pooling

### No Pooling (Separate Models)

Each region gets its own parameters:

```text
β_UK ~ Normal(0, σ)
β_US ~ Normal(0, σ)
β_DE ~ Normal(0, σ)
...
```

**Problem:** Small regions have high variance estimates.

### Complete Pooling (Single Model)

All regions share the same parameters:

```text
β ~ Normal(0, σ)  # Same for all regions
```

**Problem:** Ignores real regional differences.

### Partial Pooling (Hierarchical)

Parameters are drawn from a common distribution:

```text
μ_β ~ Normal(0, τ)           # Global mean
σ_β ~ HalfNormal(1)          # Regional variation
β_region ~ Normal(μ_β, σ_β)  # Region-specific
```

**Benefit:** Shrinkage toward common mean, especially for small samples.

---

## Shrinkage Estimation

### Concept

Regions with less data are "shrunk" toward the group mean. This reduces variance while preserving signal.

### Visual Intuition

```text
         Separate Estimates
Region A: ●─────────────────────────────●  (wide CI, few data)
Region B:          ●───────●                (narrow CI, more data)
Region C:    ●───────────●                  (medium CI)

         Hierarchical Estimates
Region A:        ●────────●                 (shrunk toward mean)
Region B:          ●───────●                (less change)
Region C:      ●────────●                   (moderate shrinkage)
```

### Shrinkage Factor

The amount of shrinkage depends on:

- Sample size (more data = less shrinkage)
- Within-group variance
- Between-group variance

---

## Mathematical Formulation

### Two-Level Hierarchy

**Level 1 (Observation):**

```text
y[t,r] ~ Normal(μ[t,r], σ)
μ[t,r] = intercept[r] + channels[t,r] @ β[r] + controls[t,r] @ γ[r]
```

**Level 2 (Region):**

```text
intercept[r] ~ Normal(μ_intercept, σ_intercept)
β[r] ~ Normal(μ_β, σ_β)
```

**Hyperpriors:**

```text
μ_intercept ~ Normal(0, 10)
σ_intercept ~ HalfNormal(1)
μ_β ~ Normal(0, 1)
σ_β ~ HalfNormal(0.5)
```

---

## Benefits of Hierarchical Models

### 1. Better Estimates for Small Regions

Borrowing strength from larger regions improves estimates for regions with limited data.

### 2. Regularization

The group-level distribution acts as a prior, preventing overfitting.

### 3. Interpretable Structure

Explicitly models:

- Common effects across regions
- Regional deviations from common patterns

### 4. Uncertainty Propagation

Uncertainty in hyperparameters propagates to regional estimates.

---

## Implementation in PyMC-Marketing

PyMC-Marketing >= 0.8.0 supports hierarchical MMM via the `dims` parameter.

### Data Preparation

```python
# Add geo column
data["geo"] = data["TERRITORY_NAME"]

# Ensure channels have regional variation
X = data[["week", "geo"] + channels + controls]
y = data["revenue"].values
```

### Model Creation

```python
mmm = MMM(
    date_column="week",
    channel_columns=channels,
    control_columns=controls,
    adstock=GeometricAdstock(l_max=8),
    saturation=LogisticSaturation(),
)
```

### Fitting

```python
mmm.fit(
    X=X,
    y=y,
    chains=4,
    draws=2000,
    tune=1500,
    nuts_sampler="numpyro",
)
```

---

## Regional ROI Computation

### Extract Region-Specific Contributions

```python
contributions = mmm.compute_channel_contribution_original_scale()

# Mean contribution by region and channel
mean_contrib = contributions.mean(dim=["chain", "draw"])
total_by_region = mean_contrib.sum(dim="date").groupby("geo")
```

### ROI by Region

```python
roi_data = []
mean_contrib = contributions.mean(dim=["chain", "draw"])
total_by_date = mean_contrib.sum(dim="date")

for region in regions:
    region_mask = X["geo"] == region
    region_data = X[region_mask]
    
    for channel in channels:
        spend = region_data[channel].sum()
        # Extract contribution for this region and channel
        contrib = float(total_by_date.sel(geo=region, channel=channel).values)
        roi = contrib / spend if spend > 0 else 0
        
        roi_data.append({
            "region": region,
            "channel": channel,
            "spend": spend,
            "contribution": contrib,
            "roi": roi,
        })

roi_df = pd.DataFrame(roi_data)
```

---

## Visualization

### Regional Comparison

```python
import seaborn as sns

pivot = roi_df.pivot(index="channel", columns="region", values="roi")
sns.heatmap(pivot, annot=True, cmap="RdYlGn", center=1.0)
```

### Shrinkage Plot

```python
# Compare hierarchical vs separate estimates
plt.scatter(separate_estimates, hierarchical_estimates)
plt.plot([0, 3], [0, 3], "k--")  # Identity line
plt.xlabel("Separate Model ROI")
plt.ylabel("Hierarchical Model ROI")
```

---

## When to Use Hierarchical Models

### Use When

- Multiple similar units (regions, brands, products)
- Some units have limited data
- Expect common patterns with local variation
- Want to generalize to new units

### Avoid When

- Units are fundamentally different
- Enough data per unit for stable estimates
- Regional effects are primary interest
- Computational constraints

---

## Computational Considerations

### Increased Complexity

| Model Type | Parameters | Fit Time |
|------------|------------|----------|
| Single Region | ~50 | 30-60 min |
| Hierarchical (5 regions) | ~150 | 1-2 hours |
| Hierarchical (19 regions) | ~500 | 2-4 hours |

### Memory Requirements

More regions = more parameters = more memory.

**Recommendation:** Start with 5-10 regions, scale up gradually.

---

## Common Issues

### 1. Convergence Problems

Hierarchical models are harder to fit. Solutions:

- More warmup samples
- Higher target_accept
- Non-centered parameterization

### 2. Label Switching

MCMC chains may swap region labels. Check manually if needed.

### 3. Weak Identification

If regional variation is low, hyperparameters may be poorly identified.

---

## References

- Gelman, A., & Hill, J. (2006). Data Analysis Using Regression and Multilevel/Hierarchical Models.
- Betancourt, M., & Girolami, M. (2015). Hamiltonian Monte Carlo for Hierarchical Models.
