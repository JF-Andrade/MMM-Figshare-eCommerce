# ROI Computation and Interpretation

## What is ROI?

Return on Investment (ROI) measures the efficiency of marketing spend:

```text
ROI = Revenue Contribution / Spend
```

An ROI of 2.0 means each $1 spent generates $2 in revenue.

---

## Channel Contribution

### Definition

The incremental revenue attributable to a channel:

```text
Contribution = Revenue_with_channel - Revenue_without_channel
```

In MMM, this is computed from the model coefficients and transformed spend.

### PyMC-Marketing Computation

```python
contributions = mmm.compute_channel_contribution_original_scale()
```

Returns xarray with dimensions: chain, draw, date, channel.

---

## Computing ROI

### Step-by-Step

```python
# 1. Get contributions
contributions = mmm.compute_channel_contribution_original_scale()

# 2. Mean across MCMC samples
mean_contrib = contributions.mean(dim=["chain", "draw"])

# 3. Sum across time
total_contrib = mean_contrib.sum(dim="date")

# 4. Compute ROI per channel
roi_data = []
for channel in channels:
    spend = X_train[channel].sum()
    contrib = float(total_contrib.sel(channel=channel).values)
    roi = contrib / spend if spend > 0 else 0
    
    roi_data.append({
        "channel": channel,
        "spend": spend,
        "contribution": contrib,
        "roi": roi,
    })

roi_df = pd.DataFrame(roi_data)
```

---

## Uncertainty Quantification

### Credible Intervals

```python
# Posterior distribution of total contribution
total_contrib_samples = contributions.sum(dim="date")

# 95% credible interval
lower = total_contrib_samples.quantile(0.025, dim=["chain", "draw"])
upper = total_contrib_samples.quantile(0.975, dim=["chain", "draw"])
```

### ROI with Uncertainty

```python
for channel in channels:
    spend = X_train[channel].sum()
    contrib_samples = total_contrib_samples.sel(channel=channel)
    
    roi_samples = contrib_samples / spend
    roi_mean = roi_samples.mean()
    roi_lower = roi_samples.quantile(0.025)
    roi_upper = roi_samples.quantile(0.975)
    
    print(f"{channel}: ROI = {roi_mean:.2f} [{roi_lower:.2f}, {roi_upper:.2f}]")
```

---

## Average vs Marginal ROI

### Average ROI

```text
Average ROI = Total Contribution / Total Spend
```

Measures overall efficiency of spend.

### Marginal ROI

```text
Marginal ROI = dContribution / dSpend
```

Measures the return from the next dollar spent.

### Key Insight

Due to saturation, **marginal ROI decreases** as spend increases.

**Derivation for Exponential Saturation:**

For the saturation function f(x) = β(1 - exp(-λx)):

```text
Marginal ROI = df/dx = β × λ × exp(-λx)
```

**Properties:**

| Spend Level x | exp(-λx) | Marginal ROI |
|---------------|----------|---------------|
| x = 0 | 1.0 | βλ (maximum) |
| x = 1/λ | 0.37 | 0.37βλ |
| x = 2/λ | 0.14 | 0.14βλ |
| x → ∞ | 0 | 0 |

This proves:

- Marginal ROI is highest at zero spend
- Marginal ROI < Average ROI at high spend
- Marginal ROI determines optimal allocation

---

## Response Curves

### Definition

Plot showing expected response at different spend levels.

### Computing Response Curves

PyMC-Marketing provides response curve visualization:

```python
import numpy as np
import matplotlib.pyplot as plt

# Get posterior means for adstock and saturation parameters
alpha_mean = mmm.idata.posterior["adstock_alpha"].mean(dim=["chain", "draw"])
lam_mean = mmm.idata.posterior["saturation_lam"].mean(dim=["chain", "draw"])
beta_mean = mmm.idata.posterior["channel_coefficients"].mean(dim=["chain", "draw"])

# Generate spend range
channel = "META_FACEBOOK_SPEND"
max_spend = X_train[channel].max() * 1.5
spend_range = np.linspace(0, max_spend, 100)

# Apply saturation (simplified for visualization)
lam = float(lam_mean.sel(channel=channel))
beta = float(beta_mean.sel(channel=channel))
response = beta * (1 - np.exp(-lam * spend_range))

# Plot
plt.plot(spend_range, response)
plt.xlabel("Weekly Spend ($)")
plt.ylabel("Expected Incremental Revenue ($)")
plt.title(f"Response Curve: {channel}")
plt.axvline(X_train[channel].mean(), color="red", linestyle="--", label="Current Avg")
plt.legend()
```

---

## Interpreting ROI

### ROI Benchmarks

| ROI | Interpretation |
|-----|----------------|
| < 1.0 | Unprofitable |
| 1.0 - 1.5 | Break-even to marginal |
| 1.5 - 2.5 | Good efficiency |
| 2.5 - 4.0 | Strong efficiency |
| > 4.0 | Exceptional (verify data) |

### Cautions

1. **Very high ROI:** May indicate underinvestment or data issues
2. **Negative ROI:** Check model fit and data quality
3. **Wide uncertainty:** Need more data or better model

---

## Decomposition Analysis

### Revenue Decomposition

```
Total Revenue = Base + Σ(Channel Contributions) + Controls
```

### Computing Shares

```python
import numpy as np

# Compute base contribution
intercept_mean = float(mmm.idata.posterior["intercept"].mean())
base_contribution = intercept_mean * len(y_train)

# Total observed revenue
total_revenue = y_train.sum()

# Channel contributions (already computed)
channel_total = float(total_contrib.sum())

# Remainder goes to controls/unexplained
control_contribution = total_revenue - base_contribution - channel_total

print(f"Base sales: {base_contribution/total_revenue:.1%}")
print(f"Channel contribution: {channel_total/total_revenue:.1%}")
print(f"Controls/Other: {control_contribution/total_revenue:.1%}")
```

---

## Visualization

### Channel Comparison

```python
roi_df_sorted = roi_df.sort_values("roi", ascending=True)

plt.barh(roi_df_sorted["channel"], roi_df_sorted["roi"])
plt.axvline(x=1.0, color="red", linestyle="--", label="Break-even")
plt.xlabel("ROI")
plt.title("Channel ROI Comparison")
```

### Spend vs Contribution

```python
plt.scatter(roi_df["spend"], roi_df["contribution"])
for _, row in roi_df.iterrows():
    plt.annotate(row["channel"], (row["spend"], row["contribution"]))
plt.xlabel("Spend")
plt.ylabel("Contribution")
```

---

## Business Recommendations

### From ROI Analysis

1. **Increase spend** on high-ROI, low-saturation channels
2. **Maintain or reduce** spend on saturated channels
3. **Test** channels with wide uncertainty
4. **Investigate** unexpectedly low/high ROI

### Example Recommendations

| Channel | ROI | Saturation | Recommendation |
|---------|-----|------------|----------------|
| META_FACEBOOK | 2.1 | High | Maintain |
| GOOGLE_PMAX | 3.2 | Low | Increase 20% |
| TIKTOK | 0.8 | Unknown | Reduce or test |

---

## Common Pitfalls

### 1. Ignoring Uncertainty

Point estimates alone are misleading. Always report credible intervals.

### 2. Confusing Average and Marginal

Average ROI doesn't indicate where to allocate next dollar.

### 3. Attribution to Wrong Period

Without adstock, short-term channels appear more effective.

### 4. Ignoring Saturation

Linear models overestimate returns at high spend levels.

---

## References

- Chan, D., & Perry, M. (2017). Challenges and Opportunities in Media Mix Modeling.
- Jin, Y., et al. (2017). Bayesian Methods for Media Mix Modeling.
