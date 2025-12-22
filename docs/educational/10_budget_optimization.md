# Budget Optimization

## The Optimization Problem

Given a fixed total budget, how should we allocate spend across channels to maximize total revenue?

---

## Mathematical Formulation

### Objective

```
Maximize: Σ Response_i(Spend_i)
Subject to: Σ Spend_i = Budget
            Spend_i >= 0 for all i
```

Where Response_i is the channel-specific response curve.

---

## Response Curves

### From MMM to Response

Each channel has a response function:

```
Response(Spend) = β × Saturation(Adstock(Spend))
```

### Key Properties

1. **Monotonically increasing**: More spend = more response
2. **Concave**: Diminishing returns
3. **Differentiable**: Enables gradient-based optimization

---

## Optimal Allocation Principle

### Equal Marginal Returns

At the optimum, marginal ROI is equal across all channels:

```
dResponse_1/dSpend_1 = dResponse_2/dSpend_2 = ... = λ
```

Where λ is the Lagrange multiplier (shadow price of budget).

### Intuition

If marginal ROI differs between channels, we can improve by:

- Reducing spend on low marginal ROI channel
- Increasing spend on high marginal ROI channel

---

## Computing Optimal Allocation

### Complete Working Example

This implementation uses posterior means from the fitted MMM:

```python
import numpy as np
from scipy.optimize import minimize

def compute_response(spend, lam, beta):
    """
    Compute expected response for a given spend level.
    Uses exponential saturation: response = beta * (1 - exp(-lam * spend))
    """
    return beta * (1 - np.exp(-lam * spend))

def optimize_budget(mmm, channels, total_budget, current_spends):
    """
    Find optimal allocation that maximizes total response.
    
    Parameters:
    -----------
    mmm : fitted MMM model
    channels : list of channel names
    total_budget : total budget constraint
    current_spends : dict of current spend per channel (for initialization)
    
    Returns:
    --------
    dict : optimal spend per channel
    """
    # Extract posterior means for saturation and coefficients
    lam_mean = mmm.idata.posterior["saturation_lam"].mean(dim=["chain", "draw"])
    beta_mean = mmm.idata.posterior["channel_coefficients"].mean(dim=["chain", "draw"])
    
    # Get parameter values per channel
    lams = [float(lam_mean.sel(channel=ch)) for ch in channels]
    betas = [float(beta_mean.sel(channel=ch)) for ch in channels]
    
    def negative_total_response(spends):
        """Objective: minimize negative response (= maximize response)"""
        total = 0
        for spend, lam, beta in zip(spends, lams, betas):
            total += compute_response(spend, lam, beta)
        return -total
    
    def budget_constraint(spends):
        """Equality constraint: sum of spends = budget"""
        return sum(spends) - total_budget
    
    # Initial guess: proportional to current allocation
    x0 = [current_spends.get(ch, total_budget / len(channels)) for ch in channels]
    
    # Bounds: non-negative spend, optional upper bounds
    bounds = [(0, None) for _ in channels]
    
    result = minimize(
        negative_total_response,
        x0=x0,
        method='SLSQP',
        constraints={'type': 'eq', 'fun': budget_constraint},
        bounds=bounds,
    )
    
    if not result.success:
        print(f"Warning: Optimization did not converge. {result.message}")
    
    return dict(zip(channels, result.x))

# Example usage
optimal = optimize_budget(
    mmm=mmm,
    channels=channel_columns,
    total_budget=X_train[channel_columns].sum().sum(),
    current_spends=X_train[channel_columns].sum().to_dict(),
)

print("Optimal Allocation:")
for ch, spend in optimal.items():
    print(f"  {ch}: ${spend:,.0f}")
```

---

## Implementation with PyMC-Marketing

### Built-in Methods

PyMC-Marketing provides optimization utilities:

```python
# Get response curves
curves = mmm.plot_response_curves(return_data=True)

# Optimize allocation
optimal = mmm.optimize_budget(
    total_budget=budget,
    num_periods=52,
)
```

### Manual Implementation

```python
def compute_channel_response(mmm, channel, spend):
    # Apply adstock
    adstocked = mmm.adstock.apply(spend)
    
    # Apply saturation
    saturated = mmm.saturation.apply(adstocked, channel)
    
    # Get beta from posterior mean
    beta = mmm.idata.posterior["beta_channel"].sel(channel=channel).mean()
    
    return beta * saturated
```

---

## Scenario Analysis

### What-If Analysis

Compare current vs alternative allocations:

```python
scenarios = {
    "current": current_allocation,
    "optimal": optimal_allocation,
    "shift_to_digital": {...},
    "reduce_tiktok": {...},
}

for name, allocation in scenarios.items():
    total = sum(compute_response(mmm, ch, spend) 
                for ch, spend in allocation.items())
    print(f"{name}: {total:.0f}")
```

### Sensitivity Analysis

Test robustness to parameter uncertainty:

```python
# Use posterior samples
for i in range(100):
    params = sample_from_posterior(mmm, i)
    optimal[i] = optimize_with_params(params, budget)

# Check consistency across samples
```

---

## Constraints

### Real-World Limitations

1. **Minimum spend**: Contractual minimums
2. **Maximum spend**: Inventory limits (search volume)
3. **Rate limits**: Can't scale instantly
4. **Channel requirements**: Strategic mandates

### Incorporating Constraints

```python
bounds = [
    (min_google, max_google),
    (min_meta, max_meta),
    (0, max_tiktok),  # Optional channel
]

result = minimize(
    objective,
    x0=x0,
    bounds=bounds,
    constraints=constraints,
)
```

---

## Uncertainty in Optimization

### Point Estimate Problem

Optimizing on posterior means ignores uncertainty.

### Robust Optimization

```python
# Optimize for worst-case in credible interval
# Or: Optimize expected utility

optimal_robust = optimize_under_uncertainty(mmm, budget)
```

### Recommendation

Report optimal allocation with credible intervals:

```
Channel A: $1.2M [1.0M - 1.5M]
Channel B: $0.8M [0.6M - 1.0M]
```

---

## Visualization

### Allocation Comparison

```python
fig, ax = plt.subplots()

x = np.arange(len(channels))
width = 0.35

ax.bar(x - width/2, current_allocation, width, label="Current")
ax.bar(x + width/2, optimal_allocation, width, label="Optimal")

ax.set_xticks(x)
ax.set_xticklabels(channels, rotation=45)
ax.legend()
ax.set_ylabel("Spend ($)")
```

### Response Curve Overlay

```python
for channel in channels:
    spend_range = np.linspace(0, max_spend, 100)
    response = [compute_response(mmm, channel, s) for s in spend_range]
    
    plt.plot(spend_range, response, label=channel)
    plt.axvline(optimal_spend[channel], linestyle="--")

plt.legend()
plt.xlabel("Spend")
plt.ylabel("Response")
```

---

## Business Communication

### Optimal vs Current

| Channel | Current | Optimal | Change |
|---------|---------|---------|--------|
| Google | $1.5M | $1.2M | -20% |
| Meta | $1.0M | $1.4M | +40% |
| TikTok | $0.5M | $0.4M | -20% |
| **Total** | **$3.0M** | **$3.0M** | 0% |

### Expected Impact

- Incremental revenue: +$X
- ROI improvement: +Y%

---

## Limitations

1. **Response curves may shift**: Future may differ from past
2. **Competitive dynamics**: Ignores competitor actions
3. **Creative effects**: Assumes constant creative quality
4. **Synergies**: May miss channel interactions

---

## References

- Jin, Y., et al. (2017). Bayesian Methods for Media Mix Modeling.
- Chan, D., & Perry, M. (2017). Challenges and Opportunities in Media Mix Modeling.
