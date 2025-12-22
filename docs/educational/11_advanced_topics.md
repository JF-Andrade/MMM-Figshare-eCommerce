# Advanced Topics

## Overview

This section covers advanced extensions and considerations for Marketing Mix Modeling beyond the standard approach.

---

## 0. Frequentist Baseline Comparison

### Why Use a Baseline?

Before investing in complex Bayesian models, it is valuable to establish frequentist baselines for comparison:

1. **Sanity check**: Does the Bayesian model outperform simpler approaches?
2. **Speed**: Baselines train in seconds vs minutes/hours for MCMC
3. **Interpretability**: Linear coefficients are directly interpretable

### Ridge Regression Baseline

This project implements a Ridge Regression baseline (`scripts/mmm_baseline.py`) with:

```python
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])),
])
```

### Manual Adstock and Saturation

Unlike PyMC-Marketing, Ridge requires manual transformations:

```python
# Apply adstock before fitting
x_adstock = apply_adstock(x, decay=0.5)

# Apply saturation
x_saturated = x_adstock / (0.5 + x_adstock)
```

### Comparison Framework

| Metric | Ridge Baseline | Bayesian MMM | Winner |
|--------|----------------|--------------|--------|
| R² Test | 0.65 | 0.78 | Bayesian |
| MAPE | 12.3% | 8.5% | Bayesian |
| Training Time | 1.2s | 312s | Ridge |
| Uncertainty | No | Yes | Bayesian |

The tradeoff is clear: Bayesian provides better accuracy and uncertainty, but at significant computational cost.

---

## 1. Channel Interactions

### The Problem

Standard MMM assumes channels act independently. In reality:

- TV may boost search intent
- Social creates awareness that converts via email
- Retargeting builds on display exposure

### Modeling Interactions

```
Response = β₁×Channel₁ + β₂×Channel₂ + β₁₂×Channel₁×Channel₂
```

### Implementation

```python
# Add interaction terms
X["google_x_meta"] = X["google_spend"] * X["meta_spend"]

# Include in model
mmm = MMM(
    channel_columns=channels + ["google_x_meta"],
    ...
)
```

### Challenges

- Many possible interactions (n² for n channels)
- Multicollinearity
- Interpretability

---

## 2. Time-Varying Parameters

### The Problem

Channel effectiveness may change over time:

- Creative fatigue
- Market saturation
- Competitive actions
- Seasonal patterns

### Approaches

**1. Regime-Switching Models**

```
β(t) = β₁ if t < breakpoint else β₂
```

**2. Dynamic Linear Models (DLM)**

```
β(t) ~ Normal(β(t-1), σ²)
```

**3. Gaussian Processes**

```
β(t) ~ GP(μ(t), K(t, t'))
```

### Implementation Note

PyMC-Marketing does not natively support time-varying parameters. Custom PyMC model required.

---

## 3. Causal Inference Considerations

### MMM vs True Causality

MMM measures association, not causation. Confounders include:

- **Targeting**: Ads shown when conversion likely anyway
- **Seasonality**: Spend and sales both increase at holidays
- **Omitted variables**: Competitor actions, macroeconomic factors

### Strengthening Causal Claims

**1. Control Variables**
Include all relevant non-marketing factors.

**2. Instrumental Variables**
Find exogenous variation in spend (budget changes, platform outages).

**3. Geo Experiments**
Holdout regions to validate MMM estimates.

**4. Difference-in-Differences**
Compare treated vs control periods/regions.

---

## 4. Lift Studies Integration

### Randomized Experiments

- **Conversion lift**: Facebook/Google experiments
- **Geo experiments**: Regional holdouts
- **Incrementality tests**: PSA-controlled tests

### Combining with MMM

**Prior Calibration:**
Use lift results to inform MMM priors:

```
β_meta ~ Normal(lift_estimate, lift_se)
```

**Posterior Calibration:**
Adjust MMM results to match lift:

```
β_adjusted = α × β_mmm + (1-α) × β_lift
```

---

## 5. Long-Term Brand Effects

### The Challenge

MMM captures short-to-medium term effects. Brand building occurs over years.

### Approaches

**1. Extended Adstock**
Use very long l_max (52+ weeks).

**2. Brand Metrics**
Include brand tracking data as mediator:

```
Awareness → Consideration → Purchase Intent → Sales
```

**3. Long-Term Response Models**
Separate short-term (performance) and long-term (brand) effects.

---

## 6. LLM-Assisted Prior Elicitation

### The Opportunity

Large language models can help:

- Synthesize domain knowledge
- Suggest reasonable prior ranges
- Explain prior choices

### Example Workflow

```
Prompt: "For a eCommerce brand spending on Google Shopping,
         what is a reasonable range for adstock decay rate?"

Response: "Based on search marketing literature, 
          Google Shopping typically has decay rates 
          of 0.3-0.6, reflecting short consideration cycles."
```

### Caution

LLM suggestions should be validated by domain experts.

---

## 7. Incrementality and Attribution

### Incrementality Framework

```
Incremental Revenue = Revenue_with_ads - Revenue_without_ads
```

This is the causal question MMM approximates.

### Attribution Models Comparison

| Model | Granularity | Causality | Coverage |
|-------|-------------|-----------|----------|
| Last-click | User-level | Low | Digital only |
| MTA | User-level | Medium | Digital only |
| MMM | Aggregate | Medium | All channels |
| Experiments | Group-level | High | Tested only |

### Unified Approach

Triangulate across methods:

1. MMM for strategic allocation
2. Attribution for tactical optimization
3. Experiments for validation

---

## 8. Real-Time Bayesian MMM

### Challenge

Traditional MMM is slow (batch). Real-time decisions need faster updates.

### Online Learning

```
Prior(t) = Posterior(t-1)
Posterior(t) = update(Prior(t), Data(t))
```

### Approximate Methods

- Variational inference
- Online gradient descent
- Kalman filtering

---

## 9. External Data Integration

### Potential Sources

| Data Type | Source | Use |
|-----------|--------|-----|
| Macroeconomic | FRED, World Bank | Controls |
| Weather | NOAA | Controls |
| Competitor spend | Adbeat, Pathmatics | Competitive context |
| Social sentiment | Twitter API | Awareness proxy |
| Google Trends | Google | Search interest |

### Implementation

```python
# Merge external data
df = df.merge(econ_data, on="week")
df = df.merge(weather_data, on=["week", "region"])

# Include as controls
mmm = MMM(
    control_columns=["trend", "is_holiday", "gdp_growth", "temperature"],
    ...
)
```

---

## 10. Multi-Objective Optimization

### Beyond Revenue

Optimize for multiple objectives:

- Revenue maximization
- Customer acquisition
- Brand awareness
- Customer lifetime value

### Pareto Optimization

Find allocations on the efficient frontier:

```
Maximize: (Revenue, New_Customers)
Subject to: Budget constraint
```

---

## 11. Model Ensembling

### Combining Models

Average predictions from multiple specifications:

```
ŷ = w₁×ŷ_model1 + w₂×ŷ_model2 + ...
```

Weight by validation performance or use Bayesian model averaging.

---

## 12. Productionization

### Deployment Considerations

| Aspect | Recommendation |
|--------|----------------|
| Frequency | Monthly refresh |
| Automation | Scheduled pipelines |
| Monitoring | Track forecast accuracy |
| Versioning | Track model artifacts |
| Governance | Document assumptions |

### Architecture

```
Data Pipeline → Feature Engineering → Model Training → 
Results API → Dashboard → Decision Support
```

---

## Future Directions

1. **Foundation models for marketing**: Pre-trained MMM priors
2. **Causal representation learning**: End-to-end causal inference
3. **Automated prior elicitation**: LLM-guided prior specification
4. **Continuous experimentation**: Always-on validation
5. **Privacy-preserving MMM**: Aggregate data only

---

## References

- Chan, D., & Perry, M. (2017). Challenges and Opportunities in Media Mix Modeling.
- Facure, M. (2022). Causal Inference for the Brave and True.
- Pearl, J. (2009). Causality: Models, Reasoning, and Inference.
