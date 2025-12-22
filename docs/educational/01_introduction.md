# Introduction to Marketing Mix Modeling

## What is Marketing Mix Modeling?

Marketing Mix Modeling (MMM) is a statistical technique used to quantify the impact of marketing activities on sales or other business outcomes. It decomposes observed sales into contributions from:

- **Base sales**: Sales that would occur without any marketing
- **Marketing activities**: Incremental sales driven by advertising
- **External factors**: Seasonality, holidays, economic conditions

## Historical Context

MMM emerged in the 1960s-1970s when consumer packaged goods companies needed to measure TV advertising effectiveness. The approach was pioneered by econometricians who adapted regression models to marketing data.

### Evolution Timeline

| Era | Approach | Characteristics |
|-----|----------|-----------------|
| 1970s | Classical Econometrics | Linear regression, aggregate data |
| 1990s | Advanced Statistics | Time series, lagged effects |
| 2010s | Digital Integration | Multi-channel, granular data |
| 2020s | Bayesian MMM | Probabilistic, uncertainty quantification |

## MMM vs Attribution Models

### Multi-Touch Attribution (MTA)

| Aspect | MMM | MTA |
|--------|-----|-----|
| Data Source | Aggregate (weekly/monthly) | User-level (cookies/IDs) |
| Channels | All (including offline) | Digital only |
| Privacy | Privacy-safe | Requires tracking |
| Time Horizon | Long-term effects | Short-term touchpoints |
| Methodology | Regression | Rules or ML |

### When to Use MMM

- Measuring offline media (TV, radio, print)
- Long-term brand effects
- Privacy-constrained environments
- Strategic budget allocation

### When to Use MTA

- Real-time optimization
- User journey analysis
- Digital-only campaigns
- Tactical decisions

## Core Concepts

### 1. Adstock (Carryover Effect)

Advertising effects persist beyond the exposure date. A TV ad seen today influences purchases for days or weeks afterward.

```text
Adstock(t) = Spend(t) + λ × Adstock(t-1)
```

Where λ is the decay rate (0 to 1).

### 2. Saturation (Diminishing Returns)

Each additional dollar spent yields less incremental return. The first million in spend is more effective than the tenth million.

```text
Response = L / (1 + exp(-k × (Spend - x₀)))
```

### 3. Base Sales

The baseline level of sales that occurs regardless of marketing activity. Driven by brand equity, distribution, and organic demand.

## The MMM Equation

```text
Sales(t) = Base + Σ f(Adstock(Channel_i)) + Controls + ε
```

Where:

- `f()` is the saturation transformation
- `Adstock()` captures carryover
- `Controls` includes trend, seasonality, promotions
- `ε` is random noise

## Bayesian vs Frequentist MMM

### Frequentist Approach

- Point estimates only
- Confidence intervals via bootstrap
- Regularization (Ridge/LASSO) optional
- Faster computation

### Bayesian Approach

- Full posterior distributions
- Credible intervals with direct interpretation
- Prior information incorporated
- Uncertainty quantification built-in
- Better for small samples

## Industry Applications

### Consumer Packaged Goods

- TV and in-store promotion effectiveness
- Seasonal planning

### Retail / eCommerce

- Digital channel optimization
- Promotional calendar planning

### Financial Services

- Lead generation ROI
- Brand vs performance trade-offs

### Automotive

- Long consideration cycles
- Dealer vs brand advertising

## Key Questions MMM Answers

1. What is the ROI of each marketing channel?
2. How should I allocate my budget optimally?
3. What are the diminishing returns thresholds?
4. How long do advertising effects last?
5. What would happen if I cut spend by 20%?

## Limitations

1. **Correlation vs Causation**: MMM measures association, not pure causality
2. **Data Requirements**: Needs sufficient variation in spend
3. **Granularity**: Weekly/monthly aggregation loses detail
4. **Omitted Variables**: Unmeasured factors can bias results
5. **Changing Dynamics**: Historical relationships may not hold

## References

1. Jin, Y., Wang, Y., Sun, Y., Chan, D., & Koehler, J. (2017). *Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects*. Google Research.
2. Chan, D., & Perry, M. (2017). *Challenges and Opportunities in Media Mix Modeling*. Google Research.
3. PyMC-Marketing Documentation: <https://www.pymc-marketing.io/>

---

## Connection to This Project

The dataset in this project contains **9 marketing channels** that map to the theory above:

| Platform | Channels in Dataset | Expected Behavior |
|----------|---------------------|-------------------|
| Google | GOOGLE_PAID_SEARCH, GOOGLE_SHOPPING, GOOGLE_PMAX, GOOGLE_DISPLAY, GOOGLE_VIDEO | High-intent (search/shopping), brand (video) |
| Meta | META_FACEBOOK, META_INSTAGRAM, META_OTHER | Mid-funnel awareness + retargeting |
| TikTok | TIKTOK | Upper-funnel, younger demographics |

> **Note**: TikTok data has significant missing values in the dataset. See [01_eda.ipynb](file:///d:/Projects/MMM-Figshare-eCommerce/notebooks/01_eda.ipynb) for details.
