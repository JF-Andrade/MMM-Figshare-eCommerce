# Deep Code Audit: Mathematical & Statistical Findings (Phase 2)

## 1. The Log-to-Linear Scaling Fallacy (Critical Flaw)
**Location:** [src/deliverables.py](file:///d:/Projects/MMM-Figshare-eCommerce/src/deliverables.py) ([_compute_contributions](file:///d:/Projects/MMM-Figshare-eCommerce/src/deliverables.py#102-135), [_compute_regional_metrics](file:///d:/Projects/MMM-Figshare-eCommerce/src/deliverables.py#137-190)) and [src/insights.py](file:///d:/Projects/MMM-Figshare-eCommerce/src/insights.py) ([compute_scaling_factor](file:///d:/Projects/MMM-Figshare-eCommerce/src/insights.py#846-852))

**Issue Description:** 
The target variable is log-transformed during preprocessing (`y_log = log(y)`). Consequently, the PyMC hierarchical Bayesian model is additive in log-space:
$$ \log(\hat{y}) = \text{base} + \text{channel}_1 + \dots + \text{channel}_n $$

When extracting the monetary contribution of each channel for the Streamlit dashboard, the codebase applies a naive linear scaling factor:
```python
scale_factor = total_revenue / (mean_log_revenue * n_obs_train + 1e-8)
contrib_df["contribution"] = contrib_df["contribution_log"] * scale_factor
```
**Mathematical Violation:**
Because $\exp(A + B) = \exp(A) \cdot \exp(B)$, an additive model in log-space is inherently **multiplicative** in linear space. Furthermore, due to Jensen's Inequality ($E[\exp(X)] \geq \exp(E[X])$), applying a simple scalar multiplier to log-space contributions fundamentally distorts the true dollar-value contribution of each channel. The code attempts to force a linear additive interpretation onto a non-linear multiplicative model.

**Impact:** The reported [contribution](file:///d:/Projects/MMM-Figshare-eCommerce/src/deliverables.py#102-135) and [roi](file:///d:/Projects/MMM-Figshare-eCommerce/src/insights.py#643-672) values in the dashboard are mathematically incorrect, particularly for channels with high variance or non-linear saturation profiles.

## 2. Compounding Error in Marginal ROAS and Optimization
**Location:** [src/insights.py](file:///d:/Projects/MMM-Figshare-eCommerce/src/insights.py) ([optimize_hierarchical_budget](file:///d:/Projects/MMM-Figshare-eCommerce/src/insights.py#214-458), [compute_marginal_roas](file:///d:/Projects/MMM-Figshare-eCommerce/src/insights.py#910-1007))

**Issue Description:** 
To optimize the budget, the code derives a pseudo-linear `scale` parameter from the flawed [contribution](file:///d:/Projects/MMM-Figshare-eCommerce/src/deliverables.py#102-135) metric:
```python
scale = contribution / (n_obs * sat_current)
```
It then uses this `scale` inside the SLSQP optimizer's objective function to compute the projected contribution at varying spend levels. Similarly, the marginal ROAS calculation derives $\beta$ computationally:
```python
beta = base_contribution / s_current
mroas = beta * dS_dx / max_spend
```

**Mathematical Violation:** 
The optimization engine relies on a mathematical heuristic (linearizing the saturated curve) that is strictly parameterized by the erroneous [contribution](file:///d:/Projects/MMM-Figshare-eCommerce/src/deliverables.py#102-135) calculated from the Log-to-Linear fallacy. Optimization on top of flawed baseline metrics means the recommended budget allocations will mathematically drift from the true optimal state derived from the Bayesian posterior.

**Impact:** The Streamlit app's budget optimization and marginal ROAS curves will recommend suboptimal allocations because they are not optimizing the true posterior expectation.

## 3. Disconnect in Saturation Scaling (Warning)
**Location:** [src/preprocessing.py](file:///d:/Projects/MMM-Figshare-eCommerce/src/preprocessing.py) and [src/transformations.py](file:///d:/Projects/MMM-Figshare-eCommerce/src/transformations.py)

**Issue Description:**
Spend variables are normalized to $[0, 1]$ using a `StandardScaler`-like approach where `max_spend` is computed strictly from the training set per currency/territory. The PyMC model learns the Hill saturation half-saturation point ($L$) in this normalized space. 
While [deliverables.py](file:///d:/Projects/MMM-Figshare-eCommerce/src/deliverables.py) correctly attempts to unscale $L$ by passing `max_spend` into the insights algorithms, if new data in a production pipeline exceeds the original training `max_spend`, the normalization will break bounds and force the Hill function to extrapolate past $X_{norm} = 1.0$, where it loses calibration.

## Summary of Resolution Required
To repair the mathematical integrity of the model's outputs:
1. **True Posterior Predictive Difference**: Channel contributions must be calculated by taking the exact difference in the expected value of the posterior predictive distribution in linear space (e.g., $E[\exp(\text{Model with all channels})] - E[\exp(\text{Model excluding channel C})]$), integrating out the log-normal/Student-T noise analytically or empirically.
2. **Direct MROAS Differentiation**: Marginal ROAS should be computed through exact differentiation of the posterior predictive expression in linear space, not derived retroactively from scaled scalars.
