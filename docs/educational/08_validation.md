# Model Validation

## Why Validation Matters

A model that fits training data well may not generalize to future data. Validation ensures:

1. Model captures true patterns, not noise
2. Predictions are reliable for decision-making
3. Uncertainty estimates are calibrated

---

## Validation Strategies

### 1. Temporal Holdout

Reserve the most recent time periods for testing.

```
|------ Training ------|--- Test ---|
Week 1             Week 200    Week 212
```

**Advantages:**

- Mimics real forecasting scenario
- Respects temporal ordering
- Simple to implement

**This project uses 12-week holdout.**

### 2. Time Series Cross-Validation

Multiple train-test splits:

```
Fold 1: |Train|Test|
Fold 2: |--Train--|Test|
Fold 3: |----Train----|Test|
```

More robust but computationally expensive.

### 3. Walk-Forward Validation

Incrementally expand training window:

```
Week 1-50:   Train 1-40, Test 41-50
Week 1-60:   Train 1-50, Test 51-60
...
```

---

## Implementation

### Train-Test Split

```python
HOLDOUT_WEEKS = 12

X_train = X.iloc[:-HOLDOUT_WEEKS]
X_test = X.iloc[-HOLDOUT_WEEKS:]
y_train = y[:-HOLDOUT_WEEKS]
y_test = y[-HOLDOUT_WEEKS:]

print(f"Train: {len(X_train)} weeks")
print(f"Test: {len(X_test)} weeks")
```

### Fit on Training Only

```python
mmm.fit(X=X_train, y=y_train, ...)
```

### Evaluate on Test

```python
y_pred_test = mmm.predict(X_test).mean(axis=0)
```

---

## Metrics

### R-squared (Coefficient of Determination)

```
R² = 1 - SS_res / SS_tot
   = 1 - Σ(y - ŷ)² / Σ(y - ȳ)²
```

**Interpretation:**

- R² = 1.0: Perfect prediction
- R² = 0.0: Same as predicting mean
- R² < 0.0: Worse than mean

### Mean Absolute Error (MAE)

```
MAE = (1/n) × Σ|y - ŷ|
```

Same units as target variable. Easy to interpret.

### Mean Absolute Percentage Error (MAPE)

```
MAPE = (1/n) × Σ|y - ŷ| / y × 100%
```

Scale-independent. Problematic when y near zero.

### Root Mean Squared Error (RMSE)

```
RMSE = sqrt((1/n) × Σ(y - ŷ)²)
```

Penalizes large errors more than MAE.

---

## Computing Metrics

```python
def compute_metrics(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    
    mae = np.abs(y_true - y_pred).mean()
    mape = np.abs((y_true - y_pred) / y_true).mean() * 100
    rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
    
    return {
        "r2": r2,
        "mae": mae,
        "mape": mape,
        "rmse": rmse,
    }

train_metrics = compute_metrics(y_train, y_pred_train)
test_metrics = compute_metrics(y_test, y_pred_test)
```

---

## Posterior Predictive Checks

### Concept

Compare observed data to data simulated from the posterior. If the model is well-specified, simulated data should resemble observed data.

### Implementation

```python
mmm.sample_posterior_predictive(X_train, extend_idata=True)

# Access replicated data
y_rep = mmm.idata.posterior_predictive["y"]
```

### Visual Check

```python
az.plot_ppc(mmm.idata)
```

**What to look for:**

- Observed data within simulated distribution
- Similar spread and shape
- No systematic deviations

---

## Residual Analysis

### Compute Residuals

```python
residuals = y_train - y_pred_train
```

### Check for Patterns

1. **Residuals vs Predicted:** Should be random scatter
2. **Residuals vs Time:** No trends
3. **Residuals Histogram:** Approximately normal

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Residuals vs Predicted
axes[0].scatter(y_pred_train, residuals)
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Residuals")
axes[0].set_title("Residuals vs Predicted")

# Residuals over Time
axes[1].plot(residuals)
axes[1].axhline(0, color="red", linestyle="--")
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Residuals")
axes[1].set_title("Residuals over Time")

# Residual Distribution
axes[2].hist(residuals, bins=30, edgecolor='black')
axes[2].set_xlabel("Residuals")
axes[2].set_ylabel("Frequency")
axes[2].set_title("Residual Distribution")

plt.tight_layout()
```

---

## Interpretation Guidelines

### Good Model Performance

| Metric | Training | Test |
|--------|----------|------|
| R² | > 0.7 | > 0.5 |
| MAPE | < 15% | < 20% |

### Warning Signs

- **Train R² high, Test R² low:** Overfitting
- **Residual trends:** Missing variables
- **MAPE > 30%:** Poor model fit

---

## Model Comparison

### Using WAIC/LOO

```python
# Compare two models
az.compare({"model1": model1.idata, "model2": model2.idata})
```

Lower WAIC/LOO-CV indicates better predictive performance.

---

## Checklist

- [ ] Temporal holdout applied
- [ ] In-sample metrics computed
- [ ] Out-of-sample metrics computed
- [ ] Posterior predictive check passed
- [ ] No residual patterns
- [ ] Test R² > 0.5

---

## References

- Gelman, A., et al. (2013). Bayesian Data Analysis. Chapter 6: Model Checking.
- Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian Model Evaluation Using LOO-CV and WAIC.
