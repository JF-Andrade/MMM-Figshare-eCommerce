# PyMC-Marketing Implementation

## Overview

PyMC-Marketing is a Python library for Bayesian marketing analytics built on top of PyMC. It provides ready-to-use implementations of Marketing Mix Models with sensible defaults.

---

## Installation

```bash
pip install pymc-marketing
```

### Version Requirements

| Package | Minimum Version | Notes |
|---------|-----------------|-------|
| pymc-marketing | 0.8.0 | Hierarchical support |
| pymc | 5.0 | Core Bayesian engine |
| arviz | 0.15 | Diagnostics and visualization |
| numpyro | 0.12 | GPU acceleration (optional) |
| jax | 0.4 | Required for numpyro |

> **Note**: For Google Colab, install JAX with GPU support first:
>
> ```python
> !pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
> !pip install numpyro pymc-marketing
> ```

---

## Core Components

### MMM Class

The main class for Marketing Mix Models.

```python
from pymc_marketing.mmm import MMM

mmm = MMM(
    date_column="week",
    channel_columns=["google_spend", "meta_spend", "tiktok_spend"],
    control_columns=["trend", "is_holiday"],
    adstock=GeometricAdstock(l_max=8),
    saturation=LogisticSaturation(),
    yearly_seasonality=2,
)
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `date_column` | Name of date column | Required |
| `channel_columns` | List of spend columns | Required |
| `control_columns` | List of control variables | None |
| `adstock` | Adstock transformation | GeometricAdstock |
| `saturation` | Saturation transformation | LogisticSaturation |
| `yearly_seasonality` | Fourier order for seasonality | None |

---

## Adstock Options

### Geometric Adstock

```python
from pymc_marketing.mmm import GeometricAdstock

adstock = GeometricAdstock(l_max=8)
```

Parameters:

- `l_max`: Maximum lag in time units

### Delayed Adstock

```python
from pymc_marketing.mmm import DelayedAdstock

adstock = DelayedAdstock(l_max=8)
```

Adds peak delay parameter.

---

## Saturation Options

### Logistic Saturation

```python
from pymc_marketing.mmm import LogisticSaturation

saturation = LogisticSaturation()
```

### Hill Saturation

```python
from pymc_marketing.mmm import HillSaturation

saturation = HillSaturation()
```

---

## Fitting the Model

### Basic Fit

```python
mmm.fit(
    X=data,
    y=data["revenue"],
)
```

### Production Settings

```python
mmm.fit(
    X=X_train,
    y=y_train,
    chains=4,
    draws=2000,
    tune=1500,
    target_accept=0.95,
    random_seed=42,
    nuts_sampler="numpyro",  # GPU acceleration
)
```

### Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `chains` | Number of MCMC chains | 4 |
| `draws` | Samples per chain | 2000 |
| `tune` | Warmup samples | 1500 |
| `target_accept` | NUTS acceptance rate | 0.95 |
| `nuts_sampler` | Backend sampler | "numpyro" for GPU |

---

## Accessing Results

### Inference Data

```python
idata = mmm.idata

# Posterior samples
idata.posterior

# Sample statistics
idata.sample_stats
```

### Model Summary

```python
import arviz as az

summary = az.summary(mmm.idata)
print(summary)
```

---

## Channel Contributions

### Compute Contributions

```python
contributions = mmm.compute_channel_contribution_original_scale()
```

Returns xarray DataArray with dimensions:

- `chain`
- `draw`
- `date`
- `channel`

### Mean Contribution

```python
mean_contrib = contributions.mean(dim=["chain", "draw"])
total_contrib = mean_contrib.sum(dim="date")

print(total_contrib)
```

---

## ROI Computation

### Manual ROI Calculation

```python
roi_data = []

for channel in channel_columns:
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

## Predictions

### Posterior Predictive

```python
# In-sample predictions
mmm.sample_posterior_predictive(X_train, extend_idata=True)

# Access predictions
y_pred = mmm.idata.posterior_predictive["y"]
y_pred_mean = y_pred.mean(dim=["chain", "draw"])
```

### Out-of-Sample

```python
y_pred_test = mmm.predict(X_test)
```

---

## Visualization

### Built-in Plots

```python
# Posterior predictive check
mmm.plot_posterior_predictive()

# Channel contributions
mmm.plot_channel_contributions()

# Saturation curves
mmm.plot_saturation_curves()
```

### Custom Plots

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(y_train, label="Actual")
ax.plot(y_pred_mean, label="Predicted")
ax.legend()
plt.show()
```

---

## Saving and Loading

### Save Trace (Recommended)

The most reliable method is saving the inference data as NetCDF:

```python
# Save trace and posterior samples
mmm.idata.to_netcdf("mmm_trace.nc")

# Load later
import arviz as az
idata = az.from_netcdf("mmm_trace.nc")
```

> **Warning**: Avoid `pickle.dump(mmm)` - PyMC-Marketing models contain local functions that fail to serialize. The trace file contains all posterior samples needed for analysis.

### Save Results Separately

```python
# Save ROI results
roi_df.to_csv(output_dir / "roi_baseline.csv", index=False)

# Save model configuration for reproducibility
import json
config = {
    "channels": channel_columns,
    "l_max": 8,
    "holdout_weeks": 12,
}
with open(output_dir / "model_config.json", "w") as f:
    json.dump(config, f, indent=2)
```

---

## Complete Example

```python
import pandas as pd
from pathlib import Path
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

# Load data
data = pd.read_parquet("mmm_data.parquet")

# Define channels
channels = ["GOOGLE_SEARCH_SPEND", "META_FACEBOOK_SPEND", "TIKTOK_SPEND"]

# Prepare data
X = data[["week"] + channels + ["trend", "is_holiday"]]
y = data["revenue"].values

# Create model
mmm = MMM(
    date_column="week",
    channel_columns=channels,
    control_columns=["trend", "is_holiday"],
    adstock=GeometricAdstock(l_max=8),
    saturation=LogisticSaturation(),
)

# Fit
mmm.fit(
    X=X,
    y=y,
    chains=4,
    draws=2000,
    tune=1500,
    nuts_sampler="numpyro",
)

# Compute ROI
contributions = mmm.compute_channel_contribution_original_scale()
mean_contrib = contributions.mean(dim=["chain", "draw"]).sum(dim="date")

for ch in channels:
    spend = X[ch].sum()
    contrib = float(mean_contrib.sel(channel=ch))
    print(f"{ch}: ROI = {contrib/spend:.2f}")
```

---

## Reference

- PyMC-Marketing Documentation: <https://www.pymc-marketing.io/>
- PyMC Documentation: <https://www.pymc.io/>
- ArviZ Documentation: <https://python.arviz.org/>
