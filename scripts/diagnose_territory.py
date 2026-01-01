"""Quick diagnostic script to check territory mismatch."""
import sys
sys.path.insert(0, '.')

import pandas as pd
import arviz as az
import numpy as np
from pathlib import Path

# Load prepared training data
df_train = pd.read_parquet('data/inspection/hierarchical_train.parquet')
print(f"Train data shape: {df_train.shape}")
print(f"Territories in train data: {df_train['geo'].unique().tolist()}")
print(f"Number of territories: {df_train['geo'].nunique()}")

# Load latest idata
from scripts.regenerate_deliverables import get_latest_hierarchical_run
from mlflow import MlflowClient

client = MlflowClient('file:./mlruns')
run_id = get_latest_hierarchical_run()
print(f"\nLoading idata from run: {run_id}")

local_path = client.download_artifacts(run_id, 'mmm_hierarchical_trace.nc')
idata = az.from_netcdf(local_path)

# Check territory dimension in idata
print(f"\nidata territory coord: {list(idata.posterior.coords['territory'].values)}")
print(f"idata n_territories: {len(idata.posterior.coords['territory'])}")

# Check shapes of key parameters
print(f"\nalpha_territory shape: {idata.posterior['alpha_territory'].shape}")
print(f"beta_channel_territory shape: {idata.posterior['beta_channel_territory'].shape}")
print(f"L_territory shape: {idata.posterior['L_territory'].shape}")

# Analyze beta values
print("\n=== BETA VALUES ===")
beta_channel = idata.posterior["beta_channel"].mean(dim=["chain", "draw"]).values
print(f"beta_channel (mean): {beta_channel}")
print(f"beta_channel sum: {beta_channel.sum():.4f}")

beta_territory = idata.posterior["beta_channel_territory"].mean(dim=["chain", "draw"]).values
print(f"\nbeta_territory (mean per territory): {beta_territory.mean(axis=1)}")
print(f"beta_territory total sum: {beta_territory.sum():.4f}")

# Check y_log scale
print("\n=== TARGET SCALE ===")
y_log = df_train['y_log'].values
print(f"y_log: mean={y_log.mean():.2f}, std={y_log.std():.2f}, min={y_log.min():.2f}, max={y_log.max():.2f}")

# Original target
if 'ALL_PURCHASES_ORIGINAL_PRICE' in df_train.columns:
    y_orig = df_train['ALL_PURCHASES_ORIGINAL_PRICE'].values
    print(f"y_original: mean={y_orig.mean():,.0f}, sum={y_orig.sum():,.0f}")

# Spend scale
print("\n=== SPEND SCALE ===")
spend_cols = [c for c in df_train.columns if c.endswith('_norm')]
for col in spend_cols[:3]:
    print(f"{col}: mean={df_train[col].mean():.4f}, max={df_train[col].max():.4f}")

