import arviz as az
import pandas as pd
import pickle
import os
from pathlib import Path
import json

# Paths
PROJECT_ROOT = Path("d:/Projects/MMM-Figshare-eCommerce")
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
IDATA_PATH = MODELS_DIR / "idata.nc"

def verify_run():
    print(f"Loading {IDATA_PATH}...")
    if not IDATA_PATH.exists():
        print(f"Error: {IDATA_PATH} not found.")
        return

    # Load inference data
    idata = az.from_netcdf(IDATA_PATH)
    
    # 1. Coordinate-Level R-hat Investigation
    print("\n--- Top 20 Parameters by R-hat ---")
    summary = az.summary(idata)
    top_rhat = summary.sort_values("r_hat", ascending=False).head(20)
    print(top_rhat[["mean", "sd", "hdi_3%", "hdi_97%", "r_hat"]])

    # 2. Channel/Territory specific R-hat (Problem Children)
    print("\n--- High R-hat Channels/Territories (R > 1.1) ---")
    problematic = summary[summary["r_hat"] > 1.1]
    if not problematic.empty:
        # Filter for the main variables we care about
        top_problems = problematic[problematic.index.str.contains("beta_channel|adstock_theta|alpha")]
        print(top_problems[["r_hat"]].sort_values("r_hat", ascending=False).head(20))
    else:
        print("All parameters have R-hat < 1.1 (Model converged).")

    # 3. Stability Check: Divergences
    print("\n--- Divergent Transitions ---")
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        divergences = idata.sample_stats.diverging.sum().values
        total_samples = idata.sample_stats.diverging.size
        print(f"Total Divergences: {divergences} ({100 * divergences / total_samples:.2f}%)")
    else:
        print("Divergent transition data NOT found.")

    # 4. Chain Variance Analysis (Stability)
    print("\n--- Chain Consistency (Mean per Chain) ---")
    if "beta_channel" in idata.posterior:
        # beta_channel dims: ('chain', 'draw', 'channel')
        means_per_chain = idata.posterior.beta_channel.mean(dim=["draw", "channel"])
        print("Global Beta Channel Mean per Chain (Pooled):")
        print(means_per_chain.values)
    
    if "beta_channel_territory" in idata.posterior:
        # beta_channel_territory dims: ('chain', 'draw', 'territory', 'channel')
        means_per_chain_t = idata.posterior.beta_channel_territory.mean(dim=["draw", "territory", "channel"])
        print("\nRegional Beta Channel Mean per Chain:")
        print(means_per_chain_t.values)

    # 5. Posterior Predictive Check (if data exists)
    if hasattr(idata, "posterior_predictive") or hasattr(idata, "observed_data"):
        print("\n--- Posterior Predictive Check (PPC) Status ---")
        if hasattr(idata, "posterior_predictive"):
            print("Posterior predictive samples available.")
        if hasattr(idata, "observed_data"):
            print("Observed data available.")

    # 6. Save audit report
    audit_data = {
        "max_rhat": float(summary["r_hat"].max()),
        "divergences": int(divergences),
        "problem_count": len(problematic),
        "top_rhat_params": top_rhat.index.tolist()
    }
    
    with open(REPORTS_DIR / "audit_report.json", "w") as f:
        json.dump(audit_data, f, indent=2)
    print(f"\nAudit report saved to {REPORTS_DIR / 'audit_report.json'}")

if __name__ == "__main__":
    verify_run()
