import arviz as az
import pandas as pd
import numpy as np
import os
import json
import logging
from pathlib import Path
from datetime import datetime
import mlflow
from src.config import (
    PROJECT_ROOT,
    MODELS_DIR,
    REPORTS_DIR,
    MAX_ALLOWED_RHAT,
    MAX_DIVERGENCE_PERCENT,
    MIN_ESS_PER_CHAIN,
    DRIFT_THRESHOLD_PCT,
    LAST_STABLE_RUN_DIR
)

logger = logging.getLogger(__name__)

class ModelHealthAuditor:
    """Standalone engine for Bayesian MMM health auditing and drift detection."""

    def __init__(self, run_id: str, idata_path: Path | None = None):
        self.run_id = run_id
        self.idata_path = idata_path or MODELS_DIR / "idata.nc"
        self.report_path = REPORTS_DIR / f"audit_{run_id}.md"
        self.results = {}

    def run_full_audit(self) -> dict:
        """Execute all health checks and drift detection."""
        if not self.idata_path.exists():
            logger.error(f"Audit failed: {self.idata_path} not found.")
            return {"status": "error", "message": "File not found"}

        logger.info(f"Starting audit for run {self.run_id}...")
        idata = az.from_netcdf(self.idata_path)
        
        self.results["convergence"] = self._check_convergence(idata)
        self.results["stability"] = self._check_stability(idata)
        self.results["drift"] = self._check_drift(idata)
        
        # Calculate overall health score (0-100)
        self.results["health_score"] = self._calculate_health_score()
        
        self._generate_markdown_report()
        self._tag_mlflow()
        
        return self.results

    def _check_convergence(self, idata) -> dict:
        """Check R-hat and ESS metrics."""
        summary = az.summary(idata)
        max_rhat = float(summary["r_hat"].max())
        min_ess = float(summary["ess_bulk"].min())
        
        # Identify bottleneck parameters
        problematic = summary[summary["r_hat"] > MAX_ALLOWED_RHAT]
        problem_params = problematic.sort_values("r_hat", ascending=False).head(5).index.tolist()
        
        is_converged = max_rhat < MAX_ALLOWED_RHAT
        
        return {
            "max_rhat": max_rhat,
            "min_ess_bulk": min_ess,
            "is_converged": is_converged,
            "problematic_parameters": problem_params
        }

    def _check_stability(self, idata) -> dict:
        """Check for divergent transitions."""
        if "sample_stats" in idata and "diverging" in idata.sample_stats:
            div = int(idata.sample_stats.diverging.sum())
            total = int(idata.sample_stats.diverging.size)
            div_pct = div / total
            is_stable = div_pct < MAX_DIVERGENCE_PERCENT
        else:
            div, div_pct, is_stable = 0, 0.0, True

        return {
            "divergences": div,
            "divergence_pct": div_pct,
            "is_stable": is_stable
        }

    def _check_drift(self, idata) -> dict:
        """Compare current posteriors with the last stable model."""
        last_stable_path = LAST_STABLE_RUN_DIR / "idata.nc"
        if not last_stable_path.exists():
            return {"status": "skipped", "message": "No stable model found for comparison"}

        logger.info(f"Comparing against stable model: {last_stable_path}")
        idata_stable = az.from_netcdf(last_stable_path)
        
        # Focus on beta_channel (Global media effectiveness)
        if "beta_channel" in idata.posterior and "beta_channel" in idata_stable.posterior:
            curr_roi = idata.posterior.beta_channel.mean(dim=["chain", "draw"]).values
            prev_roi = idata_stable.posterior.beta_channel.mean(dim=["chain", "draw"]).values
            
            # Prevent division by zero
            prev_roi_safe = np.where(prev_roi == 0, 1e-6, prev_roi)
            drift_pct = (curr_roi - prev_roi) / prev_roi_safe
            max_drift = float(np.abs(drift_pct).max())
            
            # Get channels with highest drift
            channels = idata.posterior.coords["channel"].values
            drift_map = dict(zip(channels, drift_pct.tolist()))
            top_drift = sorted(drift_map.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            
            return {
                "max_drift_pct": max_drift,
                "high_drift_channels": top_drift,
                "is_drift_detected": max_drift > DRIFT_THRESHOLD_PCT
            }
        
        return {"status": "skipped", "message": "Required variables missing in traces"}

    def _calculate_health_score(self) -> int:
        """Calculate a composite 0-100 score."""
        score = 100
        conv = self.results.get("convergence", {})
        stab = self.results.get("stability", {})
        drift = self.results.get("drift", {})
        
        # Penalties
        if conv.get("max_rhat", 1.0) > 1.1: score -= 20
        if conv.get("max_rhat", 1.0) > 1.5: score -= 30
        if stab.get("divergence_pct", 0) > 0.01: score -= 20
        if drift.get("is_drift_detected"): score -= 15
        
        return max(0, score)

    def _generate_markdown_report(self):
        """Generate a professionally formatted audit report."""
        conv = self.results.get("convergence", {})
        stab = self.results.get("stability", {})
        drift = self.results.get("drift", {})
        score = self.results.get("health_score", 0)
        
        status_emoji = "✅" if score > 80 else "⚠️" if score > 50 else "❌"
        
        report = f"""# MMM Model Health Audit — Run {self.run_id}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overall Health Score: {score}/100 {status_emoji}
Status: {"STABLE" if score > 80 else "CAUTION" if score > 50 else "UNSTABLE"}

---

## 🌀 Convergence & Sampling
- **Max R-hat**: {conv.get('max_rhat', 'N/A'):.4f} (Threshold: {MAX_ALLOWED_RHAT})
- **Min ESS**: {conv.get('min_ess_bulk', 'N/A'):.1f}
- **Divergences**: {stab.get('divergences', 0)} ({stab.get('divergence_pct', 0)*100:.2f}%)
- **Status**: {"PASS" if conv.get('is_converged') and stab.get('is_stable') else "FAIL"}

**Problematic Parameters (Top 5 R-hat):**
{chr(10).join([f"- {p}" for p in conv.get('problematic_parameters', [])])}

---

## 🌊 Concept Drift Detection
- **Max ROI Shift**: {drift.get('max_drift_pct', 0)*100:.2f}% (Threshold: {DRIFT_THRESHOLD_PCT*100}%)
- **Drift Status**: {"WARNING" if drift.get('is_drift_detected') else "OK"}

**Top Channel Shifts:**
{chr(10).join([f"- {c}: {d*100:+.2f}%" for c, d in drift.get('high_drift_channels', [])])}

---
*Report generated automatically by MMM Audit Engine.*
"""
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Audit report saved: {self.report_path}")

    def _tag_mlflow(self):
        """Tag the active or specified run in MLflow."""
        try:
            # We must use start_run with run_id because this is a background process
            with mlflow.start_run(run_id=self.run_id):
                score = self.results.get("health_score", 0)
                mlflow.set_tag("model_health_score", score)
                mlflow.set_tag("lifecycle", "stable" if score > 50 else "unstable")
                mlflow.log_metric("max_rhat", self.results["convergence"].get("max_rhat", 0))
                mlflow.log_artifact(str(self.report_path), "audit")
                logger.info(f"Successfully tagged MLflow run {self.run_id}")
        except Exception as e:
            logger.warning(f"Failed to tag MLflow: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--idata", help="Path to idata.nc", default=str(MODELS_DIR / "idata.nc"))
    args = parser.parse_args()
    
    auditor = ModelHealthAuditor(args.run_id, Path(args.idata))
    auditor.run_full_audit()
