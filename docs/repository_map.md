# Repository Architecture Map

This map visualizes the structural relationship between the different modules in the **MMM-Figshare-eCommerce** repository, as indexed by GitNexus.

## System Overview

```mermaid
flowchart TD
    %% Entry Points
    subgraph Execution ["1. Entry Points (Scripts)"]
        direction TB
        RP["run_pipeline.py<br/>(Orchestrator)"]
        MH["mmm_hierarchical.py<br/>(HB-MMM Logic)"]
        GD["generate_deliverables.py<br/>(Reporting)"]
    end

    %% Core Logic
    subgraph Core ["2. Core Logic (Source)"]
        direction TB
        PIPE["pipeline.py<br/>(Finite State Machine)"]
        HB_MODEL["models/hierarchical_bayesian.py<br/>(Probabilistic Model)"]
        PRE["preprocessing.py<br/>(Data Cleaning)"]
        TRANS["transformations.py<br/>(Adstock/Hill)"]
        CONFIG["config.py<br/>(Globals & Git)"]
        INSIGHTS["insights.py<br/>(Post-Modeling)"]
    end

    %% Analytical UI
    subgraph UI ["3. Analytical UI (App)"]
        direction TB
        ML_LOAD["mlflow_loader.py<br/>(Artifact Retrieval)"]
        COMP["components/charts.py<br/>(Visualizations)"]
        EXP["components/export.py<br/>(Excel Reports)"]
        PAGES["pages/<br/>(Streamlit Pages)"]
    end

    %% Infrastructure
    subgraph Infra ["4. External Services"]
        MLFLOW[("MLflow Tracking")]
    end

    %% Relationships
    RP --> PIPE
    MH --> HB_MODEL
    MH --> PRE
    MH --> TRANS
    PIPE --> CONFIG
    
    GD --> ML_LOAD
    ML_LOAD --> MLFLOW
    
    PAGES --> COMP
    PAGES --> EXP
    COMP --> INSIGHTS
    COMP --> CONFIG

    %% Styling
    classDef scripts fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef core fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef ui fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    classDef infra fill:#f0f0f0,stroke:#666,stroke-width:2px;

    class RP,MH,GD scripts;
    class PIPE,HB_MODEL,PRE,TRANS,CONFIG,INSIGHTS core;
    class ML_LOAD,COMP,EXP,PAGES ui;
    class MLFLOW infra;
```

## Detailed Execution Flows

### 1. Modeling Pipeline (`run_hierarchical`)
The primary flow for training the model:
1. `prepare_hierarchical_data` (scripts/mmm_hierarchical.py)
2. `prepare_weekly_data` (src/preprocessing.py)
3. `log_transform` (src/transformations.py)
4. Model Fitting (Hierarchical Bayesian Model)

### 2. Insight Generation
The flow for extracting value from trained models:
1. `load_artifacts_from_run` (scripts/generate_deliverables.py)
2. `compute_ridge_coefficients` (src/insights.py)
3. `roi_bar_chart` (app/components/charts.py)

---

> [!TIP]
> Use `gitnexus_context({name: "symbol_name"})` for a 360-degree view of any of these components, including all incoming and outgoing references.
