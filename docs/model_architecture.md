# Hierarchical MMM - Model Architecture

## Dependency Diagram

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': { 'primaryColor': '#4a90d9', 'edgeLabelBackground':'#2d3748'}}}%%
flowchart TB
    subgraph Data["Data Inputs"]
        X_spend["X_spend<br/>(n_obs × n_channels)"]
        X_features["X_features<br/>(n_obs × n_features)"]
        X_season["X_season<br/>(n_obs × n_season)"]
        territory_idx["territory_idx<br/>(n_obs,)"]
    end

    subgraph Adstock["Adstock (Carryover)"]
        alpha_ch["α_channel<br/>Beta(2,2)"]
        alpha_terr["α_territory<br/>= α_ch + raw × σ"]
        adstock_fn["geometric_adstock()"]
        X_spend --> adstock_fn
        alpha_ch --> alpha_terr
        alpha_terr --> adstock_fn
        territory_idx --> adstock_fn
    end

    subgraph Saturation["Saturation (Diminishing Returns)"]
        L_ch["L_channel<br/>HalfNormal(0.5)"]
        L_terr["L_territory<br/>= softplus(L_ch + raw × σ)"]
        k_ch["k_channel<br/>Gamma(2,2)"]
        hill_fn["hill_saturation()"]
        adstock_fn --> hill_fn
        L_ch --> L_terr
        L_terr --> hill_fn
        k_ch --> hill_fn
    end

    subgraph Betas["Channel Effects"]
        beta_ch["β_channel<br/>HalfNormal(0.3)"]
        beta_terr["β_territory<br/>= raw × σ"]
        beta_eff["β_effective<br/>= β_ch + β_terr[t]"]
        channel_effect["channel_effect<br/>Σ β × saturated"]
        beta_ch --> beta_eff
        beta_terr --> beta_eff
        beta_eff --> channel_effect
        hill_fn --> channel_effect
    end

    subgraph Features["Feature Effects"]
        horseshoe["Horseshoe Prior<br/>τ, λ, c²"]
        beta_f["β_features<br/>~N(0, τλ)"]
        feat_effect["feature_effect<br/>X · β"]
        horseshoe --> beta_f
        beta_f --> feat_effect
        X_features --> feat_effect
    end

    subgraph Season["Seasonality"]
        gamma["γ_season<br/>~N(0, 0.3)"]
        season_effect["season_effect<br/>X · γ"]
        gamma --> season_effect
        X_season --> season_effect
    end

    subgraph Intercepts["Intercepts"]
        alpha_global["α_global<br/>~N(0, 1)"]
        alpha_t_int["α_territory_int<br/>= α_global + raw × σ"]
        alpha_global --> alpha_t_int
    end

    subgraph Likelihood["Likelihood"]
        mu["μ = α_t + channels + features + season"]
        sigma["σ_obs<br/>HalfNormal(0.5)"]
        nu["ν<br/>Gamma(4, 1)"]
        y_obs["y ~ StudentT(μ, σ, ν)"]

        alpha_t_int --> mu
        channel_effect --> mu
        feat_effect --> mu
        season_effect --> mu
        mu --> y_obs
        sigma --> y_obs
        nu --> y_obs
    end

    territory_idx --> alpha_t_int
    territory_idx --> beta_eff
```

## Model Equation

```
y[i] = α_territory[t(i)]
     + Σ_c (β_channel[c] + β_territory[t,c]) × Hill(Adstock(X[i,c], α[t,c]), L[t,c], k[c])
     + X_features[i] · β_features
     + X_season[i] · γ_season
     + ε

Where:
  i = observation index
  t = territory of observation i
  c = channel index
  Hill(x, L, k) = x^k / (L^k + x^k)
  Adstock(x, α) = x[t] + α × Adstock(x[t-1], α)
```

## Key Priors Summary

| Component  | Parameter | Prior           | Interpretation              |
| ---------- | --------- | --------------- | --------------------------- |
| Adstock    | α         | Beta(2,2)       | Decay ~0.5, carryover weeks |
| Saturation | L         | HalfNormal(0.5) | Half-saturation point       |
| Saturation | k         | Gamma(2,2)      | Curve steepness             |
| Channels   | β         | HalfNormal(0.3) | Revenue impact (positive)   |
| Features   | β         | Horseshoe       | Sparse, regularized         |
| Likelihood | ν         | Gamma(4,1)      | Outlier robustness          |
