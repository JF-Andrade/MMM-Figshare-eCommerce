# EDA Key Findings

## Dataset Overview

| Metric | Value |
|--------|-------|
| Rows | 77,841 |
| Columns | 40 |
| Date Range | 2019-06-01 to 2023-12-31 |
| Regions | 18 countries |
| Channels | 9 marketing channels |

---

## Missing Data Patterns

- **TikTok columns**: 99.5% missing values (channel launched late in dataset period)
- **All other columns**: < 0.5% missing
- **Pattern**: Missing values correlate across TikTok metrics (SPEND, CLICKS, IMPRESSIONS)
- **Recommendation**: Impute TikTok with zeros for pre-launch period

---

## Channel Spend Distribution

| Channel | Total Spend | % of Total |
|---------|-------------|------------|
| META_FACEBOOK | $705.7M | 58.1% |
| META_INSTAGRAM | $246.1M | 20.3% |
| GOOGLE_PMAX | $124.0M | 10.2% |
| GOOGLE_SHOPPING | $66.2M | 5.5% |
| GOOGLE_PAID_SEARCH | $40.2M | 3.3% |
| GOOGLE_VIDEO | $17.9M | 1.5% |
| GOOGLE_DISPLAY | $7.2M | 0.6% |
| TIKTOK | $1.3M | 0.1% |
| META_OTHER | $0.6M | 0.05% |

**Insight**: Meta platforms dominate spend (78.4% combined).

---

## Channel Efficiency Metrics

| Channel | CTR (%) | CPC ($) |
|---------|---------|---------|
| GOOGLE_PAID_SEARCH | 5.28% | $1.76 |
| META_FACEBOOK | 1.70% | $5.31 |
| META_INSTAGRAM | 1.18% | $11.78 |
| GOOGLE_PMAX | 1.03% | $1.95 |
| TIKTOK | 0.86% | $1.01 |
| GOOGLE_SHOPPING | 0.70% | $0.80 |
| GOOGLE_DISPLAY | 0.53% | $3.73 |
| GOOGLE_VIDEO | 0.43% | $16.39 |
| META_OTHER | 0.20% | $7.09 |

**Insight**: Google Paid Search has highest CTR; Google Shopping has lowest CPC.

---

## Revenue Analysis

- **Mean daily revenue**: Variable by region
- **Outliers**: Present in revenue distribution (IQR method identifies ~5% as outliers)
- **Log transformation**: Reduces skewness, recommended for modeling

---

## Temporal Patterns

### Weekly Seasonality

- **Peak days**: Weekend (Saturday, Sunday)
- **Low days**: Midweek (Tuesday, Wednesday)
- **Amplitude**: ~15-20% variation from mean

### Holiday Effects

- Holiday periods show 10-25% revenue uplift across regions
- Effect varies by country/vertical

### Trend

- Upward trend 2019-2021
- Stabilization 2022-2023

---

## Regional Insights

- **Top regions by revenue**: US, UK, AU, DE
- **Highest growth**: APAC markets
- **Spend concentration**: 80% in top 5 regions

---

## Correlation Analysis

### Spend vs Revenue

- **Strongest correlations**: META_FACEBOOK (0.45), GOOGLE_PMAX (0.38)
- **Weakest correlations**: TIKTOK (0.12), META_OTHER (0.08)
- **Lagged effects**: Optimal lag 1-3 days for most channels

### Cross-Channel Correlations

- META platforms highly correlated (0.7+)
- Google channels moderately correlated (0.4-0.6)

---

## Feature Engineering Recommendations

### Transformations

1. **Log transform**: Revenue, Spend columns
2. **Adstock**: Apply to spend with decay 0.3-0.7
3. **Saturation**: Hill function for diminishing returns

### Lag Features

- Create lags 1-7 days for spend variables
- Optimal lag based on cross-correlation analysis

### Calendar Features

- `day_of_week`: Categorical (0-6)
- `month`: Categorical (1-12)  
- `is_holiday`: Binary (per region)
- `is_weekend`: Binary
- `quarter`: Categorical (1-4)

### Interaction Terms

- `spend × region`
- `spend × is_holiday`
- `spend × quarter`

---

## Data Quality Notes

1. TikTok data requires special handling (late entry)
2. Revenue outliers should be investigated before modeling
3. Some regions have sparse data in early periods

---

## Next Steps

1. Preprocess data following recommendations above
2. Create adstock/saturation transformations
3. Build baseline MMM model
4. Validate with cross-validation across time periods
