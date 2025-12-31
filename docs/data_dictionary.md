# Data Dictionary

Source: `data/raw/conjura_mmm_data_dictionary.xlsx`

## Identifiers

| Field                                 | Definition                                                 | Notes                                                 |
| ------------------------------------- | ---------------------------------------------------------- | ----------------------------------------------------- |
| `mmm_timeseries_id`                   | Unique identifier for a single MMM timeseries              | Take care with currency conversion when aggregating   |
| `organisation_id`                     | Unique, anonymous identifier for an eCommerce brand        |                                                       |
| `organisation_vertical`               | Top-level category of highest selling products             | Google's eCommerce taxonomy                           |
| `organisation_subvertical`            | Sub-category of highest selling products                   |                                                       |
| `organisation_primary_territory_name` | Territory with highest average daily orders                |                                                       |
| `territory_name`                      | Multi-country "All Territories" and country-level roll-ups | Some orgs have multiple territory splits              |
| `currency_code`                       | Currency for monetary fields                               | Uses primary territory currency for "All Territories" |
| `date_day`                            | Observation date                                           | Minimum 449 sequential days per timeseries            |

## Target Variables

| Field                            | Definition                                               | Notes                       |
| -------------------------------- | -------------------------------------------------------- | --------------------------- |
| `first_purchases`                | Number of web purchases for new customers (acquisitions) |                             |
| `first_purchases_units`          | Number of units purchased by new customers               |                             |
| `first_purchases_original_price` | New customer total value before discount                 |                             |
| `first_purchases_gross_discount` | New customer total discount value                        | **⚠️ DATA LEAKAGE WARNING** |
| `all_purchases`                  | Number of web purchases for all customers                |                             |
| `all_purchases_units`            | Number of units purchased by all customers               |                             |
| `all_purchases_original_price`   | Total value of merchandise before discount               |                             |
| `all_purchases_gross_discount`   | Total discount value                                     | **⚠️ DATA LEAKAGE WARNING** |

> **⚠️ WARNING**: Discount rates are only available for completed purchases. Take care to avoid data leakage if using as a control variable.

## Marketing Spend Channels

| Field                      | Definition                                |
| -------------------------- | ----------------------------------------- |
| `google_paid_search_spend` | Google spend on (non-branded) paid search |
| `google_shopping_spend`    | Google spend on shopping ads              |
| `google_pmax_spend`        | Google spend on Performance Max campaigns |
| `google_display_spend`     | Google spend on display ads               |
| `google_video_spend`       | Google spend on video ads                 |
| `meta_facebook_spend`      | Meta spend on Facebook ads                |
| `meta_instagram_spend`     | Meta spend on Instagram ads               |
| `meta_other_spend`         | Meta spend on ads from other platforms    |
| `tiktok_spend`             | TikTok spend                              |

> **Note**: Spend from listed sources may not be present in all timeseries for a given brand, given different utilisation by territory.

## Engagement Metrics

| Field                              | Definition                                 | Notes                                    |
| ---------------------------------- | ------------------------------------------ | ---------------------------------------- |
| `<platform>_<channel>_clicks`      | Ad clicks for each channel with spend      |                                          |
| `<platform>_<channel>_impressions` | Ad impressions for each channel with spend |                                          |
| `<channel>_clicks`                 | Web traffic from "non-paid" channels       | Affiliate and branded search not tracked |

## Engineered Features

The pipeline generates several derived features to improve model signal.

| Feature Type            | Description                                                  |
| ----------------------- | ------------------------------------------------------------ |
| **Efficiency (CTR)**    | `click / impression` for each channel.                       |
| **Cost (CPC)**          | `spend / click` for each channel.                            |
| **Rolling (7D)**        | 7-day rolling Std (volatility) for all spend channels.       |
| **Share of Spend**      | `% of total daily spend` allocated to each channel.          |
| **Customer Behavior**   | ~~Disabled: data leakage risk~~                              |
| **Traffic (Non-Paid)**  | Direct, Branded, Organic, Email, Referral, Other clicks.     |
| **Temporal**            | Holiday indicator, linear trend, and Q4 seasonality.         |
| **Fourier Seasonality** | 2-term Fourier series (sin/cos) for week and month patterns. |
| **Cyclic Encoding**     | WEEK_SIN/COS and MONTH_SIN/COS (1st and 2nd order).          |
| **Adstock**             | Geometric decay learned per channel/territory (`L_MAX=6`).   |
| **Saturation**          | Hill function with learned L (half-sat) and k (steepness).   |

> Total features in the Hierarchical Model: **27** (9 spend + 10 traffic/control + 8 season)
>
> **Last updated:** 2025-12-31 (post-audit). See [CHANGELOG.md](/CHANGELOG.md) for recent changes.
