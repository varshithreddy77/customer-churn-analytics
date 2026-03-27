# Business Framing — Customer Churn Prediction

## Problem Statement

A telecom company is losing customers (churn). Each lost customer represents
significant revenue loss — acquiring a replacement costs ~6x more than retaining
the existing customer. The business needs a system that identifies customers
likely to churn **before** they leave, so the retention team can intervene.

## Business Objective

Reduce revenue loss from churn by identifying high-risk customers early and
targeting them with cost-effective retention offers.

## Target Users

- **Customer Success / Retention Team** — receives ranked list of at-risk customers
- **Marketing** — designs targeted campaigns for medium-risk segments
- **Finance** — estimates revenue at risk and retention ROI

## Cost Assumptions (Industry-Standard Telecom Estimates)

| Metric                            | Value     |
|-----------------------------------|-----------|
| Avg Monthly Revenue per Customer  | $65       |
| Avg Customer Lifetime (non-churn) | 24 months |
| Customer Lifetime Value (CLV)     | $1,560    |
| New Customer Acquisition Cost     | ~$300     |
| Retention Offer Cost              | ~$50      |

**Key insight:** Retaining a customer costs **$50**. Losing one and replacing
them costs **$300+ acquisition + $1,560 lost CLV**. Retention is ~30x cheaper
than replacement.

## Why Errors Are Not Equal

| Error Type     | What Happens                                      | Business Cost                     |
|----------------|---------------------------------------------------|-----------------------------------|
| False Negative | Model says "won't churn" → customer leaves        | **HIGH** — $1,560 CLV lost        |
| False Positive | Model says "will churn" → customer actually stays  | **LOW** — $50 wasted on offer     |

**Implication:** We optimize for **Recall** (catching churners) over Precision.
Missing a churner is far more expensive than over-targeting a loyal customer.

## Success Metrics

### Model KPIs
- **ROC-AUC > 0.85** — strong overall discrimination
- **Recall > 0.75** — catch at least 75% of churners
- **Precision** — monitored but secondary to Recall
- **F1 Score** — balance metric

### Business KPIs
- Overall churn rate (baseline: ~26% in Telco dataset)
- Estimated revenue at risk from high-risk customers
- Retention campaign cost vs. projected revenue saved
- Churn rate by segment (contract type, tenure band, payment method)

## Risk Tiers & Intervention Logic

| Churn Probability | Risk Tier   | Action                           |
|-------------------|-------------|----------------------------------|
| >= 0.50           | HIGH RISK   | Priority outreach + strong offer |
| 0.30 – 0.50      | MEDIUM RISK | Automated email campaign         |
| < 0.30            | LOW RISK    | No action needed                 |

## Hypotheses to Validate in EDA

1. Month-to-month contract customers churn more than annual/two-year customers
2. New customers (low tenure, < 12 months) churn at higher rates
3. High monthly charges correlate with churn (price sensitivity)
4. Electronic check payment method correlates with higher churn
5. Senior citizens exhibit different churn behavior
6. Customers without tech support or online security churn more (lower stickiness)
