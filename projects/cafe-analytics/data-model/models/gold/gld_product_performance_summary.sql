-- Gold Layer: Overall product performance rankings
-- Purpose: Menu optimization and product strategy
-- Business Value: Identify top performers, underperformers, promotion candidates
-- Built on: Silver slv_fact_product_sales

{{ config(
    materialized='table',
    schema='gold_zzb_dev',
    tags=['gold', 'business', 'product_analytics']
) }}

WITH product_totals AS (
    SELECT
        item_name,
        item_category,
        SUM(quantity_sold) AS total_quantity,
        ROUND(SUM(total_sales), 2) AS total_revenue,
        COUNT(DISTINCT customer_id) AS unique_customers,
        COUNT(DISTINCT date_created) AS days_sold
    FROM {{ ref('slv_fact_product_sales') }}
    GROUP BY item_name, item_category
),

category_totals AS (
    SELECT
        item_category,
        SUM(total_revenue) AS category_revenue
    FROM product_totals
    GROUP BY item_category
)

SELECT
    pt.item_name,
    pt.item_category,
    pt.total_quantity,
    pt.total_revenue,
    pt.unique_customers,
    pt.days_sold,

    -- Rankings
    ROW_NUMBER() OVER (ORDER BY pt.total_revenue DESC) AS overall_rank,
    ROW_NUMBER() OVER (PARTITION BY pt.item_category ORDER BY pt.total_revenue DESC) AS category_rank,

    -- Category share
    ROUND(100.0 * pt.total_revenue / ct.category_revenue, 2) AS pct_of_category_revenue,

    -- Performance tier
    CASE
        WHEN ROW_NUMBER() OVER (ORDER BY pt.total_revenue DESC) <= 5 THEN 'Top 5'
        WHEN ROW_NUMBER() OVER (ORDER BY pt.total_revenue DESC) <= 10 THEN 'Top 10'
        WHEN ROW_NUMBER() OVER (ORDER BY pt.total_revenue DESC) <= 20 THEN 'Top 20'
        WHEN pt.total_revenue < 100 THEN 'Underperformer'
        ELSE 'Standard'
    END AS performance_tier,

    -- Sales velocity
    ROUND(pt.total_quantity / NULLIF(pt.days_sold, 0), 2) AS avg_quantity_per_day,
    ROUND(pt.total_revenue / NULLIF(pt.days_sold, 0), 2) AS avg_revenue_per_day

FROM product_totals pt
JOIN category_totals ct ON pt.item_category = ct.item_category

ORDER BY pt.total_revenue DESC
