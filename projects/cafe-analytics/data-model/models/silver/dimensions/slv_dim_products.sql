-- Silver Dimension: Product master table
-- Purpose: Product-level aggregated attributes
-- Type: SCD Type 1 (overwrite)
-- Grain: One row per product

{{ config(
    materialized='table',
    schema='silver_zzb_dev',
    tags=['silver', 'silver_dimensions', 'dim']
) }}

SELECT
    item AS item_name,
    category AS item_category,
    ROUND(AVG(item_price), 2) AS avg_price,
    ROUND(MIN(item_price), 2) AS min_price,
    ROUND(MAX(item_price), 2) AS max_price,
    COUNT(DISTINCT customer_id) AS unique_customers,
    COUNT(DISTINCT order_id) AS total_orders

FROM {{ ref('slv_fact_order_details') }}

GROUP BY item, category

ORDER BY total_orders DESC
