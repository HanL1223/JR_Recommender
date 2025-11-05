-- Silver Dimension: Customer master table
-- Purpose: Customer-level aggregated attributes
-- Type: SCD Type 1 (overwrite)
-- Grain: One row per customer

{{ config(
    materialized='table',
    schema='silver_zzb_dev',
    tags=['silver', 'silver_dimensions', 'dim']
) }}

SELECT
    customer_id,
    MIN(order_time) AS first_order_date,
    MAX(order_time) AS last_order_date,
    COUNT(DISTINCT order_id) AS total_orders,
    ROUND(AVG(order_price), 2) AS avg_order_value,
    ROUND(SUM(order_price), 2) AS total_lifetime_value

FROM {{ ref('slv_fact_order_details') }}

GROUP BY customer_id

ORDER BY total_lifetime_value DESC
