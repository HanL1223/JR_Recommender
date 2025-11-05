-- Silver Core Fact: Product sales aggregated by order
-- Purpose: Item-level sales summary per customer per order
-- Granularity: One row per (order, customer, item) combination
-- Built on: slv_fact_order_details

{{ config(
    materialized='table',
    schema='silver_zzb_dev',
    tags=['silver', 'silver_core', 'fact']
) }}

SELECT
    DATE(order_time) AS date_created,
    customer_id,
    item AS item_name,
    category AS item_category,
    COUNT(*) AS quantity_sold,
    ROUND(SUM(item_price), 2) AS total_sales

FROM {{ ref('slv_fact_order_details') }}

GROUP BY
    DATE(order_time),
    customer_id,
    item,
    category

ORDER BY date_created DESC, total_sales DESC
