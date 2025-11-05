-- Gold Layer: Daily product sales performance
-- Purpose: Time-series analysis of product sales by day
-- Business Value: Track daily trends, identify seasonal patterns
-- Built on: Silver slv_fact_product_sales

{{ config(
    materialized='table',
    schema='gold_zzb_dev',
    tags=['gold', 'business', 'time_series']
) }}

SELECT
    date_created AS sales_date,
    item_name,
    item_category,
    SUM(quantity_sold) AS daily_quantity,
    ROUND(SUM(total_sales), 2) AS daily_revenue,
    COUNT(DISTINCT customer_id) AS unique_customers,
    ROUND(AVG(total_sales), 2) AS avg_sale_per_customer,

    -- Day attributes
    FORMAT_DATE('%A', date_created) AS day_of_week,
    EXTRACT(MONTH FROM date_created) AS month,
    EXTRACT(YEAR FROM date_created) AS year

FROM {{ ref('slv_fact_product_sales') }}

GROUP BY
    date_created,
    item_name,
    item_category

ORDER BY sales_date DESC, daily_revenue DESC
