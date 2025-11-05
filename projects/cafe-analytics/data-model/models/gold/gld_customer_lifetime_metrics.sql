-- Gold Layer: Customer lifetime value and behavior metrics
-- Purpose: Customer segmentation and churn risk analysis
-- Business Value: Identify VIP customers, retention opportunities, churn risk
-- Built on: Silver slv_dim_customers and slv_fact_order_details

{{ config(
    materialized='table',
    schema='gold_zzb_dev',
    tags=['gold', 'business', 'customer_analytics']
) }}

WITH customer_activity AS (
    SELECT
        customer_id,
        order_time,
        order_price,
        LAG(order_time) OVER (PARTITION BY customer_id ORDER BY order_time) AS prev_order_time
    FROM {{ ref('slv_fact_order_details') }}
),

customer_intervals AS (
    SELECT
        customer_id,
        DATE_DIFF(DATE(order_time), DATE(prev_order_time), DAY) AS days_between_orders
    FROM customer_activity
    WHERE prev_order_time IS NOT NULL
)

SELECT
    dc.customer_id,
    dc.first_order_date,
    dc.last_order_date,
    dc.total_orders,
    dc.avg_order_value,
    dc.total_lifetime_value,

    -- Recency: Days since last order
    DATE_DIFF(CURRENT_DATE(), DATE(dc.last_order_date), DAY) AS days_since_last_order,

    -- Frequency: Average days between orders
    ROUND(AVG(ci.days_between_orders), 1) AS avg_days_between_orders,

    -- Customer lifetime in days
    DATE_DIFF(DATE(dc.last_order_date), DATE(dc.first_order_date), DAY) AS customer_lifetime_days,

    -- Customer Segment (RFM-inspired)
    CASE
        WHEN dc.total_orders >= 10 AND dc.avg_order_value >= 15 THEN 'VIP'
        WHEN dc.total_orders >= 5 THEN 'Loyal'
        WHEN dc.total_orders = 1 THEN 'New'
        ELSE 'Regular'
    END AS customer_segment,

    -- Churn Risk Level
    CASE
        WHEN DATE_DIFF(CURRENT_DATE(), DATE(dc.last_order_date), DAY) > 90 THEN 'High Risk'
        WHEN DATE_DIFF(CURRENT_DATE(), DATE(dc.last_order_date), DAY) > 60 THEN 'Medium Risk'
        WHEN DATE_DIFF(CURRENT_DATE(), DATE(dc.last_order_date), DAY) > 30 THEN 'Low Risk'
        ELSE 'Active'
    END AS churn_risk

FROM {{ ref('slv_dim_customers') }} dc
LEFT JOIN customer_intervals ci ON dc.customer_id = ci.customer_id

GROUP BY
    dc.customer_id,
    dc.first_order_date,
    dc.last_order_date,
    dc.total_orders,
    dc.avg_order_value,
    dc.total_lifetime_value

ORDER BY dc.total_lifetime_value DESC
