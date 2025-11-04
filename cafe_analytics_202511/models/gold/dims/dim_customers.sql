{{ config(
    materialized='table',
    schema='gold'
) }}

WITH customer_summary AS (
  SELECT
    customer_id,
    MIN(order_date_key) AS first_order_date,
    MAX(order_date_key) AS last_order_date, 
    COUNT(DISTINCT order_id) AS total_orders,
    SUM(cart_total_price) AS total_lifetime_value
  FROM {{ ref('int_orders_header') }}
  GROUP BY customer_id
)

SELECT
  {{ dbt_utils.generate_surrogate_key(['customer_id']) }} AS customer_key,
  
  customer_id,
  first_order_date,
  last_order_date,
  total_orders,
  total_lifetime_value,
  
  -- Segmentation
  CASE
    WHEN total_orders = 1 THEN 'New'
    WHEN total_orders BETWEEN 2 AND 10 THEN 'Regular'
    WHEN total_orders > 10 THEN 'VIP'
  END AS customer_segment,
  
  -- Recency / Active Flag
  CASE 
    WHEN DATE_DIFF(CURRENT_DATE(), last_order_date, DAY) <= 120 THEN TRUE
    ELSE FALSE
  END AS is_active_flag,
  
  CURRENT_TIMESTAMP() AS created_at,
  CURRENT_TIMESTAMP() AS updated_at

FROM customer_summary;
