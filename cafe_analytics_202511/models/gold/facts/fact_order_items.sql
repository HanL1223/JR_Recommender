{{ config(
    materialized='table',
    schema='gold'
) }}

SELECT
  i.order_item_sk as order_item_key,
  
  -- Foreign keys
  -- {{ dbt_utils.generate_surrogate_key(['i.order_id']) }} AS order_key,
  {{ dbt_utils.generate_surrogate_key(['i.item_name', 'i.variant_name']) }} AS product_key,
  {{ dbt_utils.generate_surrogate_key(['o.customer_id']) }} AS customer_key,
  order_date_key,
  
  -- Degenerate dimensions
  i.order_id,
  i.line_item_number,
  
  -- Measures
  i.base_variant_price,
  i.total_options_price,
  i.unit_price,
  i.quantity,
  i.line_item_total,
  
  CURRENT_TIMESTAMP() AS created_at

FROM {{ ref('int_order_items_flattened') }} i
INNER JOIN {{ ref('int_orders_header') }} o ON i.order_id = o.order_id