{{ config(
    materialized='table',
    schema='gold'
) }}

SELECT
  i.order_item_sk,
  h.order_id,
  h.customer_id,
  c.customer_key,
  i.product_base_key,
  i.item_name,
  i.variant_name,
  i.category,
  i.variant_desc,
  i.variant_image,
  i.unit_price,
  i.quantity,
  i.line_item_total,
  i.total_options_price,
  i.base_variant_price,
  i.loaded_at
FROM {{ ref('int_order_items_flattened') }} AS i
JOIN {{ ref('lnd_orders') }} AS h
  ON i.order_id = h.order_id
LEFT JOIN {{ ref('dim_customer') }} AS c
  ON h.customer_id = c.customer_id
