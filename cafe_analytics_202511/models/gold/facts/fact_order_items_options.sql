{{ config(
    materialized='table',
    schema='gold'
) }}

SELECT
  o.order_item_sk,
  h.order_id,
  h.customer_id,
  c.customer_key,
  o.option_name,
  o.option_value,
  o.option_price,
  o.option_quantity,
  {{ dbt_utils.generate_surrogate_key(['o.option_name', 'o.option_value']) }} AS product_option_key,
  o.loaded_at
FROM {{ ref('int_order_item_options') }} AS o
JOIN {{ ref('lnd_orders') }} AS h
  ON o.order_id = h.order_id
LEFT JOIN {{ ref('dim_customer') }} AS c
  ON h.customer_id = c.customer_id
