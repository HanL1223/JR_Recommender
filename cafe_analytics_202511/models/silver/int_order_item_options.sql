{{ config(
    materialized='view',
    schema='silver'
) }}


WITH parsed_cart AS (
  SELECT
    order_id,
    customer_id,
    ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY cart_item) AS order_item_id, -- unique per cart row
    TIMESTAMP(date_created) AS order_ts,
    cart_item
FROM {{ ref('lnd_orders') }},
  UNNEST(JSON_EXTRACT_ARRAY(items, '$.cart')) AS cart_item
)
SELECT
  order_id,
  order_item_id,
  customer_id,
  JSON_VALUE(cart_item, '$.name') AS product_name,
  JSON_VALUE(cart_item, '$.variant_name') AS product_variant,
  JSON_VALUE(cart_item, '$.category') AS product_category,
  CAST(JSON_VALUE(cart_item, '$.price') AS INT64) AS product_total_price,
  JSON_VALUE(option, '$.name') AS product_option_name,
  JSON_VALUE(option, '$.value') AS product_option_value,
  CAST(JSON_VALUE(option, '$.price') AS INT64) AS product_option_price
FROM parsed_cart,
  UNNEST(JSON_EXTRACT_ARRAY(cart_item, '$.options')) AS option
ORDER BY order_id, order_item_id
