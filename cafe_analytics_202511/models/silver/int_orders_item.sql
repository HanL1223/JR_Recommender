{{ config(
    materialized='view',
    schema='silver'
) }}

-- WITH items_base AS (
--   SELECT  
--     order_id,
--     position + 1 AS line_item_number,
--     CAST(JSON_VALUE(item, '$.name') AS STRING) AS item_name,
--     CAST(JSON_VALUE(item, '$.variant_name') AS STRING) AS variant_name,
--     CAST(JSON_VALUE(item, '$.category') AS STRING) AS category,
--     CAST(JSON_VALUE(item, '$.variant_desc') AS STRING) AS variant_desc,
--     CAST(JSON_VALUE(item, '$.variant_image') AS STRING) AS variant_image,
--     CAST(JSON_VALUE(item, '$.unitprice') AS INT64) AS unit_price,
--     CAST(JSON_VALUE(item, '$.price') AS INT64) AS line_item_total,
--     COALESCE(CAST(JSON_VALUE(item, '$.quantity') AS INT64), 1) AS quantity,
--     {{ dbt_utils.generate_surrogate_key(['order_id', 'position']) }} AS order_item_sk,
--     CURRENT_TIMESTAMP() AS loaded_at
--   FROM {{ ref('lnd_orders') }} AS src,
--   UNNEST(JSON_EXTRACT_ARRAY(items, '$.cart')) AS item WITH OFFSET AS position
-- ),

-- options AS (
--   SELECT  
--     {{ dbt_utils.generate_surrogate_key(['order_id', 'position']) }} AS order_item_sk,
--     CAST(JSON_VALUE(option, '$.name') AS STRING) AS option_name,
--     CAST(JSON_VALUE(option, '$.value') AS STRING) AS option_value,
--     CAST(JSON_VALUE(option, '$.price') AS INT64) AS option_price
--   FROM {{ ref('lnd_orders') }} AS src,
--   UNNEST(JSON_EXTRACT_ARRAY(items, '$.cart')) AS item WITH OFFSET AS position,
--   UNNEST(JSON_EXTRACT_ARRAY(item, '$.options')) AS option
-- )

-- SELECT
--   ib.*,
--   o.option_name,
--   o.option_value,
--   o.option_price,
--   COALESCE(o.option_price, 0) AS total_options_price,
--   ib.unit_price - COALESCE(o.option_price, 0) AS base_variant_price,

--   {{ dbt_utils.generate_surrogate_key(['ib.item_name', 'ib.variant_name']) }} AS product_base_key,
--   {{ dbt_utils.generate_surrogate_key(['ib.item_name', 'ib.variant_name', 'o.option_name', 'o.option_value']) }} AS product_option_key

-- FROM items_base ib
-- LEFT JOIN options o
--   ON ib.order_item_sk = o.order_item_sk
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
FROM parsed_cart
ORDER BY order_id, order_item_id
