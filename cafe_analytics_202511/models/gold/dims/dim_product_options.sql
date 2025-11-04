{{ config(
    materialized='table',
    schema='gold'
) }}

WITH base AS (
  SELECT  
    order_id,
    position + 1 AS line_item_number,
    CAST(JSON_VALUE(item, '$.name') AS STRING) AS item_name,
    CAST(JSON_VALUE(item, '$.variant_name') AS STRING) AS variant_name,
    CAST(JSON_VALUE(item, '$.category') AS STRING) AS category,
    CAST(JSON_VALUE(item, '$.variant_image') AS STRING) AS variant_image
  FROM {{ ref('lnd_orders') }} AS src,
  UNNEST(JSON_EXTRACT_ARRAY(items, '$.cart')) AS item WITH OFFSET AS position
),
options AS (
  SELECT  
    b.item_name,
    b.variant_name,
    b.category,
    b.variant_image,
    CAST(JSON_VALUE(option, '$.name') AS STRING) AS option_name,
    CAST(JSON_VALUE(option, '$.value') AS STRING) AS option_value,
    CAST(JSON_VALUE(option, '$.price') AS INT64) AS option_price
  FROM base b
  LEFT JOIN {{ ref('lnd_orders') }} AS src
    ON b.order_id = src.order_id,
    UNNEST(JSON_EXTRACT_ARRAY(items, '$.cart')) AS item WITH OFFSET AS position,
    UNNEST(JSON_EXTRACT_ARRAY(item, '$.options')) AS option
)

SELECT DISTINCT
  {{ dbt_utils.generate_surrogate_key([
    'item_name',
    'variant_name',
    'option_name',
    'option_value'
  ]) }} AS product_option_key,

  {{ dbt_utils.generate_surrogate_key([
    'item_name',
    'variant_name'
  ]) }} AS product_base_key, -- FK to dim_product_base

  item_name AS product_name,
  variant_name,
  option_name,
  option_value,
  option_price,
  CASE WHEN option_price > 0 THEN 'Paid' ELSE 'Included' END AS option_type,

--   CURRENT_TIMESTAMP() AS created_at,
--   CURRENT_TIMESTAMP() AS updated_at

FROM options
WHERE option_name IS NOT NULL
