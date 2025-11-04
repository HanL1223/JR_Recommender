{{ config(
    materialized='view',
    schema='silver'
) }}

SELECT  
  order_id,
  {{ dbt_utils.generate_surrogate_key(['order_id', 'position']) }} AS order_item_sk,
  CAST(JSON_VALUE(option, '$.name') AS STRING) AS option_name,
  CAST(JSON_VALUE(option, '$.value') AS STRING) AS option_value,
  CAST(JSON_VALUE(option, '$.price') AS INT64) AS option_price,
  SAFE_CAST(JSON_VALUE(option, '$.value') AS FLOAT64) AS option_quantity,
  CURRENT_TIMESTAMP() AS loaded_at
FROM {{ ref('lnd_orders') }} AS src,
UNNEST(JSON_EXTRACT_ARRAY(items, '$.cart')) AS item WITH OFFSET AS position,
UNNEST(JSON_EXTRACT_ARRAY(item, '$.options')) AS option
WHERE order_id IS NOT NULL
