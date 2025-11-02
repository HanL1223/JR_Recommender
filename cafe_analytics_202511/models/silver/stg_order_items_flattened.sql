{{ config(materialized='view',
schema = 'silver') }}

SELECT  
  order_id,
  CAST(JSON_VALUE(item, '$.name') AS STRING) AS item_name,
  CAST(JSON_VALUE(item, '$.variant_name') AS STRING) AS variant_name,
  CAST(JSON_VALUE(item, '$.category') AS STRING) AS category,
  CAST(JSON_VALUE(item, '$.price') AS INT64) AS item_price,
  
  CAST(JSON_VALUE(option, '$.name') AS STRING) AS option_name,
  CAST(JSON_VALUE(option, '$.value') AS STRING) AS option_value,
  CAST(JSON_VALUE(option, '$.price') AS NUMERIC) AS option_price,
  
  {{ dbt_utils.generate_surrogate_key([
    'order_id',
    "CAST(JSON_VALUE(item, '$.name') AS STRING)",
    "CAST(JSON_VALUE(item, '$.variant_name') AS STRING)"
  ]) }} AS order_item_sk,
  
FROM {{ ref('lnd_orders') }} as src,
UNNEST(JSON_EXTRACT_ARRAY(items, '$.cart')) AS item, 
UNNEST(JSON_EXTRACT_ARRAY(item, '$.options')) AS option

WHERE order_id IS NOT NULL