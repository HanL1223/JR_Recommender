{{ config(
    materialized='view',
    schema='silver'
) }}

SELECT
  order_id,
  customer_id,
  ip_addr,
  date_created,
  date_paid,
  total AS order_total,
  status AS status_code,
  
  -- Extract from items
  CAST(JSON_VALUE(items, '$.cart_size') AS INT64) AS cart_size,
  CAST(JSON_VALUE(items, '$.cart_total_price') AS INT64) AS cart_total_price,
  CAST(JSON_VALUE(items, '$.cart_gst') AS FLOAT64) AS cart_gst,
  CAST(JSON_VALUE(items, '$.cart_surcharge') AS INT64) AS cart_surcharge,
  CAST(JSON_VALUE(items, '$.order_time') AS STRING) AS order_time,
  CAST(JSON_VALUE(items, '$.order_time_verbose') AS STRING) AS order_time_verbose,
  CAST(JSON_VALUE(items, '$.order_name') AS STRING) AS order_name,
  CAST(JSON_VALUE(items, '$.order_phone') AS STRING) AS order_phone,
  
  -- Derived fields timestamp
  
  FORMAT_DATE('%Y%m%d',EXTRACT(DATE FROM date_created)) as order_date_key,
  FORMAT_DATE('%Y%m%d',EXTRACT(DATE FROM date_paid)) as paid_date_key,
  FORMAT_DATETIME('%H:%M', date_created) AS order_time_key,
  FORMAT_DATETIME('%H:%M', date_paid) AS paid_time_key,
  TIMESTAMP_DIFF(date_paid, date_created, SECOND) AS processing_seconds,
  
    -- Day part classification
  CASE 
    WHEN EXTRACT(HOUR FROM date_created) BETWEEN 6 AND 9 THEN 'Breakfast'
    WHEN EXTRACT(HOUR FROM date_created) BETWEEN 10 AND 11 THEN 'Morning'
    WHEN EXTRACT(HOUR FROM date_created) BETWEEN 12 AND 13 THEN 'Lunch'
    WHEN EXTRACT(HOUR FROM date_created) BETWEEN 14 AND 16 THEN 'Afternoon'
    WHEN EXTRACT(HOUR FROM date_created) BETWEEN 17 AND 20 THEN 'Evening'
    ELSE 'Other'
  END AS order_period,
  
  CURRENT_TIMESTAMP() AS loaded_at

FROM {{ ref('lnd_orders') }}
WHERE order_id IS NOT NULL