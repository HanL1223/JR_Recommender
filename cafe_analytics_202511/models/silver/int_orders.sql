{{ config(
    materialized='view',
    schema='silver'
) }}
#Per Order Level 


SELECT
    order_id,
    customer_id,
    DATE(TIMESTAMP(date_created)) AS order_date,
    TIME(TIMESTAMP(date_created)) AS order_time,
    CAST(JSON_VALUE(items, '$.cart_total_price') AS INT64) AS order_total_price,
    CAST(JSON_VALUE(items, '$.cart_gst') AS NUMERIC) AS order_gst,
    CAST(JSON_VALUE(items, '$.cart_surcharge') AS INT64) AS order_surcharge,
    LTRIM(JSON_VALUE(items, '$.cart_total_price_display'), '$') AS order_display_price,
    LTRIM(JSON_VALUE(items, '$.cart_gst_display'), '$') AS order_display_gst,
    JSON_VALUE(items, '$.cart_size') AS cart_size,
    JSON_VALUE(items, '$.order_time') AS cart_order_time
FROM  {{ ref('lnd_orders') }}