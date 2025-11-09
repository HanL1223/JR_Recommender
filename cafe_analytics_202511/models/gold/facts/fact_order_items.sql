{{ config(
        materialized='table',
        schema='gold') }}



SELECT 

order_id,
order_item_id,
order_date,
 order_time,
order_total_price,
 order_gst,
order_surcharge,
 order_display_price,
order_display_gst,
 cart_size,
cart_order_time,
--product


-- FK
{{ dbt_utils.generate_surrogate_key(['customer_id']) }} AS customer_key,
{{ dbt_utils.generate_surrogate_key(['product_name', 'product_variant']) }} AS product_key

from  {{ref('int_orders_item')}} src
