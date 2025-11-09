{{ config(
    materialized='table',
    schema='gold'
) }}

SELECT
    order_id,
    --FK
    {{ dbt_utils.generate_surrogate_key(['dc.customer_id']) }} AS customer_key,
    -- FK
{{ dbt_utils.generate_surrogate_key(['product_name', 'product_variant']) }} AS product_key,
{{ dbt_utils.generate_surrogate_key([ 'product_name',
        'product_variant',
        'product_option_name',
        'product_option_value']) }} AS product_option_key,
    -- Order variable

order_date,
 order_time,
order_total_price,
 order_gst,
order_surcharge,
 order_display_price,
order_display_gst,
 cart_size,
cart_order_time,
--ETL Measure
    CURRENT_TIMESTAMP() AS created_at,
    CURRENT_TIMESTAMP() AS updated_at
FROM  {{ref('int_order_item_options')}} src 
left join {{ref('dim_customers')}} dc on src.customer_id = dc.customer_id