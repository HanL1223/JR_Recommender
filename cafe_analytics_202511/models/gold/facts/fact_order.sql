{{ config(
    materialized='table',
    schema='gold'
) }}


SELECT
    order_id,
    {{ dbt_utils.generate_surrogate_key(['customer_id']) }} AS customer_key,
    
    -- Order variable

order_date,
 order_time,
order_total_price,
 order_gst,
order_surcharge,
 order_display_price,
order_display_gst,
 cart_size,
cart_order_time

    -- 
    --ETL Measure
    CURRENT_TIMESTAMP() AS created_at,
    CURRENT_TIMESTAMP() AS updated_at
FROM  {{ref('int_orders')}} src 
left join {{ref('dim_customers')}} dc on src.customer_id = dc.customer_id
