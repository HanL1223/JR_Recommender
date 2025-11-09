{{ config(
    materialized='table',
    schema='gold'
) }}

SELECT
    src.order_id,

    {{ dbt_utils.generate_surrogate_key(['dc.customer_id']) }} AS customer_key,

    -- Order variables
    src.order_date,
    src.order_time,
    src.order_total_price,
    src.order_gst,
    src.order_surcharge,
    src.order_display_price,
    src.order_display_gst,
    src.cart_size,
    src.cart_order_time,

    -- ETL metadata
    CURRENT_TIMESTAMP() AS created_at,
    CURRENT_TIMESTAMP() AS updated_at

FROM {{ ref('int_orders') }} AS src
LEFT JOIN {{ ref('dim_customers') }} AS dc
    ON src.customer_id = dc.customer_id
