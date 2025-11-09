{{ config(
    materialized='table',
    schema='gold'
) }}


SELECT DISTINCT
    {{ dbt_utils.generate_surrogate_key(['product_name', 'product_variant']) }} AS product_key,
    product_name,
    product_variant,
    product_category,
    
    CURRENT_TIMESTAMP() AS created_at,
    CURRENT_TIMESTAMP() AS updated_at

FROM {{ ref('int_orders_item') }}
ORDER BY product_name, product_variant