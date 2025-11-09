{{ config(
    materialized='table',
    schema='gold'
) }}

WITH joined AS (
    SELECT
        i.order_id,
        i.customer_id,
        i.order_item_id,
        o.order_date,
        o.order_time,
        i.product_name,
        i.product_variant,
        i.product_category,
        i.product_total_price,
        io.product_option_name,
        io.product_option_value,
        io.product_option_price
    FROM {{ ref('int_orders_item') }} i
    LEFT JOIN {{ ref('int_order_item_options') }} io
        ON i.order_id = io.order_id
       AND i.order_item_id = io.order_item_id
    LEFT JOIN {{ ref('int_orders') }} o
        ON i.order_id = o.order_id
)

SELECT
    -- Surrogate key for the fact record
    {{ dbt_utils.generate_surrogate_key([
        'order_id',
        'order_item_id',
        'product_option_name',
        'product_option_value'
    ]) }} AS order_item_fact_key,

    -- Foreign keys
    {{ dbt_utils.generate_surrogate_key(['customer_id']) }} AS customer_key,
    {{ dbt_utils.generate_surrogate_key(['product_name','product_variant']) }} AS product_key,
    {{ dbt_utils.generate_surrogate_key(['product_name','product_variant','product_option_name','product_option_value']) }} AS product_option_key,

    -- Natural business keys
    order_id,
    order_item_id,
    customer_id,

    -- Date/time keys
    order_date,
    order_time,

    -- Product details
    product_name,
    product_variant,
    product_category,
    product_option_name,
    product_option_value,

    -- Financial measures
    COALESCE(product_total_price, 0) AS base_price,
    COALESCE(product_option_price, 0) AS option_price,
    COALESCE(product_total_price, 0) + COALESCE(product_option_price, 0) AS total_price,


    CURRENT_TIMESTAMP() AS created_at,
    CURRENT_TIMESTAMP() AS updated_at
FROM joined
ORDER BY order_id, order_item_id
