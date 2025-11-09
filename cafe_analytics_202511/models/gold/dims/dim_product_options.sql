{{ config(
    materialized='table',
    schema='gold'
) }}

WITH distinct_options AS (
    SELECT DISTINCT
        JSON_VALUE(cart_item, '$.name') AS product_name,
        JSON_VALUE(cart_item, '$.variant_name') AS product_variant,
        JSON_VALUE(cart_item, '$.category') AS product_category,
        JSON_VALUE(option, '$.name') AS product_option_name,
        JSON_VALUE(option, '$.value') AS product_option_value,
        CAST(JSON_VALUE(option, '$.price') AS INT64) AS product_option_price
    FROM {{ ref('lnd_orders') }},
    UNNEST(JSON_EXTRACT_ARRAY(items, '$.cart')) AS cart_item,
    UNNEST(JSON_EXTRACT_ARRAY(cart_item, '$.options')) AS option
)

SELECT
    {{ dbt_utils.generate_surrogate_key([
        'product_name',
        'product_variant',
        'product_option_name',
        'product_option_value'
    ]) }} AS product_option_key,

    {{ dbt_utils.generate_surrogate_key([
        'product_name',
        'product_variant'
    ]) }} AS product_key,

    product_name,
    product_variant,
    product_category,
    product_option_name,
    product_option_value,
    product_option_price,

    CURRENT_TIMESTAMP() AS created_at,
    CURRENT_TIMESTAMP() AS updated_at
FROM distinct_options
ORDER BY product_name, product_variant, product_option_name
