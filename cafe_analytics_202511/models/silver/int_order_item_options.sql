{{ config(
    materialized='view',
    schema='silver'
) }}

WITH parsed_cart AS (
    SELECT
        order_id,
        customer_id,
        ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY cart_item) AS order_item_id,
        DATE(TIMESTAMP(date_created)) AS order_date,
        TIME(TIMESTAMP(date_created)) AS order_time,
        cart_item,
        CAST(JSON_VALUE(items, '$.cart_total_price') AS INT64) AS order_total_price,
        CAST(JSON_VALUE(items, '$.cart_gst') AS NUMERIC) AS order_gst,
        CAST(JSON_VALUE(items, '$.cart_surcharge') AS INT64) AS order_surcharge,
        LTRIM(JSON_VALUE(items, '$.cart_total_price_display'), '$') AS order_display_price,
        LTRIM(JSON_VALUE(items, '$.cart_gst_display'), '$') AS order_display_gst,
        JSON_VALUE(items, '$.cart_size') AS cart_size,
        JSON_VALUE(items, '$.order_time') AS cart_order_time
    FROM {{ ref('lnd_orders') }},
         UNNEST(JSON_EXTRACT_ARRAY(items, '$.cart')) AS cart_item
)

SELECT
    order_id,
    order_item_id,
    customer_id,

    ----------------------
    -- PRODUCT FIELDS
    ----------------------
    JSON_VALUE(cart_item, '$.name') AS product_name,
    JSON_VALUE(cart_item, '$.variant_name') AS product_variant,

    /* ------------------------------
       CORRECTED CATEGORY MAPPING
       ------------------------------ */
    CASE
        
        /* Hot Drinks */
        WHEN JSON_VALUE(cart_item, '$.name') IN (
            'White Hot Chocolate', 'Short Macchiato', 'Piccolo', 'Mocha',
            'Magic', 'Long Macchiato', 'Long Black', 'Latte', 'Hot Chocolate',
            'Golden Latte', 'Flat White', 'Espresso', 'Dirty Chai Latte',
            'Chai Latte', 'Cappuccino', 'Babychino'
        ) THEN 'Hot Drinks'

        /* Cold Drinks */
        WHEN JSON_VALUE(cart_item, '$.name') IN (
            'Thickshake (Syrup)', 'Smoothie', 'Milkshake (Syrup)',
            'Iced Mocha with Cream', 'Iced Coffee with Cream',
            'Iced Chocolate with Cream', 'Ice Latte', 'Ice Chai',
            'Freshly Squeezed Juice', 'Affogato', 'Coffee Milkshake',
            'Coffee Frappe', 'Drinks from fridge'
        ) THEN 'Cold Drinks'
        
        /* Everything else is Kitchen */
        ELSE 'Kitchen'
    END AS product_category,

    CAST(JSON_VALUE(cart_item, '$.price') AS INT64) AS product_total_price,

    ----------------------
    -- PRODUCT OPTIONS
    ----------------------
    JSON_VALUE(option, '$.name') AS product_option_name,
    JSON_VALUE(option, '$.value') AS product_option_value,
    CAST(JSON_VALUE(option, '$.price') AS INT64) AS product_option_price,

    ----------------------
    -- ORDER FIELDS
    ----------------------
    order_date,
    order_time,
    order_total_price,
    order_gst,
    order_surcharge,
    order_display_price,
    order_display_gst,
    cart_size,
    cart_order_time

FROM parsed_cart,
     UNNEST(JSON_EXTRACT_ARRAY(cart_item, '$.options')) AS option

ORDER BY order_id, order_item_id
