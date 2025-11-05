-- Silver Core Fact: Order-Item level details
-- Purpose: Clean transaction-level data with parsed JSON items
-- Granularity: One row per order-item combination
-- Filters: Only paid orders (status 1 or 2)

{{ config(
    materialized='table',
    schema='silver_zzb_dev',
    tags=['silver', 'silver_core', 'fact']
) }}

WITH order_base AS (
    SELECT
        order_id,
        customer_id,
        date_created,
        date_paid,
        ROUND(total / 100.0, 2) AS total_amount,
        status,
        CAST(JSON_EXTRACT_SCALAR(items_json, '$.cart_total_price') AS FLOAT64) / 100.0 AS cart_total_price,
        items_json
    FROM {{ ref('brz_orders') }}
    WHERE date_paid IS NOT NULL
      AND status IN (1, 2)  -- Only successfully paid orders
),

cart_items AS (
    SELECT
        order_id,
        customer_id,
        date_created,
        cart_total_price,
        JSON_EXTRACT_ARRAY(items_json, '$.cart') AS cart_array
    FROM order_base
),

unnested_items AS (
    SELECT
        order_id,
        customer_id,
        date_created,
        cart_total_price,
        JSON_EXTRACT_SCALAR(item, '$.name') AS item_name,
        JSON_EXTRACT_SCALAR(item, '$.category') AS item_category,
        CAST(JSON_EXTRACT_SCALAR(item, '$.price') AS FLOAT64) / 100.0 AS item_price,
        ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY JSON_EXTRACT_SCALAR(item, '$.name')) AS item_sequence
    FROM cart_items,
    UNNEST(cart_array) AS item
)

SELECT
    -- Composite primary key
    CONCAT(CAST(order_id AS STRING), '-', CAST(item_sequence AS STRING)) AS item_tracking_id,

    -- Order info
    order_id,
    customer_id,
    date_created AS order_time,
    ROUND(cart_total_price, 2) AS order_price,

    -- Item info
    item_name AS item,
    item_category AS category,
    1 AS quantity,  -- Each row represents 1 item
    ROUND(item_price, 2) AS item_price

FROM unnested_items
ORDER BY order_time, order_id, item_tracking_id
