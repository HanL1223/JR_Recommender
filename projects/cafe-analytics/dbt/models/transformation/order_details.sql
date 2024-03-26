-- extracts general order details along with some top-level JSON attributes.
select
    order_id,
    customer_id,
    ip_addr,
    date_created,
    date_paid,
    -- Adjusting the total price from cents to dollars and rounding to 2 decimal places
    round(total / 100, 2) as total,
    status,
    -- Extracting top-level JSON attributes
    cast(json_extract_scalar(items_json, '$.cart_size') as int64) as cart_size,
    -- Adjusting the cart surcharge from cents to dollars and rounding to 2 decimal
    -- places
    round(
        cast(json_extract_scalar(items_json, '$.cart_surcharge') as float64) / 100, 2
    ) as cart_surcharge,
    -- Adjusting the cart total price from cents to dollars and rounding to 2 decimal
    -- places
    round(
        cast(json_extract_scalar(items_json, '$.cart_total_price') as float64) / 100, 2
    ) as cart_total_price,
    -- Adjusting the GST from cents to dollars and rounding to 2 decimal places
    round(
        cast(json_extract_scalar(items_json, '$.cart_gst') as float64) / 100, 2
    ) as cart_gst
from {{ ref("stg_orders") }}
