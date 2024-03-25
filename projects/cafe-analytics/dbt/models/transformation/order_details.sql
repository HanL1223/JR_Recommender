-- extracts general order details along with some top-level JSON attributes.
select
    order_id,
    customer_id,
    ip_addr,
    date_created,
    date_paid,
    total,
    status,
    -- Extracting top-level JSON attributes
    cast(json_extract_scalar(items_json, '$.cart_size') as int64) as cart_size,
    cast(
        json_extract_scalar(items_json, '$.cart_surcharge') as float64
    ) as cart_surcharge,
    cast(
        json_extract_scalar(items_json, '$.cart_total_price') as float64
    ) as cart_total_price,
    cast(json_extract_scalar(items_json, '$.cart_gst') as float64) as cart_gst
from {{ ref("stg_orders") }}

