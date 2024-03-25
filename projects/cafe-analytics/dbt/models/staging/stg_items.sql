-- stg_items.sql
with
    raw_data as (select order_id, items from {{ source("cafe", "cafe-sales") }}),
    unpacked_items as (
        select
            order_id,
            cast(json_extract(items, '$.cart_size') as int64) as cart_size,
            cast(json_extract(items, '$.cart_surcharge') as float64) as cart_surcharge,
            cast(
                json_extract(items, '$.cart_total_before_discounts') as float64
            ) as total_before_discounts
        from raw_data
    )

select *
from unpacked_items
