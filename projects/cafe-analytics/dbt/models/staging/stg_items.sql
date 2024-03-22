-- stg_items.sql
with
    raw_data as (select order_id, items from {{ source("cafe", "cafe-sales") }}),
    unpacked_items as (
        select
            order_id,
            cast(json_extract_path_text(items, 'cart_size') as integer) as cart_size,
            cast(
                json_extract_path_text(items, 'cart_surcharge') as decimal(10, 2)
            ) as cart_surcharge,
            cast(
                json_extract_path_text(items, 'cart_total_before_discounts') as decimal(
                    10, 2
                )
            ) as total_before_discounts
        from raw_data
    )

select *
from unpacked_items
