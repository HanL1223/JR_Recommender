-- Extracting item options with price adjustments from cents to dollars directly from
-- stg_orders
select
    order_id,
    item_name,
    -- Extracting options using a cross join similar to item_details but targeting
    -- options
    cast(json_extract_scalar(option, '$.name') as string) as option_name,
    cast(json_extract_scalar(option, '$.value') as string) as option_value,
    round(
        cast(json_extract_scalar(option, '$.price') as float64) / 100, 2
    ) as option_price
from
    (
        select
            s.order_id, json_extract_scalar(cart_item, '$.name') as item_name, cart_item
        from {{ ref("stg_orders") }} s
        cross join unnest(json_extract_array(s.items_json, '$.cart')) as cart_item
    ) as items
cross join unnest(json_extract_array(items.cart_item, '$.options')) as option
