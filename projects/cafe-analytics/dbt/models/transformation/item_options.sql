-- Extracting item options with price adjustments from cents to dollars directly from
-- stg_orders
select
    items.order_id,
    items.item_name,
    -- Including the item_hash_id to build the relationship
    items.item_hash_id,
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
            s.order_id,
            json_extract_scalar(cart_item, '$.name') as item_name,
            -- Repeating the hash ID calculation as done in item_details for consistency
            farm_fingerprint(
                concat(
                    json_extract_scalar(cart_item, '$.name'),
                    json_extract_scalar(cart_item, '$.category')
                )
            ) as item_hash_id,
            cart_item
        from {{ ref("stg_orders") }} s
        cross join unnest(json_extract_array(s.items_json, '$.cart')) as cart_item
    ) as items
cross join unnest(json_extract_array(items.cart_item, '$.options')) as option
