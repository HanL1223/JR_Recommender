select
    items.order_id,
    items.item_name,
    -- Replace item_hash_id with order_item_ranking
    concat(
        cast(items.order_id as string), '-', cast(item_ranks.item_rank as string)
    ) as order_item_ranking,
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
-- Assuming item_ranks is a derived table or CTE that contains order_id, item_name,
-- and item_rank
join
    {{ ref("item_details") }} as item_ranks
    on items.order_id = item_ranks.order_id
    and items.item_name = item_ranks.item_name
cross join unnest(json_extract_array(items.cart_item, '$.options')) as option
