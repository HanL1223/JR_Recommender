-- extracting item details from the JSON array in BigQuery, with price and quantity
-- adjustment, including order time details, and adding a hashed ID
select
    farm_fingerprint(
        concat(extracted.item_name, coalesce(extracted.item_category, 'NoCategory'))
    ) as item_hash_id,
    extracted.*

from
    (
        select
            o.order_id,
            -- Directly extracting item-level attributes
            cast(json_extract_scalar(item, '$.name') as string) as item_name,
            cast(json_extract_scalar(item, '$.category') as string) as item_category,
            -- Extracting the quantity, assuming 1 when null
            ifnull(
                safe_cast(json_extract_scalar(item, '$.quantity') as int64), 1
            ) as item_quantity,
            -- Adjusting the price from cents to dollars and rounding to 2 decimal
            -- places
            round(
                cast(json_extract_scalar(item, '$.price') as float64) / 100, 2
            ) as item_price,
            -- Extracting order time details
            cast(
                json_extract_scalar(o.items_json, '$.order_time') as string
            ) as order_time,
            cast(
                json_extract_scalar(o.items_json, '$.order_time_verbose') as string
            ) as order_time_verbose
        from {{ ref("stg_orders") }} as o
        cross join unnest(json_extract_array(o.items_json, '$.cart')) as item
    ) as extracted
