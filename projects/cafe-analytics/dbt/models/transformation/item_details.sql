-- extracting item details from the JSON array in BigQuery, with price adjustment
select
    o.order_id,
    -- Directly extracting item-level attributes
    cast(json_extract_scalar(item, '$.name') as string) as item_name,
    cast(json_extract_scalar(item, '$.category') as string) as item_category,
    -- Adjusting the price from cents to dollars and rounding to 2 decimal places
    round(cast(json_extract_scalar(item, '$.price') as float64) / 100, 2) as item_price
from {{ ref("stg_orders") }} as o
cross join unnest(json_extract_array(o.items_json, '$.cart')) as item
