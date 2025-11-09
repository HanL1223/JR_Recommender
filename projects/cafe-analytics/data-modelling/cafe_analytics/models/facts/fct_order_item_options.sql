with 

orders_extracted as (
    select *
    from {{ ref('int_order_details_extracted') }}
),

options_extracted as (
    select
        orders_extracted.order_item_id,
        orders_extracted.order_id,
        orders_extracted.item_name,

        cast(
            json_extract_scalar(item_options, '$.name') as string
        ) as option_name,

        cast(
            json_extract_scalar(item_options, '$.value') as string
        ) as option_value,

        round(
            cast(
                json_extract_scalar(item_options, '$.price') as float64
            ) / 100,
            2
        ) as option_price,

        orders_extracted.date_created

    from orders_extracted
    cross join unnest(
        json_extract_array(orders_extracted.cart_items, '$.options')
    ) as item_options
),

options_indexed as (
    select
        {{ 
            dbt_utils.generate_surrogate_key(
                [
                    'order_item_id',
                    'order_id',
                    'item_name',
                    'option_name',
                    'option_value',
                    'option_price'
                ]
            ) 
        }} as order_item_option_id,
        *
    from options_extracted
)

select * 
from options_indexed
order by 
    order_item_id, option_name desc, 
    option_value, option_price