with 

options as (
    select
        order_item_id,
        order_id,
        item_name,
        option_name,
        option_value,
        option_price
    from {{ ref('int_order_item_options_extracted') }}
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
    from options
)

select * 
from options_indexed
order by 
    order_item_id, option_name desc, 
    option_value, option_price