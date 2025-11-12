with

order_item_options as (
    select *
    from {{ ref("fct_order_item_options") }}
),

item_options as (
    select
        item_name,
        option_name,
        option_value,
        option_price,
        date(min(date_created)) as start_date
    from order_item_options
    group by 1, 2, 3, 4
),

item_options_enddates as (
    select
        *,
        lead(start_date) over (
            partition by item_name, option_name, option_value
            order by start_date
        ) as end_date
    from item_options
),

item_options_indexed as (
    select
        {{ 
            dbt_utils.generate_surrogate_key(
                [
                    'item_name',
                    'option_name',
                    'option_value',
                    'option_price'
                ]
            ) 
        }} as item_option_id,
        *
    from item_options_enddates
)

select *
from item_options_indexed
order by 
    item_name, option_name desc, option_value, 
    start_date, option_price