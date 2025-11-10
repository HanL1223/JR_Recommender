{{
    config(
        tags=["table", "full replace"]    
    )
}}

with

item_options as (
    select *
    from {{ ref("dim_item_options") }}
),

item_price_history as (
    select 
        item_name,
        option_value as size_or_flavour,
        option_price as unit_price,
        start_date,
        end_date
    from item_options
    where option_name = 'size'
),

item_price_history_indexed as (
    select
        {{ 
            dbt_utils.generate_surrogate_key(
                [
                    'item_name',
                    'size_or_flavour',
                    'unit_price'
                ]
            ) 
        }} as item_price_id,
        *
    from item_price_history
)

select * 
from item_price_history_indexed
order by 
    item_name, size_or_flavour, 
    start_date, unit_price