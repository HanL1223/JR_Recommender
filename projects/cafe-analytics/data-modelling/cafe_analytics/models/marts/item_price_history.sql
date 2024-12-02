with

item_options as (
    select *
    from {{ ref("dim_item_options") }}
),

order_dates as (
    select distinct
        order_id,
        date_created
    from {{ ref("fct_order_details") }}
),

order_items as (
    select
        order_item_id,
        item_name,
        order_id
    from {{ ref("fct_product_sales") }}
),

item_prices as (
    select distinct
        order_items.item_name,
        item_options.option_value as size_or_flavour,
        order_items.order_id,
        item_options.option_price as unit_price
    from item_options
    right join order_items
    on item_options.order_item_id = order_items.order_item_id
    where item_options.option_name = 'size'
),

item_price_history as (
    select 
        item_prices.item_name,
        item_prices.size_or_flavour,
        item_prices.unit_price,
        date(
            min(order_dates.date_created)
        ) as starting_date,
    from item_prices
    left join order_dates
    on item_prices.order_id = order_dates.order_id
    group by 1, 2, 3
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
    starting_date, unit_price