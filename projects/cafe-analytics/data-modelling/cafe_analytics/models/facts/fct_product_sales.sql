with

orders_extracted as (
    select *
    from {{ ref('int_order_details_extracted') }}
),

product_sales as (
    select
        order_item_id,
        item_name,
        order_id,
        round(
            item_price / item_quantity,
            2
        ) as price,
        item_quantity as quantity,
        item_price as item_sales,
        date_created
    from orders_extracted
)

select * 
from product_sales
order by order_item_id