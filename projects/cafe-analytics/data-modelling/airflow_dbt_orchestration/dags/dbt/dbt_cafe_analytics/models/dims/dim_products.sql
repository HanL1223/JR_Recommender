with

orders_extracted as (
    select *
    from {{ ref('int_order_details_extracted') }}
),

products as (
    select distinct
        item_name,
        item_category
    from orders_extracted
)

select * from products