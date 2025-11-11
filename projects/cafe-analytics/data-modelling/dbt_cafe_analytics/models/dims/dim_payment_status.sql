with

orders_extracted as (
    select *
    from {{ ref('int_order_details_extracted') }}
),

payment_status as (
    select distinct
        status,
        status_description
    from orders_extracted
)

select * from payment_status