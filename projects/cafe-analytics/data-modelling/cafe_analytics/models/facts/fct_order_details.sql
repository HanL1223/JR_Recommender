with

orders_extracted as (
    select *
    from {{ ref('int_order_details_extracted') }}
),

order_details as (
    select
        order_id,
        customer_id,
        date_created,
        date_paid,
        order_time,
        order_time_verbose,
        total,
        status,
        cart_size,
        cart_surcharge,
        cart_total_price,
        cart_gst
    from orders_extracted
)

select distinct * 
from order_details
order by order_id

