with 

orders_extracted as (
    select *
    from {{ ref("int_order_details_extracted") }}
),

paid_orders as (
    select
        order_id,
        customer_id,
        cart_total_price as order_price,
        item_id as item_tracking_id,
        item_name as item,
        item_category as category,
        item_quantity as quantity,
        item_price,
        date_created as order_time
    from orders_extracted
    /* Keep the successfully paid transactions only */
    where date_paid IS NOT NULL AND status IN (1, 2)
    ORDER BY order_time, order_id, item_tracking_id
)

select distinct * from paid_orders