-- Import sources
with
    source_data as (select * from {{ source("cafe", "cafe-sales") }}),
    parsed_items as (
        select
            order_id,
            customer_id,
            ip_addr,
            date_created,
            date_paid,
            total,
            status,
            cast(json_extract(items, '$.cart_size') as int64) as cart_size,
            cast(json_extract(items, '$.cart_surcharge') as float64) as cart_surcharge,
            cast(
                json_extract(items, '$.cart_total_price') as float64
            ) as cart_total_price
        from source_data
    ),

    final_result as (
        select
            customer_id,
            count(order_id) as number_of_orders,
            avg(cart_size) as avg_cart_size,
            sum(cart_surcharge) as total_cart_surcharge,
            sum(cart_total_price) as total_sales,
            avg(total) as average_order_value
        from parsed_items
        group by customer_id
    )

select *
from final_result  -- Use final_result CTE instead of parsed_items

