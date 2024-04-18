-- models/dim_customers.sql
with
    customer_orders as (
        select
            customer_id,
            count(distinct order_id) as total_orders,
            round(avg(total), 2) as avg_order_value,
            max(date_created) as last_order_date
        from {{ ref("fact_orders") }}
        group by customer_id
    ),

    customer_items as (
        select customer_id, item_name, count(*) as item_count
        from {{ ref("fact_product_sales") }}
        group by customer_id, item_name
    ),

    ranked_items as (
        select
            customer_id,
            item_name,
            rank() over (partition by customer_id order by item_count desc) as item_rank
        from customer_items
    ),

    most_frequent_items as (
        select customer_id, item_name as most_frequent_item
        from ranked_items
        where item_rank = 1
    )

select
    c.customer_id,
    o.total_orders,
    o.avg_order_value,
    o.last_order_date,
    m.most_frequent_item
from {{ ref("customer") }} c
left join customer_orders o on c.customer_id = o.customer_id
left join most_frequent_items m on c.customer_id = m.customer_id
