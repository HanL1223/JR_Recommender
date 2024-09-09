-- customer purchase patterns, including total orders, average order value, last order
-- date, visits last year, and most frequently ordered item
with
    order_details_agg as (
        select
            customer_id,
            count(distinct order_id) as total_orders,
            round(avg(total), 2) as avg_order_value,
            max(date_paid) as last_order_date
        from {{ ref("order_details") }}
        group by customer_id
    ),
    visits_last_year as (
        select customer_id, count(distinct order_id) as visits_last_year
        from {{ ref("order_details") }}
        -- Convert DATE_SUB result to TIMESTAMP for comparison
        where
            timestamp(date_paid) >= timestamp(date_sub(current_date(), interval 1 year))
        group by customer_id
    ),
    item_counts as (
        select o.customer_id, i.item_name, count(*) as order_count
        from {{ ref("item_details") }} i
        join {{ ref("order_details") }} o on i.order_id = o.order_id
        group by o.customer_id, i.item_name
    ),
    ranked_items as (
        select
            customer_id,
            item_name,
            rank() over (partition by customer_id order by order_count desc) as rank
        from item_counts
    ),
    most_frequent_items as (
        select customer_id, item_name as most_frequent_item
        from ranked_items
        where rank = 1
    )
select
    oda.customer_id,
    oda.total_orders,
    oda.avg_order_value,
    oda.last_order_date,
    vly.visits_last_year,
    mfi.most_frequent_item
from order_details_agg oda
left join visits_last_year vly on oda.customer_id = vly.customer_id
left join most_frequent_items mfi on oda.customer_id = mfi.customer_id
order by oda.total_orders desc
