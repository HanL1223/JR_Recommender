with

orders_extracted as (
    select *
    from {{ ref('int_order_details_extracted') }}
),

cust_orders as (
    select distinct
        customer_id,
        order_id,
        total,
        date_created
    from orders_extracted
),

order_items as (
    select distinct
        customer_id,
        order_id,
        item_name
    from orders_extracted
),

order_details_agg as (
    select
        customer_id,
        count(order_id) as total_orders,
        round(avg(total), 2) as avg_order_value,
        max(date_created) as last_order_date
    from cust_orders
    group by 1
),

visits_last_year as (
    select
        customer_id,
        count(order_id) as visits_last_year
    from cust_orders
    -- Convert DATE_SUB result to TIMESTAMP for comparison
    where 
        timestamp(date_created) >= timestamp(
            date_sub(current_date(), interval 1 year)
        )
    group by 1
),

item_purchase_counts as (
    select
        customer_id,
        item_name,
        count(*) as purchase_count
    from order_items
    group by 1, 2
),

item_purchase_counts_ranked as (
    select
        dense_rank() over (
            partition by customer_id
            order by purchase_count desc
        ) as rank,
        *
    from item_purchase_counts
),

most_frequent_items as (
    select
        customer_id,
        string_agg(
            item_name 
            order by item_name
        ) as most_frequent_item
    from item_purchase_counts_ranked
    where rank = 1
    group by customer_id
),

final as (
    select
        order_details_agg.customer_id,
        order_details_agg.total_orders,
        order_details_agg.avg_order_value,
        order_details_agg.last_order_date,
        ifnull(
            visits_last_year.visits_last_year, 0
        ) as visits_last_year,
        most_frequent_items.most_frequent_item
    from order_details_agg
    left join visits_last_year 
    on order_details_agg.customer_id = visits_last_year.customer_id
    left join most_frequent_items 
    on order_details_agg.customer_id = most_frequent_items.customer_id
    order by order_details_agg.total_orders desc
)

select * from final
    
    
    


