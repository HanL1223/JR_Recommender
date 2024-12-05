with

paid_orders as (
    select *
    from {{ ref("paid_order_details") }}
),

cust_orders as (
    select distinct
        customer_id,
        order_time,
        order_id
    from paid_orders
    order by customer_id, order_time, order_id
),

cust_next_orders as (
    select 
        customer_id, 
        order_time,
        lead(order_time) over (
            partition by customer_id 
            order by order_time, order_id
        ) as next_order_time,
        order_id
    from cust_orders
    order by customer_id, order_time, order_id
),

order_intervals as (
  select 
    *,
    timestamp_diff(
        next_order_time, order_time, day
    ) as purchase_interval_days
  from cust_next_orders
),

interval_mean_std as (
  select 
    customer_id,
    avg(purchase_interval_days) as interval_mean,
    stddev(purchase_interval_days) as interval_std,
  from order_intervals
  group by customer_id
  order by customer_id
),

churn_thresholds as (
  select 
    customer_id, 
    interval_mean + 1 * interval_std as churn_threshold
  from interval_mean_std
),

churn_threshold_all as (
  select 
    avg(purchase_interval_days) 
    + 1 * stddev(purchase_interval_days) 
    as churn_threshold_all
  from order_intervals
),

thresholds as (
    select 
        churn_thresholds.customer_id,
        coalesce(
            churn_thresholds.churn_threshold,
            churn_threshold_all.churn_threshold_all
        ) as churn_threshold
    from churn_thresholds
    cross join churn_threshold_all
)

select * 
from thresholds 
order by customer_id
