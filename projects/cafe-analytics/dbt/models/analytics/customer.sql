-- customer purchase patterns
-- Analyzing how often customers place orders and their average spending.
select
    customer_id,
    count(distinct order_id) as total_orders,
    round(avg(total), 2) as avg_order_value
from {{ ref("order_details") }}
group by customer_id
order by total_orders desc
