-- total sales revenue
-- aggregate total sales by item, taking into account the adjusted item prices.
select item_name, sum(item_price) as total_sales, count(order_id) as total_orders
from {{ ref("item_details") }}
group by item_name
order by total_sales desc
