-- models/dim_products.sql
select distinct item_name, item_category, avg(item_price) as average_price  -- Since item_price might vary, consider average or latest.
from {{ ref("item_details") }}
group by item_name, item_category
