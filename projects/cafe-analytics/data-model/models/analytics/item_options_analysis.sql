-- analytics/item_options_analysis.sql
-- Understanding popular item customizations
select
    item_name,
    option_name,
    option_value,
    count(*) as count,
    round(avg(option_price), 2) as avg_option_price
from {{ ref("item_options") }}
group by item_name, option_name, option_value
order by item_name, count desc
